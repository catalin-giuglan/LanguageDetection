import os
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T

DEFAULT_TRAIN_DIR = "train"
DEFAULT_VALID_DIR = "validation"
DEFAULT_BASE_MODEL_DIR = "models"

TARGET_SR = 16000 # samples/second
TARGET_DURATION = 3.0 # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

DEFAULT_BATCH = 16
DEFAULT_EPOCHS = 30
DEFAULT_LR = 3e-4
DEFAULT_MAX_FILES = None   # use all
EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# makes the audio 3s long
# it either cuts a random segment from it or adds zero-padding 
def pad_or_trim(waveform, target_len):
    Tcur = waveform.shape[1]
    if Tcur > target_len:
        # random crop for variability
        start = random.randint(0, Tcur - target_len)
        return waveform[:, start:start + target_len]
    elif Tcur < target_len:
        pad_left = random.randint(0, target_len - Tcur)
        pad_right = target_len - Tcur - pad_left
        return nn.functional.pad(waveform, (pad_left, pad_right))
    else:
        return waveform

# audio augmentation, adds noise at random SNR
# helps robustness to background noise
def add_noise(waveform, snr_db_min=10.0, snr_db_max=20.0):
    rms = waveform.pow(2).mean().sqrt()
    snr_db = random.uniform(snr_db_min, snr_db_max)
    snr = 10 ** (snr_db / 20.0)
    noise_std = float(rms / snr)
    noise = torch.randn_like(waveform) * noise_std
    return waveform + noise

# speed perturbation by resampling the waveform
# changes both speed and pitch slightly
# simulates speaking faster/slower
def speed_perturb(waveform, orig_sr, low=0.9, high=1.1):
    rate = random.uniform(low, high)
    new_sr = int(round(orig_sr * rate))
    # resample to new_sr then back to TARGET_SR to simulate speed
    wav_sp = torchaudio.functional.resample(waveform, orig_sr, new_sr)
    wav_sp = torchaudio.functional.resample(wav_sp, new_sr, TARGET_SR)
    return wav_sp

# Dataset for language folders
class LangFolderDataset(Dataset):
    def __init__(self, root_dir, langs, max_files_per_lang=None, augment=False):
        self.items = []
        self.langs = langs
        for i, lang in enumerate(langs):
            folder = os.path.join(root_dir, lang)
            if not os.path.isdir(folder):
                continue
            files = [str(Path(folder) / f) for f in sorted(os.listdir(folder)) if f.lower().endswith(".wav")]
            if max_files_per_lang:
                files = files[:max_files_per_lang]
            for f in files:
                self.items.append((f, i))

        self.augment = augment
        self.mel_transform = T.MelSpectrogram(sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        self.db_transform = T.AmplitudeToDB()
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        waveform, sr = torchaudio.load(path)
        # stereo -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # optional speed perturb BEFORE resample
        if self.augment and random.random() < 0.2:
            try:
                waveform = speed_perturb(waveform, sr)
                sr = TARGET_SR
            except Exception:
                pass  # fallback to no speed perturb

        # resample to target sr if needed
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

        # pad/trim to target length
        waveform = pad_or_trim(waveform, TARGET_SAMPLES)

        # additive noise sometimes
        if self.augment and random.random() < 0.3:
            waveform = add_noise(waveform, snr_db_min=10, snr_db_max=20)

        return waveform, label

# collate: compute mel + optional specaugment + pad in time dimension
def collate_fn(batch, apply_spec_augment=False):
    waves = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # compute mel for each
    mel_list = []
    mel_transform = T.MelSpectrogram(sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    db_transform = T.AmplitudeToDB()
    for w in waves:
        mel = mel_transform(w)
        mel = db_transform(mel)
        mel_list.append(mel)

    # time dimension should be same because waveforms were pad_or_trim to fixed length, but be safe:
    max_t = max(m.shape[-1] for m in mel_list)
    mel_padded = []
    for m in mel_list:
        if m.shape[-1] < max_t:
            m = nn.functional.pad(m, (0, max_t - m.shape[-1]))
        if apply_spec_augment:
            if random.random() < 0.5:
                m = T.TimeMasking(time_mask_param=30)(m)
            if random.random() < 0.5:
                m = T.FrequencyMasking(freq_mask_param=15)(m)
        mel_padded.append(m)

    mel_tensor = torch.stack(mel_padded)   # [B, 1, n_mels, time]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return mel_tensor, labels_tensor

class CRNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=10, gru_hidden=256, bidirectional=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), # extracts patterns
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),   # reduce freq/time by 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),   # reduce again
        )
        assert n_mels % 4 == 0, "n_mels must be divisible by 4"
        feat_per_frame = 128 * (n_mels // 4)
        self.gru = nn.GRU(input_size=feat_per_frame,
                          hidden_size=gru_hidden,
                          batch_first=True,
                          bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(gru_hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x):
        # x: [B,1,n_mels,time]
        x = self.conv(x)  # [B, C, M', T']
        b, c, m, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T', C, M']
        x = x.view(b, t, c * m)                 # [B, T', feat]
        out, _ = self.gru(x)
        h = out[:, -1, :]                       # last timestep
        h = self.dropout(h)
        return self.classifier(h)

def save_history(run_dir, history):
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

def plot_history(run_dir, history):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], marker='o')
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, history["val_acc"], marker='o')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "metrics.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--valid_dir", default=DEFAULT_VALID_DIR)
    parser.add_argument("--langs", nargs="+", required=False, help="List of language folders OR omit to auto-detect from train_dir")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--runname", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aug", action="store_true", help="enable augmentations")
    args = parser.parse_args()

    set_seed()

    # autodetect langs if not provided
    if not args.langs:
        if not os.path.isdir(args.train_dir):
            raise RuntimeError(f"Train dir {args.train_dir} not found")
        candidates = sorted([d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))])
        args.langs = candidates
        print("[INFO] Auto-detected langs from train_dir:", args.langs)

    # filter languages that actually have wav files in train and valid
    valid_langs = []
    for lang in args.langs:
        train_folder = os.path.join(args.train_dir, lang)
        valid_folder = os.path.join(args.valid_dir, lang)
        train_has = os.path.isdir(train_folder) and any(f.lower().endswith(".wav") for f in os.listdir(train_folder))
        valid_has = os.path.isdir(valid_folder) and any(f.lower().endswith(".wav") for f in os.listdir(valid_folder))
        if train_has and valid_has:
            valid_langs.append(lang)
        else:
            print(f"[WARN] Skipping lang='{lang}' because train_has={train_has} valid_has={valid_has}")

    if len(valid_langs) == 0:
        raise RuntimeError("No valid languages found with WAV files in both train and validation folders")

    args.langs = valid_langs
    print("[INFO] Using languages:", args.langs)

    run_name = args.runname or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(DEFAULT_BASE_MODEL_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print("[INFO] Run folder:", run_dir)

    # save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # datasets and loaders
    train_ds = LangFolderDataset(args.train_dir, args.langs, max_files_per_lang=args.max_files, augment=args.aug)
    val_ds = LangFolderDataset(args.valid_dir, args.langs, max_files_per_lang=args.max_files, augment=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, apply_spec_augment=args.aug),
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, apply_spec_augment=False),
                            num_workers=args.workers, pin_memory=True)

    model = CRNN(n_mels=N_MELS, n_classes=len(args.langs)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=REDUCE_LR_PATIENCE)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for mel, labels in loop:
            mel = mel.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            loop.set_postfix(train_loss=running_loss / n_batches if n_batches else 0.0)

        train_loss = running_loss / max(1, n_batches)
        history["train_loss"].append(train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct = 0
        total = 0

        loopv = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False)
        with torch.no_grad():
            for mel, labels in loopv:
                mel = mel.to(device)
                labels = labels.to(device)
                logits = model(mel)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_batches += 1

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loopv.set_postfix(val_loss=val_loss / val_batches if val_batches else 0.0)

        val_loss = val_loss / max(1, val_batches)
        val_acc = correct / total if total > 0 else 0.0
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # scheduler & early stopping bookkeeping
        scheduler.step(val_loss)
        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            print(f"  [INFO] New best model saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1

        # periodic checkpoint
        torch.save(model.state_dict(), os.path.join(run_dir, f"epoch_{epoch}.pt"))

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"[INFO] Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break

    # final artifacts
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pt"))
    save_history(run_dir, history)
    plot_history(run_dir, history)
    print("[INFO] Training finished. Artifacts saved to:", run_dir)
    print("[INFO] Best validation accuracy:", best_val_acc)

if __name__ == "__main__":
    main()