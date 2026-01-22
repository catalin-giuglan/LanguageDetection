#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE = 16000
TARGET_DURATION = 3.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
EPS = 1e-9

def pad_or_trim(w, target):
    if w.shape[1] > target:
        return w[:, :target]
    return torch.nn.functional.pad(w, (0, target - w.shape[1]))

# RMS normalization to target dB
def normalize_rms(waveform, target_db=-20.0):
    rms = torch.sqrt(torch.mean(waveform ** 2) + EPS)
    rms_db = 20.0 * torch.log10(rms + EPS)
    gain_db = target_db - rms_db
    gain = 10.0 ** (gain_db / 20.0)
    return waveform * gain

# trims empty/silent parts from the audio file
def trim_silence(waveform, sample_rate, top_db=40.0, frame_ms=30, hop_ms=10):
    # convert to mono if not already
    if waveform.ndim > 1:
        w = waveform.mean(dim=0, keepdim=True)
    else:
        w = waveform

    frame_len = int(sample_rate * frame_ms / 1000)
    hop_len = int(sample_rate * hop_ms / 1000)
    if frame_len <= 0:
        frame_len = 1
    if hop_len <= 0:
        hop_len = 1

    # pad to fit frames
    pad_len = (frame_len - (w.shape[1] - frame_len) % hop_len) % hop_len
    if pad_len > 0:
        w = torch.nn.functional.pad(w, (0, pad_len))

    # frame-wise RMS
    frames = w.unfold(1, frame_len, hop_len)  # [1, n_frames, frame_len]
    # compute RMS per frame
    rms_frames = torch.sqrt(torch.mean(frames ** 2, dim=2) + EPS).squeeze(0)  # [n_frames]
    max_rms = torch.max(rms_frames) + EPS
    thresh = max_rms * (10 ** (-top_db / 20.0))

    # find first & last frame above threshold
    mask = (rms_frames >= thresh)
    if mask.sum() == 0:
        # nothing above threshold: return original (or last frame)
        return w[:, : w.shape[1] - pad_len] if pad_len > 0 else w
    first = torch.nonzero(mask, as_tuple=False)[0].item()
    last = torch.nonzero(mask, as_tuple=False)[-1].item()

    start = first * hop_len
    end = min((last * hop_len) + frame_len, w.shape[1])
    trimmed = w[:, start:end]
    return trimmed

# same as in train.py
class CRNN(nn.Module):
    def __init__(self, n_mels=64, n_classes=10, gru_hidden=256, bidirectional=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        feat_per_frame = 128 * (n_mels // 4)
        self.gru = nn.GRU(input_size=feat_per_frame,
                          hidden_size=gru_hidden,
                          batch_first=True,
                          bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(gru_hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x):
        x = self.conv(x)
        b, c, m, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, c * m)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        h = self.dropout(h)
        return self.classifier(h)

def load_model(model_path, config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    langs = cfg["langs"]
    num_classes = len(langs)

    model = CRNN(n_mels=N_MELS, n_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, langs

# preprocess audio file to match training conditions
def preprocess_audio(wav_file, target_sr=SAMPLE_RATE, do_trim=True, trim_top_db=40.0, target_db=-20.0):
    w, sr = torchaudio.load(wav_file)  # [channels, T]
    # mono
    if w.shape[0] > 1:
        w = w.mean(dim=0, keepdim=True)

    # resample if needed
    if sr != target_sr:
        w = T.Resample(sr, target_sr)(w)
        sr = target_sr

    # trim silence (aggressive for voice messages)
    if do_trim:
        try:
            w = trim_silence(w, sr, top_db=trim_top_db, frame_ms=30, hop_ms=10)
        except Exception:
            pass

    # normalize RMS to target_db
    w = normalize_rms(w, target_db=target_db)

    # pad/trim to exact length used in training
    w = pad_or_trim(w, TARGET_SAMPLES)

    return w, sr

def predict(model, langs, wav_file, save_chart_path=None):
    # Preprocess audio to match training pipeline
    w, sr = preprocess_audio(wav_file, target_sr=SAMPLE_RATE, do_trim=True, trim_top_db=40.0, target_db=-20.0)

    # Create mel spectrogram (same params as training)
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    db_transform = T.AmplitudeToDB()

    mel = mel_transform(w)   # [1, n_mels, time]
    mel = db_transform(mel)

    # Add batch dimension: [1, 1, n_mels, time]
    mel = mel.unsqueeze(0)

    # Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    mel = mel.to(device)

    with torch.no_grad():
        # gets the % for each language
        out = model(mel)
        probs = torch.softmax(out, dim=1)
        probabilities = probs[0].cpu().numpy()
        # returns the language with the highest %
        idx = int(out.argmax(1).item())

    predicted_lang = langs[idx]
    return predicted_lang, probabilities

# plot pie chart of language probabilities
def plot_language_pie_chart(langs, probabilities, predicted_lang, wav_file, save_path=None):
    percentages = probabilities * 100
    colors = plt.cm.Set3(np.linspace(0, 1, len(langs)))
    explode = [0.1 if lang == predicted_lang else 0 for lang in langs]

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        percentages,
        labels=langs,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=True,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )

    for i, lang in enumerate(langs):
        if lang == predicted_lang:
            autotexts[i].set_color('white')
            autotexts[i].set_fontsize(13)

    audio_filename = os.path.basename(wav_file)
    ax.set_title(f'Language Prediction Probabilities\n{audio_filename}\n\nPredicted: {predicted_lang.upper()}',
                 fontsize=14, weight='bold', pad=20)
    ax.axis('equal')
    legend_labels = [f'{lang}: {pct:.2f}%' for lang, pct in zip(langs, percentages)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pie chart saved to: {save_path}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_path> <wav_file> [--save-chart <output_path>]")
        sys.exit(1)

    model_path = sys.argv[1]
    wav_file = sys.argv[2]

    save_chart_path = None
    if len(sys.argv) >= 5 and sys.argv[3] == "--save-chart":
        save_chart_path = sys.argv[4]

    run_dir = os.path.dirname(model_path)
    config_path = os.path.join(run_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    if not os.path.exists(wav_file):
        print(f"Error: Audio file not found at {wav_file}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model, langs = load_model(model_path, config_path)
    print(f"Model loaded. Languages: {langs}")

    print(f"\nAnalyzing {wav_file}...")
    predicted_lang, probabilities = predict(model, langs, wav_file, save_chart_path)

    print(f"\n{'='*50}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"\nPredicted Language: {predicted_lang.upper()}")
    print(f"\nAll probabilities:")
    for lang, prob in zip(langs, probabilities):
        bar = '█' * int(prob * 50)
        print(f"  {lang:>10}: {prob*100:5.2f}% {bar}")
    print(f"{'='*50}\n")

    # Create pie chart
    if save_chart_path:
        plot_language_pie_chart(langs, probabilities, predicted_lang, wav_file, save_chart_path)
    else:
        audio_name = os.path.splitext(os.path.basename(wav_file))[0]
        default_chart_path = f"prediction_{audio_name}.png"
        plot_language_pie_chart(langs, probabilities, predicted_lang, wav_file, default_chart_path)
