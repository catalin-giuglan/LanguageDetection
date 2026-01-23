# Language Detection using CRNN

This project implements an automatic language detection system for short audio recordings using a **Convolutional Recurrent Neural Network (CRNN)** trained on Mel Spectrograms. 

The final model is integrated into a **Streamlit** web application for easy demonstration and usage.

## 📝 Overview

Spoken language identification is a critical component for virtual assistants and automated transcription platforms. 
This project focuses on identifying the language based on acoustic information (speech rhythm, pronunciation, frequency energy distribution) rather than understanding the spoken content.

The system is designed to classify languages from short audio samples (approx. **3 seconds**) with high accuracy.

## 🚀 Key Features

* **Multi-language Support:** Detects 9 distinct languages: German, English, Spanish, French, Italian, Japanese, Portuguese, Romanian, and Chinese.
* **Hybrid Architecture (CRNN):** Utilizes CNNs to extract local spectral features and RNNs to capture the temporal evolution of the speech signal.
* **Robust Preprocessing:** Includes volume normalization, silence removal, and duration standardization.
* **Interactive Web UI:** A Streamlit application that allows users to upload audio files, play them, and visualize prediction probabilities.

## 🛠️ System Architecture

### 1. Audio Preprocessing
To ensure model robustness, all input audio goes through a standardized pipeline:
* **Resampling:** All files are converted to **16 kHz**.
* **Silence Removal:** Stripping non-speech sequences to force the model to learn from actual vocal data.
* **RMS Normalization:** Uniforming the volume across recordings.
* **Duration Adjustment:** Trimming or padding audio to a fixed length of **3 seconds**.
* **Feature Extraction:** Generating Mel Spectrograms and converting them to the decibel scale.

### 2. Neural Network
The model is inspired by the work of Singh et al.:
* **Input:** Mel Spectrogram images.
* **CNN Layers:** Extract spatial features from the spectrograms.
* **RNN Layers:** Analyze the sequential nature of the data.
* **Training:** Trained for 30 epochs using `EarlyStopping` to ensure stability and prevent overfitting.

## 📊 Dataset

The model was trained on a subset of the **VoxLingua107** dataset. This dataset contains recordings from real-world contexts (YouTube), providing natural variations in accent, intonation, and acoustic quality.

## 📈 Performance

* **Accuracy:** The model achieves ~92% accuracy on the validation set.
* **F1 Score:** Consistently high across classes, indicating a good balance between precision and recall.
* **Confusion Matrix Analysis:**
    * Distinct languages like Japanese or Chinese are identified with very high precision.
    * Minor confusion exists between Romance languages (Italian, Spanish, Portuguese) due to phonetic and rhythmic similarities.

## 📚 References

1. G. Singh, S. Sharma, V. Kumar, M. Kaur, M. Baz, M. Masud, "Spoken Language Identification Using Deep Learning," Computational Intelligence and Neuroscience, 2021.
https://pmc.ncbi.nlm.nih.gov/articles/PMC8478554/

2. "VoxLingua107: A Large-Scale Multilingual Speech Dataset".
https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/
