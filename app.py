import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from predict import load_model, predict, plot_language_pie_chart

MODEL_PATH = "models/run_20251204_004257/best_model.pt" 
CONFIG_PATH = "models/run_20251204_004257/config.json"

st.set_page_config(page_title="Language Detector AI", layout="centered")

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        return None, None
    return load_model(MODEL_PATH, CONFIG_PATH)

def get_mel_spectrogram_fig(audio_path):
    y, sr = librosa.load(audio_path)
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='viridis')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram')
    plt.tight_layout()
    
    return fig

st.title("Language Detector")
st.markdown("Incarca un fisier audio sau inregistreaza un mesaj vocal")

model, langs = get_model()

if model is None:
    st.error("Model not found")
else:
    file_recording = st.radio("Alege metoda de input:", ("Incarca fisier audio", "Inregistreaza mesaj vocal"))

    audio_file = None
    
    if file_recording == "Inregistreaza mesaj vocal":
        from audio_recorder_streamlit import audio_recorder
        
        st.info("Apasa butonul de mai jos pentru a incepe inregistrarea")
        audio_bytes = audio_recorder(
            pause_threshold=2.0,
            sample_rate=16000,
            text="Click pentru a inregistra",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            audio_file = audio_bytes
            
    else:
        uploaded_file = st.file_uploader("Alege un fisier WAV", type=["wav"])
        if uploaded_file is not None:
            st.audio(uploaded_file)
            audio_file = uploaded_file.getvalue()

    if audio_file is not None:
        if st.button("Analizeaza mesajul"):
            temp_filename = "temp_input.wav"
            chart_path = "temp_chart.png"
            
            with st.spinner('Se proceseaza...'):
                try:
                    with open(temp_filename, "wb") as f:
                        f.write(audio_file)
                    
                    predicted_lang, probabilities = predict(model, langs, temp_filename)
                    
                    st.success(f"Limba detectata: **{predicted_lang.upper()}**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Probabilitati")
                        probs_dict = {lang: float(prob) for lang, prob in zip(langs, probabilities)}
                        sorted_probs = dict(sorted(probs_dict.items(), key=lambda item: item[1], reverse=True))
                        st.table(sorted_probs)

                    with col2:
                        st.subheader("Distributie")
                        plot_language_pie_chart(langs, probabilities, predicted_lang, temp_filename, save_path=chart_path)
                        st.image(chart_path)
                    
                    st.markdown("---")
                    st.subheader("Analiză Spectrală")
                    
                    mel_fig = get_mel_spectrogram_fig(temp_filename)
                    st.pyplot(mel_fig)

                except Exception as e:
                    st.error(f"Eroare la procesare: {e}")
                
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                    if os.path.exists(chart_path):
                        os.remove(chart_path)