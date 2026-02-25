import streamlit as st
import librosa
import numpy as np
import pickle
import pandas as pd

# Load trained model
model, le = pickle.load(open("model.pkl", "rb"))

st.title("🐾 AI-Based Livestock Acoustic Emotion Analyzer")

st.write("Upload animal sound to analyze emotional state.")

audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

def extract_features(file):
    audio, sr = librosa.load(file, duration=3)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    features = np.hstack([mfcc, rms, zcr, spectral_centroid])
    return features

if audio_file is not None:
    st.audio(audio_file)

    features = extract_features(audio_file)
    probabilities = model.predict_proba([features])[0]

    labels = le.classes_

    result_df = pd.DataFrame({
        "Emotion": labels,
        "Confidence": probabilities
    })

    st.subheader("Emotion Analysis Result")
    st.bar_chart(result_df.set_index("Emotion"))

    predicted_label = labels[np.argmax(probabilities)]

    st.success(f"Predicted State: {predicted_label}")

    if predicted_label == "distress":
        st.error("⚠ Veterinary attention recommended.")
