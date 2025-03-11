import streamlit as st
from streamlit_mic_recorder import mic_recorder
import numpy as np
import scipy.io.wavfile as wav
from speech_recognition import Recognizer, AudioFile
import matplotlib.pyplot as plt
import tempfile
import os
import noisereduce as nr
import subprocess
import cohere

# Noise reduction using noisereduce
def reduce_noise_advanced(data, rate):
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    return reduced_noise.astype(np.int16)

# Display spectrogram
def display_spectrogram(data, rate, title):
    plt.figure(figsize=(10, 4))
    plt.specgram(data, Fs=rate, cmap='inferno')
    plt.title(title, fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)
    plt.colorbar(label="Intensity (dB)")
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(plt)
    plt.clf()

# Display waveform
def display_waveform(data, title):
    plt.figure(figsize=(10, 2))
    plt.plot(data)
    plt.title(title, fontsize=12)
    plt.xlabel("Samples", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(plt)
    plt.clf()

# Initialize Cohere client
co = cohere.Client('X9Pnnul4KJFCncqCKcljES4qpglWShtrfvlRujAG')

# Text generation function with loading animation
def generate_text(prompt):
    try:
        with st.spinner("🤖 Generating response from Cohere API... Please wait!"):
            response = co.generate(
                model='command-xlarge',
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )
            return response.generations[0].text
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Streamlit UI
st.title("🎤 Voice-to-Text with Noise Filtering & Cohere API")
st.write("Upload or record a `.wav` file. This app will transcribe the audio and generate additional content using Cohere's API.")

# Upload option
option = st.radio("Choose an option:", ["Record Audio", "Upload Audio"])

audio_bytes = None
if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file:
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes, format="audio/wav")

if audio_bytes:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(audio_bytes)
    temp_file.close()

    try:
        rate, data = wav.read(temp_file.name)
    except ValueError as e:
        st.warning(f"Audio format not supported. Converting to WAV... {e}")
        subprocess.run(["ffmpeg", "-y", "-i", temp_file.name, "-ar", "16000", "-ac", "1", "-f", "wav", temp_file.name])
        rate, data = wav.read(temp_file.name)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Noise reduction
    denoised_data = reduce_noise_advanced(data, rate)

    # Save denoised audio
    denoised_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(denoised_temp_file.name, rate, denoised_data)

    # Speech recognition
    recognizer = Recognizer()
    try:
        with AudioFile(denoised_temp_file.name) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)

        if text.strip():
            st.subheader("📝 Transcribed Text")
            st.write(text)

            # Generate response from Cohere with a loading animation
            generated_text = generate_text(text)

            st.subheader("🤖 Generated Text from Cohere API")
            st.write(generated_text)
        else:
            st.error("Transcription was empty, unable to generate text.")
    except Exception as e:
        st.error(f"An error occurred during transcription: {str(e)}")

    # Display audio visuals
    display_spectrogram(data, rate, "Original Audio Spectrogram")
    display_spectrogram(denoised_data, rate, "Denoised Audio Spectrogram")
    display_waveform(data, "Original Audio Waveform")
    display_waveform(denoised_data, "Denoised Audio Waveform")

    # Play audio
    st.audio(audio_bytes, format="audio/wav")
    st.audio(denoised_temp_file.name, format="audio/wav")

    # Clean up
    os.unlink(temp_file.name)
    os.unlink(denoised_temp_file.name)
