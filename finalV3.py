import streamlit as st
from streamlit_mic_recorder import mic_recorder  # For recording audio
import numpy as np
import scipy.io.wavfile as wav
from speech_recognition import Recognizer, AudioFile
import matplotlib.pyplot as plt
import tempfile
import os
import noisereduce as nr
import subprocess
import mimetypes
import cohere

# Initialize Cohere client
co = cohere.Client('X9Pnnul4KJFCncqCKcljES4qpglWShtrfvlRujAG')  # Replace with your Cohere API key

# Noise reduction function using FFT
def reduce_noise_fft(data, rate):
    chunk_size = 1024
    num_chunks = len(data) // chunk_size
    denoised_data = []
    for i in range(num_chunks):
        chunk = data[i * chunk_size : (i + 1) * chunk_size]
        fft_data = np.fft.fft(chunk)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        threshold = np.median(magnitude) * 1.5
        magnitude[magnitude < threshold] = 0
        cleaned_fft = magnitude * np.exp(1j * phase)
        cleaned_chunk = np.fft.ifft(cleaned_fft).real
        denoised_data.append(cleaned_chunk)
    return np.concatenate(denoised_data).astype(np.int16)

# Advanced noise reduction using noisereduce
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

# Text generation function with animation
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

# Streamlit app
st.title("🎤 Voice-to-Text with Noise Filtering & AI-Powered Content Generation")
st.write("Record or upload a `.wav` file, and this app will filter out noise, transcribe the audio, and generate additional content using Cohere's API.")

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Record audio using the microphone or upload a `.wav` file.
2. The app will filter out noise and display the spectrograms.
3. The transcribed text will appear below.
4. Additional content will be generated using Cohere's API.
5. You can listen to the original and denoised audio.
""")

# Option to record or upload audio
option = st.radio("Choose an option:", ["Record Audio", "Upload Audio"])

audio_bytes = None
if option == "Record Audio":
    st.subheader("🎤 Record Audio")
    audio = mic_recorder(start_prompt="🎙 Click to record", stop_prompt="🛑 Stop recording")
    if audio:
        audio_bytes = audio["bytes"]
        st.success("Recording complete!")
        st.audio(audio_bytes, format="audio/wav")
elif option == "Upload Audio":
    st.subheader("📂 Upload Audio")
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("File size exceeds the limit of 10 MB.")
        else:
            audio_bytes = uploaded_file.getvalue()
            st.audio(audio_bytes, format="audio/wav")

# Process audio if available
if audio_bytes:
    # Save audio bytes to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(audio_bytes)
    temp_file.close()

    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(temp_file.name)
    st.write(f"Detected MIME type: {mime_type}")

    # Convert to WAV if not a supported format
    converted_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        rate, data = wav.read(temp_file.name)
    except ValueError:
        st.warning("Audio format not supported. Converting to WAV...")
        subprocess.run(["ffmpeg", "-y", "-i", temp_file.name, "-ar", "16000", "-ac", "1", "-f", "wav", converted_file.name])
        rate, data = wav.read(converted_file.name)

    # Handle stereo audio
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Noise reduction
    denoised_data = reduce_noise_advanced(data, rate)

    # Save denoised audio to a temporary file
    denoised_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(denoised_temp_file.name, rate, denoised_data)

    # Perform speech recognition with animation
    recognizer = Recognizer()
    try:
        with st.spinner("🗣️ Transcribing audio using Google Speech-to-Text... Please wait!"):
            with AudioFile(denoised_temp_file.name) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        if text.strip():
            st.subheader("📝 Transcribed Text")
            st.write(text)

            # Generate text using Cohere API
            generated_text = generate_text(text)
            st.subheader("🤖 Generated Text from Cohere API")
            st.write(generated_text)
        else:
            st.error("Transcription was empty, unable to generate text.")
    except Exception as e:
        st.error(f"An error occurred during transcription: {str(e)}")

    # Display spectrograms
    st.subheader("📊 Original Audio Spectrogram")
    display_spectrogram(data, rate, "Original Audio Spectrogram")
    st.subheader("📊 Denoised Audio Spectrogram")
    display_spectrogram(denoised_data, rate, "Denoised Audio Spectrogram")

    # Display waveforms
    st.subheader("📈 Original Audio Waveform")
    display_waveform(data, "Original Audio Waveform")
    st.subheader("📈 Denoised Audio Waveform")
    display_waveform(denoised_data, "Denoised Audio Waveform")

    # Play original and denoised audio
    st.subheader("🎧 Listen to the Original Audio")
    st.audio(audio_bytes, format="audio/wav")
    st.subheader("🎧 Listen to the Denoised Audio")
    st.audio(denoised_temp_file.name, format="audio/wav")

    # Download denoised audio
    with open(denoised_temp_file.name, "rb") as f:
        st.download_button(
            label="📥 Download Denoised Audio",
            data=f,
            file_name="denoised_audio.wav",
            mime="audio/wav"
        )

    # Clean up temporary files
    os.unlink(temp_file.name)
    os.unlink(denoised_temp_file.name)