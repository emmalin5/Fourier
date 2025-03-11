import streamlit as st
from speech_recognition import Recognizer, AudioFile
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
import tempfile
import os
import re
from jiwer import wer

# Preprocessing function for text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Transcription function using Google Speech-to-Text
def transcribe_audio(file_path):
    recognizer = Recognizer()
    try:
        with AudioFile(file_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except Exception as e:
        return f"Error during transcription: {str(e)}"

# Calculate transcription accuracy
def calculate_accuracy(ground_truth, transcribed_text):
    ground_truth = preprocess_text(ground_truth)
    transcribed_text = preprocess_text(transcribed_text)
    
    ground_truth_tokens = ground_truth.split()
    transcribed_tokens = transcribed_text.split()
    
    correct_words = sum(1 for gt, tt in zip(ground_truth_tokens, transcribed_tokens) if gt == tt)
    total_words = len(ground_truth_tokens)
    accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
    return accuracy

# FFT-based noise reduction using noisereduce
def reduce_noise_fft(data, rate):
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    return reduced_noise.astype(np.int16)

# Save audio to temporary file
def save_temp_wav(rate, data):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, rate, data)
    return temp_file.name

# Signal-to-Noise Ratio (SNR) calculation
def calculate_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    return snr

# Evaluate FFT-based noise reduction
def evaluate_fft_noise_reduction(noisy_audio_path, ground_truth):
    # Load noisy audio
    rate, data = wav.read(noisy_audio_path)
    
    # Transcribe noisy audio
    noisy_transcription = transcribe_audio(noisy_audio_path)
    noisy_accuracy = calculate_accuracy(ground_truth, noisy_transcription)
    
    # Apply FFT-based noise reduction
    denoised_data = reduce_noise_fft(data, rate)
    denoised_audio_path = save_temp_wav(rate, denoised_data)
    
    # Transcribe denoised audio
    denoised_transcription = transcribe_audio(denoised_audio_path)
    denoised_accuracy = calculate_accuracy(ground_truth, denoised_transcription)
    
    # Calculate SNR
    noise = data - denoised_data
    snr_before = calculate_snr(data, noise)
    snr_after = calculate_snr(denoised_data, noise)
    
    # Calculate Word Error Rate (WER)
    noisy_wer = wer(ground_truth, noisy_transcription)
    denoised_wer = wer(ground_truth, denoised_transcription)
    
    # Clean up temporary files
    os.unlink(denoised_audio_path)
    
    return {
        "noisy_accuracy": noisy_accuracy,
        "denoised_accuracy": denoised_accuracy,
        "snr_before": snr_before,
        "snr_after": snr_after,
        "noisy_wer": noisy_wer,
        "denoised_wer": denoised_wer
    }

# Streamlit app
st.title("🎤 FFT-Based Noise Reduction Evaluation")
st.write("Upload noisy audio files and evaluate the impact of FFT-based noise reduction on transcription accuracy.")

# Upload noisy audio file
uploaded_file = st.file_uploader("Upload a noisy .wav file", type=["wav"])
if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()

    # Ground truth input
    ground_truth = st.text_input("Enter the ground truth transcription:", "")
    
    if ground_truth.strip():
        # Evaluate FFT-based noise reduction
        results = evaluate_fft_noise_reduction(temp_file.name, ground_truth)
        
        # Display results
        st.subheader("Evaluation Results")
        st.write(f"Transcription Accuracy without Noise Reduction: {results['noisy_accuracy']:.2f}%")
        st.write(f"Transcription Accuracy with FFT-Based Noise Reduction: {results['denoised_accuracy']:.2f}%")
        st.write(f"Signal-to-Noise Ratio (SNR) before Noise Reduction: {results['snr_before']:.2f} dB")
        st.write(f"Signal-to-Noise Ratio (SNR) after Noise Reduction: {results['snr_after']:.2f} dB")
        st.write(f"Word Error Rate (WER) without Noise Reduction: {results['noisy_wer'] * 100:.2f}%")
        st.write(f"Word Error Rate (WER) with FFT-Based Noise Reduction: {results['denoised_wer'] * 100:.2f}%")
        
        # Play original and denoised audio
        st.subheader("🎧 Listen to the Original Noisy Audio")
        st.audio(uploaded_file.getvalue(), format="audio/wav")
        
        rate, data = wav.read(temp_file.name)
        denoised_data = reduce_noise_fft(data, rate)
        st.subheader("🎧 Listen to the Denoised Audio")
        st.audio(save_temp_wav(rate, denoised_data), format="audio/wav")
        
        # Clean up
        os.unlink(temp_file.name)