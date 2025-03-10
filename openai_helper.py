# #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

# import streamlit as st
# from streamlit_mic_recorder import mic_recorder
# import openai
# import os

# # Set up OpenAI API key
# openai.api_key = os.getenv("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")
# #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
# st.title("🎤 Streamlit Audio Recorder + OpenAI Transcription")

# # Start recording
# audio = mic_recorder(start_prompt="🎙 Click to record", stop_prompt="🛑 Stop recording")

# if audio:
#     # Display recorded audio
#     st.audio(audio["bytes"], format="audio/wav")
    
#     # Download button for recorded audio
#     st.download_button(
#         label="📥 Download Audio",
#         data=audio["bytes"],
#         file_name="recorded_audio.wav",
#         mime="audio/wav"
#     )
    
#     # Transcribe the audio using OpenAI Whisper API
#     if st.button("📝 Transcribe Audio"):
#         with st.spinner("Transcribing..."):
#             try:
#                 # Save audio bytes to a temporary file
#                 with open("temp_audio.wav", "wb") as f:
#                     f.write(audio["bytes"])
                
#                 # Open the file and send it to OpenAI Whisper API
#                 with open("temp_audio.wav", "rb") as audio_file:
#                     transcript = openai.Audio.transcribe("whisper-1", audio_file)
                
#                 # Display the transcription
#                 st.success("Transcription complete!")
#                 st.text_area("Transcribed Text", value=transcript["text"], height=200)
            
#             except Exception as e:
#                 st.error(f"Error during transcription: {e}")