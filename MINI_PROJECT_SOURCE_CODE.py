import os
import streamlit as st
import moviepy.editor as mp
from pydub import AudioSegment
from googletrans import Translator
from youtube_transcript_api import YouTubeTranscriptApi
import speech_recognition as sr
from transformers import BartTokenizer, BartForConditionalGeneration

# Set TOKENIZERS_PARALLELISM to avoid parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to process local video and extract text using Google Speech Recognition
def process_local_video(video_path):
    clip = mp.VideoFileClip(video_path)
    audio_path = "audio1.wav"
    clip.audio.write_audiofile(audio_path)  # Generate audio file

    # Loading audio file
    audio_file = AudioSegment.from_wav(audio_path)
    
    # Define the duration of each segment in milliseconds
    segment_duration = 30000  # 30 seconds for more efficient processing
    
    result_text = ""

    # Process each segment
    recognizer = sr.Recognizer()
    for i in range(0, len(audio_file), segment_duration):
        # Calculate start and end times for the segment
        start_time = i
        end_time = min(i + segment_duration, len(audio_file))  # Ensure end time doesn't exceed the audio duration

        # Extract the segment
        segment = audio_file[start_time:end_time]

        # Export the segment as a temporary WAV file
        segment_path = "temp_segment.wav"
        segment.export(segment_path, format="wav")

        # Recognize speech from the segment
        with sr.AudioFile(segment_path) as source:
            audio_data = recognizer.record(source)  # Read the entire audio file
            try:
                text = recognizer.recognize_google(audio_data)  # Use Google Speech Recognition
                result_text += text + " "
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

    return result_text

# Function to summarize text using BART model
def summarize_text(text, summary_percentage):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)

    # Generate summary
    max_length = int(len(text.split()) * (summary_percentage / 100))
    min_length = int(max_length * 0.6)  # Ensure min_length is roughly 60% of max_length
    summary_ids = model.generate(inputs['input_ids'], max_length=min_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Function to translate text to a specified language using Google Translate API
def translate_text(text, target_language):
    translator = Translator()
    language_codes = {
        "Telugu": "te",
        "Kannada": "kn",
        "Hindi": "hi",
        "Tamil": "ta"
    }
    translated = translator.translate(text, src='en', dest=language_codes[target_language])
    return translated.text

# Function to create a downloadable text file
def create_download_link(text, filename):
    return f"data:text/plain;charset=utf-8,{text}"

# Streamlit frontend
st.markdown(
    """
    <style>
    .main {
        background-color:white
        
    }
    .title {
        color: black;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
    }
    .subtitle {
        color: crimson;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .instruction {
        color: crimson;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>Video Summarizer with Translation</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Choose an option to summarize video content:</h3>", unsafe_allow_html=True)

option = st.selectbox("Select an option", ["Upload Local Video", "Enter YouTube URL"])

if option == "Upload Local Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        st.markdown("<p class='instruction'>Processing the video...</p>", unsafe_allow_html=True)
        result_text = process_local_video("uploaded_video.mp4")
        
        st.markdown("<h3 class='subtitle'>Transcript:</h3>", unsafe_allow_html=True)
        st.write(result_text)

        summary_percentage = st.select_slider(
            "Select summary length as percentage of the original text:",
            options=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            value=30
        )

        st.markdown(f"<h3 class='subtitle'>English Summary ({summary_percentage}%):</h3>", unsafe_allow_html=True)
        summary = summarize_text(result_text, summary_percentage)
        st.write(summary)

        target_language = st.selectbox(
            "Select target language for translation:",
            options=["Kannada", "Telugu", "Hindi", "Tamil"]
        )

        st.markdown(f"<h3 class='subtitle'>{target_language} Summary ({summary_percentage}%):</h3>", unsafe_allow_html=True)
        translated_summary = translate_text(summary, target_language)
        st.write(translated_summary)

        # Add download button for the English summary
        st.download_button(
            label="Download English Summary",
            data=summary,
            file_name="english_summary.txt",
            mime="text/plain"
        )

        # Add download button for the translated summary
        st.download_button(
            label=f"Download {target_language} Summary",
            data=translated_summary,
            file_name=f"{target_language.lower()}_summary.txt",
            mime="text/plain"
        )

elif option == "Enter YouTube URL":
    youtube_url = st.text_input("Enter the YouTube URL")
    if youtube_url:
        st.markdown("<p class='instruction'>Processing the YouTube video...</p>", unsafe_allow_html=True)
        video_id = youtube_url.split("=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        result_text = ""
        for i in transcript:
            result_text += ' ' + i['text']
        
        st.markdown("<h3 class='subtitle'>Transcript:</h3>", unsafe_allow_html=True)
        st.write(result_text)

        summary_percentage = st.select_slider(
            "Select summary length as percentage of the original text:",
            options=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            value=30
        )

        st.markdown(f"<h3 class='subtitle'>English Summary ({summary_percentage}%):</h3>", unsafe_allow_html=True)
        summary = summarize_text(result_text, summary_percentage)
        st.write(summary)

        target_language = st.selectbox(
            "Select target language for translation:",
            options=["Telugu", "Kannada", "Hindi", "Tamil"]
        )

        st.markdown(f"<h3 class='subtitle'>{target_language} Summary ({summary_percentage}%):</h3>", unsafe_allow_html=True)
        translated_summary = translate_text(summary, target_language)
        st.write(translated_summary)

        # Add download button for the English summary
        st.download_button(
            label="Download English Summary",
            data=summary,
            file_name="english_summary.txt",
            mime="text/plain"
        )

        # Add download button for the translated summary
        st.download_button(
            label=f"Download {target_language} Summary",
            data=translated_summary,
            file_name=f"{target_language.lower()}_summary.txt",
            mime="text/plain"
        )
