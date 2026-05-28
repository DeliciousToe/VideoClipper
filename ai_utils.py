import os
import json
import subprocess
from faster_whisper import WhisperModel
import google.generativeai as genai

def extract_audio(video_path, audio_path):
    """
    Extracts audio from a video file using FFmpeg for maximum performance and stability.
    Falls back to MoviePy only if FFmpeg is not found or fails.
    """
    try:
        print(f"Extracting audio using FFmpeg: {video_path} -> {audio_path}")
        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',                 # Disable video recording
            '-acodec', 'pcm_s16le', # WAV format
            '-ar', '16000',        # 16kHz sample rate (Whisper standard)
            '-ac', '1',            # Mono audio channel
            audio_path
        ]
        # Run FFmpeg silently, checking for errors
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return audio_path
    except Exception as e:
        print(f"FFmpeg audio extraction failed, falling back to MoviePy: {e}")
        try:
            from moviepy import VideoFileClip
        except ImportError:
            from moviepy.editor import VideoFileClip
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, logger=None, verbose=False)
        video.close()
        return audio_path

def transcribe_audio(audio_path, model_size="base"):
    """
    Transcribes audio locally using faster-whisper.
    Auto-detects GPU (CUDA) availability for high-speed local processing.
    """
    device = "cpu"
    compute_type = "int8"
    
    # Auto-detect CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            print("Nvidia GPU with CUDA detected! Running Whisper on GPU.")
    except Exception:
        # Alt-check if torch is not installed or check fails, try initializing cuda
        try:
            test_model = WhisperModel("tiny", device="cuda", compute_type="float16")
            device = "cuda"
            compute_type = "float16"
            del test_model
            print("CUDA device initialized successfully! Running Whisper on GPU.")
        except Exception:
            print("No CUDA device initialized. Running Whisper on CPU.")
            device = "cpu"
            compute_type = "int8"

    print(f"Initializing faster-whisper model '{model_size}' on device: {device}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # Enable word-level timestamps
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    
    transcript = []
    for segment in segments:
        words = []
        if segment.words:
            for w in segment.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word.strip()
                })
        transcript.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": words
        })
    return transcript

def analyze_transcript_with_gemini(transcript, api_key):
    """
    Analyzes local Whisper transcript text with Gemini API to identify viral moments
    and constructs matching phrase-level subtitle blocks.
    """
    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Format transcript text with timestamps
    transcript_text = "\n".join([f"[{t['start']:.2f} - {t['end']:.2f}] {t['text']}" for t in transcript])
    
    prompt = f"""
    You are an expert video editor and social media manager.
    Analyze the following video transcript with timestamps.
    Your goal is to identify the 3 most engaging, viral-worthy moments that would make great TikToks or YouTube Shorts.
    Each clip should be between 15 and 60 seconds long.
    
    Transcript:
    {transcript_text}
    
    Return a valid JSON array where each object has the following keys:
    - "title": A catchy, viral title for the clip.
    - "start_time": The start time in seconds (float).
    - "end_time": The end time in seconds (float).
    - "explanation": Why this makes a good short.
    
    Output ONLY valid JSON without any markdown formatting. Do not wrap in ```json ```.
    """
    
    response = model.generate_content(prompt)
    try:
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        highlights = json.loads(text.strip())
        
        # Build subtitle phrases for each highlight from local word-level timestamps
        for highlight in highlights:
            start_time = highlight["start_time"]
            end_time = highlight["end_time"]
            
            subtitles = []
            current_phrase = []
            phrase_start = None
            
            for segment in transcript:
                for w in segment.get("words", []):
                    w_start = w["start"]
                    w_end = w["end"]
                    
                    # Capture words within the highlight window
                    if w_start >= start_time and w_end <= end_time:
                        if not current_phrase:
                            phrase_start = w_start
                        current_phrase.append(w["word"])
                        
                        # Dynamic grouping of 3 words per subtitle block for nice vertical reels
                        if len(current_phrase) >= 3:
                            subtitles.append({
                                "start": phrase_start,
                                "end": w_end,
                                "text": " ".join(current_phrase)
                            })
                            current_phrase = []
                            phrase_start = None
                            
            if current_phrase:
                subtitles.append({
                    "start": phrase_start if phrase_start is not None else start_time,
                    "end": end_time,
                    "text": " ".join(current_phrase)
                })
            highlight["subtitles"] = subtitles
            
        return highlights
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return []

def analyze_video_cloud(audio_path, api_key):
    """
    Sends the audio file directly to the Gemini API (Cloud Mode) for lightning-fast, 
    low-compute transcription and highlight detection in a single step.
    Includes subtitle chunk extraction only for the targeted highlights to minimize token payload.
    """
    genai.configure(api_key=api_key.strip())
    print(f"Uploading audio file to Gemini cloud: {audio_path}")
    audio_file = genai.upload_file(path=audio_path)
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    You are an expert video editor and social media manager.
    Analyze the uploaded audio file and identify the 3 most engaging, viral-worthy highlights that would make great TikToks or YouTube Shorts.
    Each clip should be between 15 and 60 seconds long.
    
    IMPORTANT NOTE FOR SILENT OR NON-VOICE AUDIOS (e.g. gameplay audio with background music, combat sounds, or sound effects, but no spoken voices):
    If the audio contains no spoken voice or very little speech, you must still identify the 3 most exciting, high-energy, or action-packed highlights (based on dramatic gameplay sounds, weapon clashes, explosions, music beats, or intensity changes).
    For these non-voice highlights, instead of spoken words, you must still populate the "subtitles" list with descriptive sound effect captions in brackets (e.g., "[EXPLOSION]", "[SWORD CLASH]", "[ACTION MUSIC INTENSIFIES]", "[AMBIENT WIND]", "[GUNSHOTS]") that match the excitement of the timeframe, so they can be burned onto the video.
    
    Your output MUST be a valid JSON array where each object has the following keys:
    - "title": A catchy, viral title for the clip.
    - "start_time": The start time in seconds (float).
    - "end_time": The end time in seconds (float).
    - "explanation": Why this makes a good short.
    - "subtitles": A list of small, phrase-level subtitle blocks within this highlight. Each subtitle block should be an object with:
        - "start": Start time of the phrase in seconds (float).
        - "end": End time of the phrase in seconds (float).
        - "text": The exact transcribed words spoken during this phrase, OR a descriptive sound effect caption in brackets if there is no speech (keep it short, 2-4 words per phrase).
        Ensure these subtitles cover the entire duration of the clip from start_time to end_time so they can be burned onto the video.
        
    Ensure the JSON is properly formatted, do not truncate, and do not wrap in ```json ``` markdown code blocks.
    """
    
    try:
        response = model.generate_content([audio_file, prompt])
        text = response.text.strip()
        
        # Clean up the cloud file immediately
        try:
            genai.delete_file(audio_file.name)
            print("Successfully deleted audio file from Gemini cloud storage.")
        except Exception as delete_err:
            print(f"Failed to delete audio file: {delete_err}")
            
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
            
        highlights = json.loads(text.strip())
        return highlights
    except Exception as e:
        print(f"Error in Gemini cloud analysis: {e}")
        try:
            genai.delete_file(audio_file.name)
        except Exception:
            pass
        return []
