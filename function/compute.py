"""
Simple Audio Analysis for Accent Detection
Uses Whisper API for transcription + GPT-4/Ollama for accent analysis
"""

import os
import tempfile
import re
import shutil
import json
import threading
from typing import Dict, Any
from urllib.parse import urlparse
from pathlib import Path

import requests
import yt_dlp
from pydub import AudioSegment
from openai import OpenAI
import whisper

# ================================
# CONFIGURATION - TOGGLE LLM PROVIDER HERE
# ================================
LLM_PROVIDER = "ollama"  # Options: "openai", "ollama", "ollama_chat"
OLLAMA_MODEL = "llama3.1:8b"  # Ollama model to use: llama3.1, llama3, mistral, etc.
USE_FALLBACK = True  # If Ollama fails, fallback to OpenAI?

# Initialize OpenAI client

# Create temp directory for processing
TEMP_DIR = tempfile.mkdtemp(prefix='accent_detection_')

# Create JSON directory for saving transcripts
JSON_DIR = 'json'
os.makedirs(JSON_DIR, exist_ok=True)

# Global session ID for status tracking
_current_session_id = None

def set_session_id(session_id):
    """Set the current session ID for status tracking"""
    global _current_session_id
    _current_session_id = session_id

def send_status(message, status_type='info'):
    """Send status update by writing directly to status file"""
    print(f"[{status_type.upper()}] {message}")  # Always log to console
    
    # Write to status file if session ID is available
    if _current_session_id:
        write_status_to_file(_current_session_id, message, status_type)

def write_status_to_file(session_id, message, status_type='info'):
    """Write status update directly to file"""
    try:
        import time
        import json
        
        status_folder = 'status'
        os.makedirs(status_folder, exist_ok=True)
        
        status_file = os.path.join(status_folder, f"{session_id}.json")
        status_data = {
            'message': message,
            'type': status_type,
            'timestamp': time.time()
        }
        
        # Read existing status or create new
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                existing_data = json.load(f)
            existing_data['logs'].append(status_data)
        else:
            existing_data = {
                'session_id': session_id,
                'status': 'processing',
                'logs': [status_data]
            }
        
        # Write updated status
        with open(status_file, 'w') as f:
            json.dump(existing_data, f)
            
    except Exception as e:
        print(f"Error writing status to file: {e}")

def set_status_callback(callback):
    """Legacy function - now deprecated but kept for compatibility"""
    print("Warning: set_status_callback is deprecated, using direct file writing instead")
    pass


# Load Whisper model for timestamped transcripts (running in background)
whisper_model = None

def load_whisper_model():
    """Load Whisper model in background thread"""
    global whisper_model
    try:
        send_status("🔄 Loading Whisper model for timestamped transcripts...", 'info')
        whisper_model = whisper.load_model("base")
        send_status("✅ Whisper model loaded successfully", 'success')
    except Exception as e:
        send_status(f"⚠️ Could not load Whisper model: {str(e)}", 'warning')

# Load the model in a background thread at startup
threading.Thread(target=load_whisper_model, daemon=True).start()


def create_timestamped_transcript(audio_file_path: str, session_id: str):
    """
    Create timestamped transcript using local Whisper model
    Runs in background thread and saves to JSON file
    """
    def process_transcript():
        try:
            # Wait for model to be loaded
            if whisper_model is None:
                send_status("⏳ Waiting for Whisper model to load...", 'info')
                # Wait up to 60 seconds for model to load
                for _ in range(60):
                    if whisper_model is not None:
                        break
                    threading.Event().wait(1)
                
                if whisper_model is None:
                    send_status("❌ Whisper model not available for timestamped transcripts", 'error')
                    return
            
            send_status("🎯 Creating timestamped transcript...", 'info')
            
            # Ensure we have a clean file path
            if not os.path.exists(audio_file_path):
                send_status(f"❌ Audio file not found: {audio_file_path}", 'error')
                return
            
            # Get absolute path to avoid any path issues
            abs_audio_path = os.path.abspath(audio_file_path)
            
            # Create a clean temporary file for Whisper processing
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp()
            temp_audio_path = os.path.join(temp_dir, f"audio_{session_id}.wav")
            
            try:
                # Copy the audio file to temporary location
                shutil.copy2(abs_audio_path, temp_audio_path)
                
                # Transcribe with word timestamps using the clean temporary file
                # Use string path and disable verbose to avoid Whisper src attribute issues
                result = whisper_model.transcribe(str(temp_audio_path), word_timestamps=True, verbose=False)
                
            finally:
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
            
            # Format the transcript data
            transcript_data = {
                "session_id": session_id,
                "full_text": result["text"],
                "language": result.get("language", "en"),
                "segments": []
            }
            
            # Process segments with timestamps
            for segment in result["segments"]:
                segment_data = {
                    "start": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "text": segment["text"].strip(),
                    "duration": round(segment["end"] - segment["start"], 2)
                }
                transcript_data["segments"].append(segment_data)
            
            # Save to JSON file
            json_file_path = os.path.join(JSON_DIR, f"{session_id}_transcript.json")
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            send_status(f"✅ Timestamped transcript saved: {len(transcript_data['segments'])} segments", 'success')
            
            # Print preview to console
            print(f"\n📋 Timestamped Transcript Preview:")
            for i, segment in enumerate(transcript_data["segments"][:5]):
                print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
            if len(transcript_data["segments"]) > 5:
                print(f"... and {len(transcript_data['segments']) - 5} more segments")
            print()
            
        except Exception as e:
            send_status(f"❌ Failed to create timestamped transcript: {str(e)}", 'error')
            print(f"Detailed Whisper error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run in background thread
    thread = threading.Thread(target=process_transcript, daemon=True)
    thread.start()
    return thread


def get_timestamped_transcript(session_id: str) -> Dict[str, Any]:
    """
    Retrieve timestamped transcript for a session
    Used for Q&A functionality
    """
    try:
        json_file_path = os.path.join(JSON_DIR, f"{session_id}_transcript.json")
        
        if not os.path.exists(json_file_path):
            return {
                "status": "not_ready",
                "message": "Timestamped transcript is still being created or doesn't exist"
            }
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        return {
            "status": "ready",
            "data": transcript_data,
            "segments_count": len(transcript_data.get("segments", [])),
            "duration": transcript_data["segments"][-1]["end"] if transcript_data.get("segments") else 0
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving transcript: {str(e)}"
        }


def ask_about_video(session_id: str, question: str) -> str:
    """
    Answer questions about the video using timestamped transcript and GPT-4
    """
    try:
        # Get the timestamped transcript
        transcript_info = get_timestamped_transcript(session_id)
        
        if transcript_info["status"] != "ready":
            return f"Sorry, the transcript is not ready yet. {transcript_info.get('message', '')}"
        
        transcript_data = transcript_info["data"]
        
        # Format transcript for GPT-4
        formatted_transcript = ""
        for segment in transcript_data["segments"]:
            formatted_transcript += f"[{segment['start']:.1f}s-{segment['end']:.1f}s]: {segment['text']}\n"
        
        # Ask GPT-4 the question with transcript context
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about video/audio content. You have access to a timestamped transcript. When answering, reference specific timestamps when relevant."
                },
                {
                    "role": "user",
                    "content": f"""
Here's the timestamped transcript of the video:

{formatted_transcript}

Question: {question}

Please answer the question based on the transcript. Include relevant timestamps in your response when applicable.
"""
                }
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error processing your question: {str(e)}"


def cleanup_temp_files():
    """Remove all temporary files when done"""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
    except Exception:
        pass  # Don't worry if cleanup fails


def is_video_platform_url(url: str) -> bool:
    """Check if the URL is from a video platform like YouTube, Loom, etc."""
    video_sites = [
        'youtube.com', 'youtu.be', 'loom.com', 'vimeo.com',
        'dailymotion.com', 'twitch.tv', 'facebook.com',
        'instagram.com', 'tiktok.com', 'twitter.com', 'x.com'
    ]
    return any(site in url.lower() for site in video_sites)


def is_direct_audio_file(url: str) -> bool:
    """Check if URL points directly to an audio file"""
    audio_extensions = ['.wav', '.mp3', '.ogg', '.m4a', '.aac', '.flac']
    parsed_url = urlparse(url)
    return any(parsed_url.path.lower().endswith(ext) for ext in audio_extensions)


def download_video_and_extract_audio(video_url: str) -> str:
    """Download video from any platform and extract audio as optimized WAV file"""
    try:
        output_template = os.path.join(TEMP_DIR, '%(title)s.%(ext)s')
        
        # yt-dlp settings optimized for speech analysis
        download_options = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '128',  # Lower quality for smaller files
            }],
            'quiet': True,
            'no_warnings': True,
            'cookiefile':  'cookies.txt',  # Path to cookies file
         }
        
        with yt_dlp.YoutubeDL(download_options) as downloader:
            downloader.extract_info(video_url, download=True)
            
            # Find the downloaded WAV file
            raw_audio_path = None
            for filename in os.listdir(TEMP_DIR):
                if filename.endswith('.wav'):
                    raw_audio_path = os.path.join(TEMP_DIR, filename)
                    break
            
            if not raw_audio_path:
                raise Exception("Could not find extracted audio file")
            
            # Optimize the audio for Whisper API
            send_status("🔧 Optimizing downloaded audio...", 'info')
            audio = AudioSegment.from_file(raw_audio_path)
            audio = optimize_audio_for_whisper(audio)
            
            optimized_path = os.path.join(TEMP_DIR, 'optimized_video_audio.wav')
            audio.export(optimized_path, format='wav')
            
            # Remove the original larger file
            try:
                os.remove(raw_audio_path)
            except:
                pass
            
            return optimized_path
            
    except Exception as e:
        raise Exception(f"Failed to download video: {str(e)}")


def download_audio_file(audio_url: str) -> str:
    """Download audio file directly from URL"""
    try:
        response = requests.get(audio_url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Guess file extension from URL or content type
        file_extension = '.wav'
        if 'content-type' in response.headers:
            content_type = response.headers['content-type'].lower()
            if 'mp3' in content_type:
                file_extension = '.mp3'
            elif 'ogg' in content_type:
                file_extension = '.ogg'
        else:
            url_path = urlparse(audio_url).path.lower()
            for ext in ['.wav', '.mp3', '.ogg', '.m4a']:
                if url_path.endswith(ext):
                    file_extension = ext
                    break
        
        file_path = os.path.join(TEMP_DIR, f'downloaded_audio{file_extension}')
        
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        return file_path
        
    except Exception as e:
        raise Exception(f"Failed to download audio: {str(e)}")


def convert_to_wav(input_file_path: str) -> str:
    """Convert any audio/video file to WAV format with size optimization"""
    try:
        output_path = os.path.join(TEMP_DIR, 'converted_audio.wav')
        
        # Load audio file
        audio = AudioSegment.from_file(input_file_path)
        
        # Optimize audio to reduce file size while maintaining quality for speech
        audio = optimize_audio_for_whisper(audio)
        
        # Export as WAV
        audio.export(output_path, format='wav')
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Failed to convert audio: {str(e)}")


def optimize_audio_for_whisper(audio: AudioSegment) -> AudioSegment:
    """Optimize audio for Whisper API - reduce size while keeping speech quality"""
    try:
        # Convert to mono (reduces file size by ~50%)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set sample rate to 16kHz (good for speech, smaller than 44.1kHz)
        audio = audio.set_frame_rate(16000)
        
        # Limit duration to 10 minutes (600 seconds) to prevent huge files
        max_duration = 10 * 60 * 1000  # 10 minutes in milliseconds
        if len(audio) > max_duration:
            print(f"⚠️ Audio is {len(audio)//1000//60} minutes, truncating to 10 minutes for analysis")
            audio = audio[:max_duration]
        
        # Normalize volume (can help with file size)
        audio = audio.normalize()
        
        return audio
        
    except Exception as e:
        print(f"Warning: Could not optimize audio: {e}")
        return audio


def check_file_size_for_whisper(file_path: str) -> bool:
    """Check if file size is acceptable for Whisper API (under 25MB)"""
    max_size = 25 * 1024 * 1024  # 25 MB in bytes
    file_size = os.path.getsize(file_path)
    
    print(f"📊 Audio file size: {file_size / (1024*1024):.1f} MB")
    
    if file_size > max_size:
        print(f"⚠️ File too large for Whisper API ({file_size / (1024*1024):.1f} MB > 25 MB)")
        return False
    return True


def compress_audio_if_needed(audio_file_path: str) -> str:
    """Compress audio file if it's too large for Whisper API"""
    try:
        if check_file_size_for_whisper(audio_file_path):
            return audio_file_path
        
        send_status("🔧 Compressing audio to fit Whisper API limits...", 'warning')
        
        # Load and compress audio
        audio = AudioSegment.from_file(audio_file_path)
        audio = optimize_audio_for_whisper(audio)
        
        # Try different compression levels until file is small enough
        compressed_path = os.path.join(TEMP_DIR, 'compressed_audio.wav')
        
        # First attempt: standard compression
        audio.export(compressed_path, format='wav')
        
        if check_file_size_for_whisper(compressed_path):
            return compressed_path
        
        # Second attempt: reduce quality further
        print("🔧 Applying additional compression...")
        audio = audio.set_frame_rate(8000)  # Lower quality but smaller
        audio.export(compressed_path, format='wav')
        
        if check_file_size_for_whisper(compressed_path):
            return compressed_path
        
        # Final attempt: truncate to 5 minutes
        print("🔧 Truncating to 5 minutes for analysis...")
        max_duration = 5 * 60 * 1000  # 5 minutes
        audio = audio[:max_duration]
        audio.export(compressed_path, format='wav')
        
        return compressed_path
        
    except Exception as e:
        raise Exception(f"Failed to compress audio: {str(e)}")


def transcribe_audio_with_whisper(audio_file_path: str) -> str:
    """Use OpenAI Whisper to transcribe the audio file"""
    try:
        with open(audio_file_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        return transcription.text
        
    except Exception as e:
        raise Exception(f"Whisper transcription failed: {str(e)}")


def analyze_accent_from_text(transcribed_text: str) -> str:
    """Analyze the transcribed text using GPT-4 to detect accent and proficiency"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a linguistics expert specializing in English accent detection and pronunciation analysis. You must ALWAYS provide a specific analysis based on the text patterns, word choices, and linguistic markers present.

Even without audio, you can analyze:
- Word choice patterns typical of different English-speaking regions
- Grammar structures common to specific accents
- Spelling variations that suggest accent influence
- Idiomatic expressions linked to particular regions

Never respond with "unknown", "undetermined", or that you need audio. Instead, analyze the available textual evidence and make reasoned inferences about the most likely accent and proficiency level."""
                },
                {
                    "role": "user",
                    "content": f"""
Based on the text patterns and linguistic markers in this transcription, provide:

1. **Accent Classification**: (<ACCENT_NAME>:<DESCRIPTION>)Determine the most likely English accent based on word choices, grammar patterns, and regional expressions. Be specific (e.g., "Midwestern American", "London British", "Mumbai Indian English")

2. **Confidence Score**: Provide a confidence score (0-100%) based on how many clear linguistic markers are present. Explain what specific patterns informed your score.

3. **Proficiency Level**: Assess as Beginner, Intermediate, Advanced, or Native based on:
   - Grammar complexity
   - Vocabulary range
   - Sentence structure
   - Idiomatic usage

4. **Analysis**: List specific examples from the text that indicate the accent:
   - Word choices
   - Grammar patterns  
   - Regional expressions
   - Spelling patterns (if present)

5. **Speech Quality**: Analyze:
   - Grammar accuracy
   - Vocabulary appropriateness
   - Sentence complexity
   - Overall communication effectiveness

Text to analyze: "{transcribed_text}"

Remember: You must provide specific analysis based on textual evidence. Do not say you need audio or that something is unknown."""
                }
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"analysis failed: {str(e)}")


def analyze_accent_from_text_ollama(transcribed_text: str, model: str = "llama3.1") -> str:
    """Analyze the transcribed text using Ollama Llama models to detect accent and proficiency"""
    try:
        # Ollama API endpoint (default local installation)
        ollama_url = "http://localhost:11434/api/generate"
        
        # Enhanced prompt for Llama models
        system_prompt = """You are a linguistics expert specializing in English accent detection and pronunciation analysis. You must ALWAYS provide a specific analysis based on the text patterns, word choices, and linguistic markers present.

Even without audio, you can analyze:
- Word choice patterns typical of different English-speaking regions
- Grammar structures common to specific accents
- Spelling variations that suggest accent influence
- Idiomatic expressions linked to particular regions

Never respond with "unknown", "undetermined", or that you need audio. Instead, analyze the available textual evidence and make reasoned inferences about the most likely accent and proficiency level."""

        user_prompt = f"""Based on the text patterns and linguistic markers in this transcription, provide:

1. **Accent Classification**: Determine the most likely English accent based on word choices, grammar patterns, and regional expressions. Be specific (e.g., "Midwestern American", "London British", "Mumbai Indian English")

2. **Confidence Score**: Provide a confidence score (0-100%) based on how many clear linguistic markers are present. Explain what specific patterns informed your score.

3. **Proficiency Level**: Assess as Beginner, Intermediate, Advanced, or Native based on:
   - Grammar complexity
   - Vocabulary range
   - Sentence structure
   - Idiomatic usage

4. **Analysis**: List specific examples from the text that indicate the accent:
   - Word choices
   - Grammar patterns  
   - Regional expressions
   - Spelling patterns (if present)

5. **Speech Quality**: Analyze:
   - Grammar accuracy
   - Vocabulary appropriateness
   - Sentence complexity
   - Overall communication effectiveness

Text to analyze: "{transcribed_text}"

Remember: You must provide specific analysis based on textual evidence. Do not say you need audio or that something is unknown."""

        # Combine system and user prompts for Ollama
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}"
        
        # Ollama API request
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        response = requests.post(
            ollama_url,
            json=payload,
            timeout=120  # 2 minutes timeout for analysis
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Analysis failed - no response from model")
        else:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to Ollama. Make sure Ollama is running on localhost:11434")
    except requests.exceptions.Timeout:
        raise Exception("Ollama request timed out. The model might be too slow or busy.")
    except Exception as e:
        raise Exception(f"Ollama analysis failed: {str(e)}")

def analyze_accent_from_text_ollama_chat(transcribed_text: str, model: str = "llama3.1") -> str:
    """Alternative Ollama function using chat API format (if supported)"""
    try:
        # Ollama chat API endpoint
        ollama_url = "http://localhost:11434/api/chat"
        
        # Chat format payload
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": """You are a linguistics expert specializing in English accent detection and pronunciation analysis. You must ALWAYS provide a specific analysis based on the text patterns, word choices, and linguistic markers present.

Even without audio, you can analyze:
- Word choice patterns typical of different English-speaking regions
- Grammar structures common to specific accents
- Spelling variations that suggest accent influence
- Idiomatic expressions linked to particular regions

Never respond with "unknown", "undetermined", or that you need audio. Instead, analyze the available textual evidence and make reasoned inferences about the most likely accent and proficiency level."""
                },
                {
                    "role": "user",
                    "content": f"""Based on the text patterns and linguistic markers in this transcription, provide:

1. **Accent Classification**: Determine the most likely English accent based on word choices, grammar patterns, and regional expressions. Be specific (e.g., "Midwestern American", "London British", "Mumbai Indian English")

2. **Confidence Score**: Provide a confidence score (0-100%) based on how many clear linguistic markers are present. Explain what specific patterns informed your score.

3. **Proficiency Level**: Assess as Beginner, Intermediate, Advanced, or Native based on:
   - Grammar complexity
   - Vocabulary range
   - Sentence structure
   - Idiomatic usage

4. **Analysis**: List specific examples from the text that indicate the accent:
   - Word choices
   - Grammar patterns  
   - Regional expressions
   - Spelling patterns (if present)

5. **Speech Quality**: Analyze:
   - Grammar accuracy
   - Vocabulary appropriateness
   - Sentence complexity
   - Overall communication effectiveness

Text to analyze: "{transcribed_text}"

Remember: You must provide specific analysis based on textual evidence. Do not say you need audio or that something is unknown."""
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 1000
            }
        }
        
        response = requests.post(
            ollama_url,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "Analysis failed - no response from model")
        else:
            raise Exception(f"Ollama Chat API error: {response.status_code} - {response.text}")
        
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to Ollama. Make sure Ollama is running on localhost:11434")
    except requests.exceptions.Timeout:
        raise Exception("Ollama request timed out. The model might be too slow or busy.")
    except Exception as e:
        raise Exception(f"Ollama chat analysis failed: {str(e)}")


def analyze_with_configured_provider(transcribed_text: str) -> str:
    """
    Analyze text using the configured LLM provider with detailed status messages
    Handles fallback from Ollama to OpenAI if enabled
    """
    try:
        if LLM_PROVIDER == "openai":
            send_status("🤖 Using OpenAI GPT-4 for accent analysis...", 'info')
            result = analyze_accent_from_text(transcribed_text)
            send_status("✅ OpenAI GPT-4 analysis completed", 'success')
            return result
        
        elif LLM_PROVIDER == "ollama":
            send_status(f"🦙 Using Ollama {OLLAMA_MODEL} for accent analysis...", 'info')
            result = analyze_accent_from_text_ollama(transcribed_text, OLLAMA_MODEL)
            send_status(f"✅ Ollama {OLLAMA_MODEL} analysis completed", 'success')
            return result
        
        elif LLM_PROVIDER == "ollama_chat":
            send_status(f"🦙 Using Ollama {OLLAMA_MODEL} (Chat API) for accent analysis...", 'info')
            result = analyze_accent_from_text_ollama_chat(transcribed_text, OLLAMA_MODEL)
            send_status(f"✅ Ollama {OLLAMA_MODEL} (Chat) analysis completed", 'success')
            return result
        
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
    
    except Exception as e:
        if USE_FALLBACK and LLM_PROVIDER != "openai":
            send_status(f"❌ {LLM_PROVIDER.title()} {OLLAMA_MODEL} failed: {str(e)}", 'error')
            send_status("🔄 Attempting fallback to OpenAI GPT-4...", 'info')
            try:
                result = analyze_accent_from_text(transcribed_text)
                send_status("✅ OpenAI GPT-4 fallback successful", 'success')
                return result
            except Exception as fallback_error:
                send_status("❌ OpenAI fallback also failed", 'error')
                raise Exception(f"Both {LLM_PROVIDER} and OpenAI failed. {LLM_PROVIDER}: {str(e)}, OpenAI: {str(fallback_error)}")
        else:
            raise e


def analyze_audio(audio_source: str, source_type: str, session_id: str = None) -> Dict[str, Any]:
    """
    Main function: Analyze audio from URL or uploaded file
    
    Steps:
    1. Get the audio file (download from URL or process uploaded file)
    2. Convert to WAV if needed
    3. Transcribe using Whisper
    4. Analyze transcription using configured LLM (OpenAI/Ollama)
    5. Return structured results
    """
    try:
        send_status(f"Starting analysis of {source_type}: {audio_source[:50]}...", 'info')
        
        # Show current LLM configuration
        if LLM_PROVIDER == "openai":
            send_status("⚙️ Configuration: Using OpenAI GPT-4 for analysis", 'info')
        elif LLM_PROVIDER in ["ollama", "ollama_chat"]:
            api_type = " (Chat API)" if LLM_PROVIDER == "ollama_chat" else ""
            send_status(f"⚙️ Configuration: Using Ollama {OLLAMA_MODEL}{api_type} for analysis", 'info')
            if USE_FALLBACK:
                send_status("⚙️ Fallback to OpenAI enabled if Ollama fails", 'info')
        
        # Step 1: Get the audio file
        if source_type == 'url':
            if is_video_platform_url(audio_source):
                send_status("📹 Detected video platform URL - downloading and extracting audio...", 'info')
                audio_file_path = download_video_and_extract_audio(audio_source)
            elif is_direct_audio_file(audio_source):
                send_status("🎵 Detected direct audio URL - downloading...", 'info')
                audio_file_path = download_audio_file(audio_source)
            else:
                send_status("🔗 Trying to download as audio file...", 'info')
                audio_file_path = download_audio_file(audio_source)
                
        elif source_type == 'file':
            send_status("📁 Processing uploaded file...", 'info')
            # For uploaded files, convert to WAV if needed
            audio_file_path = convert_to_wav(audio_source)
        else:
            raise ValueError("source_type must be 'url' or 'file'")
        
        # Step 2: Make sure we have a valid audio file
        if not os.path.exists(audio_file_path):
            raise Exception("Audio file not found after processing")
        
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            raise Exception("Audio file is empty")
        
        send_status(f"✅ Audio file ready: {file_size / (1024*1024):.1f} MB", 'success')
        
        # Step 2.5: Compress audio if needed for Whisper API
        send_status("🔍 Checking file size for Whisper API...", 'info')
        audio_file_path = compress_audio_if_needed(audio_file_path)
        file_size = os.path.getsize(audio_file_path)
        
        # Step 3: Transcribe the audio using Whisper
        send_status("🎙️ Transcribing audio with Whisper API...", 'info')
        transcribed_text = transcribe_audio_with_whisper(audio_file_path)
        
        if not transcribed_text or len(transcribed_text.strip()) < 5:
            raise Exception("Transcription too short or empty - audio may not contain clear speech")
        
        send_status(f"📝 Transcription complete: {len(transcribed_text)} characters", 'success')
        
        # Step 3.5: Start creating timestamped transcript in background (if session_id provided)
        if session_id:
            send_status("🎯 Starting timestamped transcript creation...", 'info')
            create_timestamped_transcript(audio_file_path, session_id)
        
        # Step 4: Analyze the transcription with configured LLM provider
        llm_analysis = analyze_with_configured_provider(transcribed_text)
        
        # Step 5: Extract structured data
        result = extract_analysis_details(llm_analysis)
        
        # Add metadata and transcription
        provider_name = "OpenAI GPT-4" if LLM_PROVIDER == "openai" else f"Ollama {OLLAMA_MODEL}"
        result.update({
            "status": "success",
            "transcribed_text": transcribed_text,
            "source_type": source_type,
            "audio_size_bytes": file_size,
            "processing_method": f"Whisper + {provider_name} Analysis"
        })
        
        send_status("✅ Analysis complete!", 'success')
        return result
        
    except Exception as e:
        error_message = str(e)
        send_status(f"❌ Analysis failed: {error_message}", 'error')
        
        return {
            "status": "error",
            "error_message": error_message,
            "accent_classification": "Analysis Failed",
            "confidence_score": 0,
            "proficiency_level": "Unknown",
            "detailed_analysis": f"Error occurred: {error_message}",
            "summary": "Analysis failed",
            "transcribed_text": "",
            "processing_method": "Failed"
        }
    
    finally:
        # Always clean up temporary files
        cleanup_temp_files()

def check_ollama_connection(model: str = "llama3.1") -> Dict[str, Any]:
    """
    Check if Ollama is running and the specified model is available
    
    Returns:
        Dictionary with connection status and available models
    """
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            available_models = [model_info['name'] for model_info in data.get('models', [])]
            
            model_available = any(model in model_name for model_name in available_models)
            
            return {
                "status": "connected",
                "ollama_running": True,
                "available_models": available_models,
                "target_model": model,
                "model_available": model_available,
                "message": f"Ollama is running with {len(available_models)} models available"
            }
        else:
            return {
                "status": "error",
                "ollama_running": False,
                "message": f"Ollama responded with status {response.status_code}"
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "ollama_running": False,
            "message": "Could not connect to Ollama. Make sure it's running on localhost:11434"
        }
    except Exception as e:
        return {
            "status": "error",
            "ollama_running": False,
            "message": f"Error checking Ollama: {str(e)}"
        }


def extract_analysis_details(gpt_response: str) -> Dict[str, Any]:
    """Parse   response and extract structured information"""
    result = {
        "accent_classification": "Unknown",
        "confidence_score": 0,
        "proficiency_level": "Unknown",
        "detailed_analysis": gpt_response,
        "summary": "Analysis completed"
    }
    
    lines = gpt_response.split('\n')
    
    for line in lines:
        line_clean = line.lower().strip()
        original_line = line.strip()
        
        # Look for accent classification - check after the colon or label
        if any(keyword in line_clean for keyword in ['accent classification', '**accent classification**']):
            # Extract text after the colon
            if ':' in original_line:
                accent_text = original_line.split(':', 1)[1].strip()
                if accent_text and len(accent_text) > 2:
                    result["accent_classification"] = accent_text
            else:
                # Fallback to looking for common accents
                accents = ['general american', 'american', 'british', 'australian', 'indian', 'canadian', 'scottish', 'irish', 'south african']
                for accent in accents:
                    if accent in line_clean:
                        result["accent_classification"] = accent.title().replace('General American', 'American English')
                        if not result["accent_classification"].endswith('English'):
                            result["accent_classification"] += " English"
                        break
        
        # Look for confidence score - be more specific about the pattern
        elif any(keyword in line_clean for keyword in ['confidence score', '**confidence score**']):
            # Look for percentage after colon or in the line
            percentage_match = re.search(r'(\d{1,3})%', original_line)
            if percentage_match:
                score = int(percentage_match.group(1))
                if 0 <= score <= 100:
                    result["confidence_score"] = score
            else:
                # Fallback: look for any number that could be a score
                numbers = re.findall(r'\b(\d{1,3})\b', original_line)
                for num in numbers:
                    score = int(num)
                    if 70 <= score <= 100:  # Reasonable confidence score range
                        result["confidence_score"] = score
                        break
        
        # Look for proficiency level
        elif any(keyword in line_clean for keyword in ['proficiency level', '**proficiency level**']):
            if 'native' in line_clean:
                result["proficiency_level"] = "Native"
            elif 'advanced' in line_clean:
                result["proficiency_level"] = "Advanced"
            elif 'intermediate' in line_clean:
                result["proficiency_level"] = "Intermediate"
            elif 'beginner' in line_clean:
                result["proficiency_level"] = "Beginner"
    
    # If we still haven't found confidence score, try a broader search
    if result["confidence_score"] == 0:
        # Look for any percentage in the entire response
        all_percentages = re.findall(r'(\d{1,3})%', gpt_response)
        for percentage in all_percentages:
            score = int(percentage)
            if 70 <= score <= 100:  # Reasonable confidence range
                result["confidence_score"] = score
                break
    
    # Clean up accent classification
    if result["accent_classification"] != "Unknown":
        # Ensure it ends with "English" if it's a regional accent
        accent = result["accent_classification"]
        if not accent.endswith("English") and any(x in accent.lower() for x in ['american', 'british', 'australian', 'indian', 'canadian']):
            result["accent_classification"] = accent + " English"
    
    return result 