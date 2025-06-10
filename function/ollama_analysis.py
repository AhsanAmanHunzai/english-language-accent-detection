"""
Ollama LLama Models Integration for Accent Detection
Alternative to OpenAI GPT-4 using local Ollama models
"""

import requests
import os
import json
import time
from typing import Dict, Any


# Global session ID for status tracking
_current_session_id = None

def set_session_id(session_id):
    """Set the current session ID for status tracking"""
    global _current_session_id
    _current_session_id = session_id

def write_status_to_file(session_id, message, status_type='info'):
    """Write status update directly to file"""
    try:
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

def send_status(message, status_type='info'):
    """Send status update by writing directly to status file"""
    print(f"[{status_type.upper()}] {message}")  # Always log to console
    
    # Write to status file if session ID is available
    if _current_session_id:
        write_status_to_file(_current_session_id, message, status_type)

def read_status(session_id):
    """Read status from file"""
    try:
        status_folder = 'status'
        status_file = os.path.join(status_folder, f"{session_id}.json")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error reading status: {e}")
        return None


def analyze_accent_from_text_ollama(transcribed_text: str, model: str = "llama3.1") -> str:
    """Analyze the transcribed text using Ollama Llama models to detect accent and proficiency"""
    try:
        send_status(f"🦙 Connecting to Ollama with {model}...", 'info')
        
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
        
        send_status(f"🧠 Analyzing accent with Ollama {model}...", 'info')
        
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
            analysis_result = result.get("response", "Analysis failed - no response from model")
            send_status(f"✅ Ollama analysis completed ({len(analysis_result)} characters)", 'success')
            return analysis_result
        else:
            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
            send_status(error_msg, 'error')
            raise Exception(error_msg)
        
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to Ollama. Make sure Ollama is running on localhost:11434"
        send_status(error_msg, 'error')
        raise Exception(error_msg)
    except requests.exceptions.Timeout:
        error_msg = "Ollama request timed out. The model might be too slow or busy."
        send_status(error_msg, 'warning')
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Ollama analysis failed: {str(e)}"
        send_status(error_msg, 'error')
        raise Exception(error_msg)


def analyze_accent_from_text_ollama_chat(transcribed_text: str, model: str = "llama3.1") -> str:
    """Alternative Ollama function using chat API format (if supported)"""
    try:
        send_status(f"🦙 Connecting to Ollama Chat API with {model}...", 'info')
        
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
        
        send_status(f"🧠 Analyzing accent with Ollama {model} (chat mode)...", 'info')
        
        response = requests.post(
            ollama_url,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            analysis_result = result.get("message", {}).get("content", "Analysis failed - no response from model")
            send_status(f"✅ Ollama chat analysis completed ({len(analysis_result)} characters)", 'success')
            return analysis_result
        else:
            error_msg = f"Ollama Chat API error: {response.status_code} - {response.text}"
            send_status(error_msg, 'error')
            raise Exception(error_msg)
        
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to Ollama. Make sure Ollama is running on localhost:11434"
        send_status(error_msg, 'error')
        raise Exception(error_msg)
    except requests.exceptions.Timeout:
        error_msg = "Ollama request timed out. The model might be too slow or busy."
        send_status(error_msg, 'warning')
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Ollama chat analysis failed: {str(e)}"
        send_status(error_msg, 'error')
        raise Exception(error_msg)


# Configuration for LLM provider
LLM_PROVIDER = "ollama"  # Options: "openai", "ollama", "ollama_chat"
OLLAMA_MODEL = "llama3.1"  # Can be changed to llama3, codellama, etc.


def analyze_accent_with_provider(transcribed_text: str, provider: str = None, model: str = None) -> str:
    """
    Unified function to analyze accent using different LLM providers
    
    Args:
        transcribed_text: The text to analyze
        provider: "openai", "ollama", or "ollama_chat" (defaults to LLM_PROVIDER)
        model: Model name (for Ollama, defaults to OLLAMA_MODEL)
    
    Returns:
        Analysis result string
    """
    provider = provider or LLM_PROVIDER
    
    if provider == "openai":
        # Import and use OpenAI function
        send_status("🧠 Analyzing accent with OpenAI GPT-4...", 'info')
        from .compute import analyze_accent_from_text
        return analyze_accent_from_text(transcribed_text)
    
    elif provider == "ollama":
        model_name = model or OLLAMA_MODEL
        send_status(f"🦙 Analyzing accent with Ollama {model_name}...", 'info')
        return analyze_accent_from_text_ollama(transcribed_text, model_name)
    
    elif provider == "ollama_chat":
        model_name = model or OLLAMA_MODEL
        send_status(f"🦙 Analyzing accent with Ollama {model_name} (chat)...", 'info')
        return analyze_accent_from_text_ollama_chat(transcribed_text, model_name)
    
    else:
        error_msg = f"Unknown provider: {provider}. Use 'openai', 'ollama', or 'ollama_chat'"
        send_status(error_msg, 'error')
        raise ValueError(error_msg)


def check_ollama_connection(model: str = "llama3.1") -> Dict[str, Any]:
    """
    Check if Ollama is running and the specified model is available
    
    Returns:
        Dictionary with connection status and available models
    """
    try:
        send_status("🔍 Checking Ollama connection...", 'info')
        
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            available_models = [model_info['name'] for model_info in data.get('models', [])]
            
            model_available = any(model in model_name for model_name in available_models)
            
            status_result = {
                "status": "connected",
                "ollama_running": True,
                "available_models": available_models,
                "target_model": model,
                "model_available": model_available,
                "message": f"Ollama is running with {len(available_models)} models available"
            }
            
            if model_available:
                send_status(f"✅ Ollama connected - {model} is available", 'success')
            else:
                send_status(f"⚠️ Ollama connected but {model} not found. Available: {available_models}", 'warning')
            
            return status_result
        else:
            error_msg = f"Ollama responded with status {response.status_code}"
            send_status(error_msg, 'error')
            return {
                "status": "error",
                "ollama_running": False,
                "message": error_msg
            }
            
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to Ollama. Make sure it's running on localhost:11434"
        send_status(error_msg, 'error')
        return {
            "status": "error",
            "ollama_running": False,
            "message": error_msg
        }
    except Exception as e:
        error_msg = f"Error checking Ollama: {str(e)}"
        send_status(error_msg, 'error')
        return {
            "status": "error",
            "ollama_running": False,
            "message": error_msg
        }


# Usage Examples and Documentation
if __name__ == "__main__":
    print("🦙 Ollama Accent Analysis Functions")
    print("=" * 50)
    
    # Test connection
    print("\n1. Testing Ollama Connection...")
    connection_status = check_ollama_connection()
    print(f"Status: {connection_status}")
    
    if connection_status["status"] == "connected":
        print(f"\n✅ Ollama is running!")
        print(f"Available models: {connection_status['available_models']}")
        
        # Example analysis
        sample_text = "Hello, I'm testing this accent detection system. How are you doing today?"
        
        print(f"\n2. Testing Analysis with sample text...")
        print(f"Text: '{sample_text}'")
        
        try:
            # Test regular Ollama API
            print(f"\n🔄 Testing Ollama API...")
            result1 = analyze_accent_from_text_ollama(sample_text)
            print(f"Result: {result1[:200]}...")
            
            # Test chat API
            print(f"\n🔄 Testing Ollama Chat API...")
            result2 = analyze_accent_from_text_ollama_chat(sample_text)
            print(f"Result: {result2[:200]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    else:
        print(f"\n❌ Ollama not available: {connection_status['message']}")
        print("\nTo use Ollama:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Run: ollama pull llama3.1")
        print("3. Start Ollama service") 