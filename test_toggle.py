#!/usr/bin/env python3
"""
Test Script: Demonstrate LLM Provider Toggle
Shows how to switch between OpenAI and Ollama providers
"""

import sys
import os
sys.path.append('.')

# Test different configurations
def test_provider_config():
    """Test and display current provider configuration"""
    
    # Import after adding to path
    from function.compute import LLM_PROVIDER, OLLAMA_MODEL, USE_FALLBACK, check_ollama_connection
    
    print("🔧 Current LLM Configuration:")
    print(f"   Provider: {LLM_PROVIDER}")
    if LLM_PROVIDER in ["ollama", "ollama_chat"]:
        print(f"   Ollama Model: {OLLAMA_MODEL}")
        print(f"   Fallback to OpenAI: {USE_FALLBACK}")
        
        # Check Ollama connection
        print("\n🔍 Checking Ollama connection...")
        status = check_ollama_connection(OLLAMA_MODEL)
        print(f"   Status: {status['message']}")
        if status.get('available_models'):
            print(f"   Available models: {', '.join(status['available_models'])}")
    
    print(f"\n📝 To change the provider, edit function/compute.py lines 8-10:")
    print(f"   LLM_PROVIDER = \"{LLM_PROVIDER}\"  # Options: \"openai\", \"ollama\", \"ollama_chat\"")
    print(f"   OLLAMA_MODEL = \"{OLLAMA_MODEL}\"  # For Ollama: llama3.1:8b, llama3, mistral, etc.")
    print(f"   USE_FALLBACK = {USE_FALLBACK}  # Fallback to OpenAI if Ollama fails")

def test_sample_analysis():
    """Test analysis with a sample text"""
    from function.compute import analyze_with_configured_provider
    
    sample_text = "Hello there, how are you doing today? I hope you're having a wonderful time."
    
    print(f"\n🧪 Testing analysis with sample text:")
    print(f"   Text: \"{sample_text}\"")
    print(f"   Provider: Currently configured for {LLM_PROVIDER}")
    
    try:
        result = analyze_with_configured_provider(sample_text)
        print(f"✅ Analysis successful!")
        print(f"   Result length: {len(result)} characters")
        print(f"   Preview: {result[:200]}...")
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")

if __name__ == "__main__":
    print("🎯 LLM Provider Toggle Test")
    print("=" * 50)
    
    test_provider_config()
    
    # Only run sample analysis if user wants to
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_sample_analysis()
    else:
        print(f"\n💡 Run with --test flag to test actual analysis:")
        print(f"   python test_toggle.py --test") 