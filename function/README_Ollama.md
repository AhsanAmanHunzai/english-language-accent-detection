# 🦙 Ollama Integration for Accent Detection

This document explains how to use the new Ollama Llama models for accent analysis instead of OpenAI GPT-4.

## 📋 **Functions Available**

### 1. **Core Analysis Functions**
- `analyze_accent_from_text_ollama()` - Standard Ollama API
- `analyze_accent_from_text_ollama_chat()` - Chat API format (alternative)
- `analyze_accent_with_provider()` - Unified function (OpenAI/Ollama)

### 2. **Utility Functions**
- `check_ollama_connection()` - Test Ollama connection and models
- Configuration variables for easy switching

## 🚀 **Setup Instructions**

### **Step 1: Install Ollama**
```bash
# Visit https://ollama.ai/ and install Ollama
# OR on Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh
```

### **Step 2: Pull Llama Models**
```bash
# Pull the default model (recommended)
ollama pull llama3.1

# OR other models:
ollama pull llama3
ollama pull codellama
ollama pull mistral
```

### **Step 3: Start Ollama Service**
```bash
# Start Ollama (runs on localhost:11434)
ollama serve

# OR if using Docker:
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## 💻 **Usage Examples**

### **Basic Usage**
```python
from function.ollama_analysis import analyze_accent_from_text_ollama

# Analyze text using Ollama
text = "Hello, how are you doing today? I hope everything is going well."
result = analyze_accent_from_text_ollama(text, model="llama3.1")
print(result)
```

### **Check Connection First**
```python
from function.ollama_analysis import check_ollama_connection

# Test if Ollama is running
status = check_ollama_connection()
if status["status"] == "connected":
    print(f"✅ Ollama is ready with {len(status['available_models'])} models")
    print(f"Available: {status['available_models']}")
else:
    print(f"❌ {status['message']}")
```

### **Unified Provider Function**
```python
from function.ollama_analysis import analyze_accent_with_provider

# Use OpenAI (default)
result1 = analyze_accent_with_provider(text, provider="openai")

# Use Ollama
result2 = analyze_accent_with_provider(text, provider="ollama", model="llama3.1")

# Use Ollama Chat API
result3 = analyze_accent_with_provider(text, provider="ollama_chat", model="llama3.1")
```

### **Integration with Main App**
```python
# In function/compute.py, you can replace:
gpt_analysis = analyze_accent_from_text(transcribed_text)

# With:
from .ollama_analysis import analyze_accent_from_text_ollama
ollama_analysis = analyze_accent_from_text_ollama(transcribed_text, "llama3.1")
```

## ⚙️ **Configuration**

### **Change Default Provider**
```python
# In ollama_analysis.py
LLM_PROVIDER = "ollama"  # Switch from "openai" to "ollama"
OLLAMA_MODEL = "llama3"  # Change default model
```

### **Available Models**
- `llama3.1` (recommended) - Latest Llama model
- `llama3` - Stable Llama 3 model  
- `codellama` - Code-specialized model
- `mistral` - Alternative high-performance model
- `llama2` - Previous generation

## 📊 **Performance Comparison**

| Provider | Speed | Cost | Quality | Local |
|----------|-------|------|---------|-------|
| OpenAI GPT-4 | Fast | $$ | Excellent | ❌ |
| Ollama Llama3.1 | Medium | Free | Very Good | ✅ |
| Ollama Llama3 | Medium | Free | Good | ✅ |

## 🔧 **Troubleshooting**

### **Common Issues**

**1. Connection Error**
```
Could not connect to Ollama. Make sure Ollama is running on localhost:11434
```
**Solution:** Start Ollama service: `ollama serve`

**2. Model Not Found**
```
Ollama API error: 404 - model not found
```
**Solution:** Pull the model: `ollama pull llama3.1`

**3. Timeout Errors**
```
Ollama request timed out
```
**Solution:** Increase timeout or use smaller model

### **Test Connection**
```bash
# Test if Ollama is running
curl http://localhost:11434/api/tags

# Should return JSON with available models
```

## 🎯 **Recommendations**

### **For Production**
- Use `llama3.1` for best quality
- Set up proper error handling
- Monitor response times
- Consider GPU acceleration for speed

### **For Development**
- Use `llama3` for faster responses
- Test with `check_ollama_connection()` first
- Keep OpenAI as fallback option

### **For Resources**
- **High RAM**: Use `llama3.1` (8GB+ RAM)
- **Low RAM**: Use `llama3` or `mistral` (4GB+ RAM)
- **GPU Available**: Any model will be much faster

## 📝 **Example Output**

```
1. **Accent Classification**: Midwestern American English

2. **Confidence Score**: 75% - Based on neutral pronunciation patterns and standard grammar usage typical of American Midwest.

3. **Proficiency Level**: Native - Complex sentence structures and natural idiomatic expressions.

4. **Analysis**: 
   - Word choices: Standard American vocabulary
   - Grammar patterns: Perfect subject-verb agreement
   - Regional expressions: Neutral, non-regional specific
   - No spelling variations detected

5. **Speech Quality**: 
   - Grammar accuracy: Excellent (100%)
   - Vocabulary appropriateness: Native level
   - Sentence complexity: Advanced structures
   - Overall communication effectiveness: Highly effective
```

## 🔄 **Migration Guide**

To switch from OpenAI to Ollama in your existing app:

1. **Install Ollama** and pull models
2. **Import Ollama functions** in your compute.py
3. **Replace function calls**:
   ```python
   # Before
   analysis = analyze_accent_from_text(text)
   
   # After  
   analysis = analyze_accent_from_text_ollama(text, "llama3.1")
   ```
4. **Update status messages** to mention Ollama instead of GPT-4
5. **Test thoroughly** with sample texts

The rest of your application (parsing, UI, etc.) remains exactly the same! 