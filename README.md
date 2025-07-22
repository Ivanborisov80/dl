# Qwen2.5 3B Spam/Junk Classification Setup

## Overview

This repository provides a complete setup for using Qwen2.5 3B model for binary classification tasks (spam vs not spam, junk vs legitimate content). 

**Note**: The specific model `RefalMachine/ruadapt_qwen2.5_3B_ext_u48_instruct_v4_gguf` mentioned in your request doesn't exist on Hugging Face. Instead, we'll use the official `Qwen/Qwen2.5-3B-Instruct-GGUF` model, which is excellent for classification tasks.

## Features

- ✅ Easy one-script setup
- ✅ Automatic model download
- ✅ GGUF format for efficient inference
- ✅ Batch processing support
- ✅ Interactive classification mode
- ✅ Evaluation metrics
- ✅ Sample data included
- ✅ CPU-optimized inference

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone or download the setup script
python qwen_spam_classifier_setup.py
```

This script will:
1. Install all required dependencies
2. Download the Qwen2.5-3B-Instruct GGUF model
3. Set up the classifier
4. Run evaluation on sample data
5. Start interactive mode

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install huggingface_hub llama-cpp-python transformers torch datasets scikit-learn numpy pandas

# 2. Download model manually
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    filename="qwen2.5-3b-instruct-q4_k_m.gguf",
    local_dir="./models"
)
```

## System Requirements

- **RAM**: 4-8 GB (depending on quantization level)
- **Storage**: 2-3 GB for model files
- **CPU**: Modern multi-core processor (GPU optional)
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher

## Model Options

The setup script uses `qwen2.5-3b-instruct-q4_k_m.gguf` by default, but you can choose different quantization levels:

| Quantization | Size | RAM Usage | Quality | Speed |
|-------------|------|-----------|---------|-------|
| q2_k        | ~1.2GB | ~2GB     | Lower   | Fastest |
| q4_k_m      | ~1.9GB | ~3GB     | Good    | Fast |
| q5_k_m      | ~2.3GB | ~4GB     | Better  | Medium |
| q8_0        | ~3.2GB | ~5GB     | Best    | Slower |

## Usage Examples

### Basic Classification

```python
from qwen_spam_classifier_setup import SpamClassifier

# Initialize classifier
classifier = SpamClassifier("./models/qwen2.5-3b-instruct-q4_k_m.gguf")

# Classify single text
result = classifier.classify("Win $1000 now! Click here!")
print(result["classification"])  # "SPAM"
print(result["explanation"])     # Detailed explanation
```

### Batch Classification

```python
texts = [
    "Congratulations! You've won $1,000,000!",
    "Meeting scheduled for 2 PM tomorrow",
    "Buy cheap medications online!",
    "Your order has been shipped"
]

results = classifier.classify_batch(texts)
for result in results:
    print(f"{result['text'][:30]}... -> {result['classification']}")
```

### Custom Data Processing

```python
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Classify
classifications = []
for text in df['message']:
    result = classifier.classify(text)
    classifications.append({
        'original_text': text,
        'classification': result['classification'],
        'confidence': result['explanation']
    })

# Save results
results_df = pd.DataFrame(classifications)
results_df.to_csv("classified_results.csv", index=False)
```

## Customization

### Modify Classification Prompt

You can customize the classification behavior by modifying the prompt in the `create_prompt` method:

```python
def create_prompt(self, text: str) -> str:
    prompt = f"""<|im_start|>system
You are an expert content moderator. Classify the following text as either "SPAM" or "NOT_SPAM".

Custom rules:
- SPAM: Unsolicited commercial messages, phishing, scams, excessive promotions
- NOT_SPAM: Personal messages, legitimate business communication, informational content

Consider context and intent. Respond with classification and reasoning.
<|im_end|>
<|im_start|>user
Analyze: "{text}"
<|im_end|>
<|im_start|>assistant"""
    
    return prompt
```

### Adjust Model Parameters

```python
# In the SpamClassifier __init__ method
self.llm = Llama(
    model_path=model_path,
    n_ctx=4096,          # Larger context window
    n_threads=8,         # More CPU threads
    n_gpu_layers=32,     # Use GPU if available
    verbose=False
)
```

### Fine-tuning for Your Data

For better performance on your specific use case, consider:

1. **Few-shot prompting**: Add examples in the prompt
2. **Domain-specific rules**: Modify the classification criteria
3. **Post-processing**: Add confidence thresholds or rule-based filters

## Alternative Models

If you need different capabilities, consider these alternatives:

### 1. Larger Qwen Models
```python
# Qwen2.5-7B-Instruct (better accuracy, more resources)
model_id = "Qwen/Qwen2.5-7B-Instruct-GGUF"
filename = "qwen2.5-7b-instruct-q4_k_m.gguf"
```

### 2. Russian-Adapted Models
```python
# For Russian text classification
model_id = "ai-forever/ruGPT-3.5-13B"  # Requires different setup
```

### 3. Specialized Classification Models
```python
# For transformer-based classification
from transformers import pipeline
classifier = pipeline("text-classification", 
                     model="martin-ha/toxic-comment-model")
```

## Performance Optimization

### CPU Optimization
```bash
# Set CPU threads for better performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### GPU Acceleration (if available)
```python
# Install CUDA-enabled version
pip uninstall llama-cpp-python
pip install llama-cpp-python --extra-index-url https://download.pytorch.org/whl/cu118

# Use GPU layers
n_gpu_layers=32  # Adjust based on your GPU memory
```

### Memory Management
```python
# For large datasets, process in batches
def process_large_dataset(texts, batch_size=100):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = classifier.classify_batch(batch)
        results.extend(batch_results)
        
        # Optional: clear memory
        import gc
        gc.collect()
    
    return results
```

## Troubleshooting

### Common Issues

1. **Model download fails**
   ```bash
   # Manual download
   huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ./models
   ```

2. **Out of memory errors**
   - Use smaller quantization (q2_k)
   - Reduce context window (`n_ctx=1024`)
   - Process smaller batches

3. **Slow inference**
   - Increase CPU threads (`n_threads=8`)
   - Use GPU acceleration
   - Use smaller model or higher quantization

4. **Import errors**
   ```bash
   # Reinstall with specific versions
   pip install llama-cpp-python==0.2.11
   pip install transformers==4.35.0
   ```

### Getting Help

- Check model compatibility: [Qwen documentation](https://github.com/QwenLM/Qwen)
- llama.cpp issues: [GitHub Issues](https://github.com/ggerganov/llama.cpp/issues)
- Hugging Face support: [Community forums](https://discuss.huggingface.co/)

## Example Use Cases

### 1. Email Spam Detection
```python
emails = [
    "Subject: Meeting tomorrow\nHi team, let's meet at 2 PM to discuss the project.",
    "Subject: URGENT! Claim your prize\nCongratulations! You've won $10,000!"
]

for email in emails:
    result = classifier.classify(email)
    print(f"Email: {result['classification']}")
```

### 2. Social Media Content Moderation
```python
posts = [
    "Check out this amazing product! Get 90% off now!",
    "Had a great day at the beach with family",
    "CLICK HERE FOR FREE MONEY!!!"
]

spam_posts = []
for post in posts:
    if classifier.classify(post)["classification"] == "SPAM":
        spam_posts.append(post)

print(f"Found {len(spam_posts)} spam posts")
```

### 3. Customer Support Ticket Filtering
```python
tickets = [
    "My order hasn't arrived yet, can you help?",
    "BUY NOW! Limited time offer!",
    "I need help with my account settings"
]

legitimate_tickets = [
    ticket for ticket in tickets 
    if classifier.classify(ticket)["classification"] == "NOT_SPAM"
]
```

## Contributing

Feel free to submit issues, feature requests, or improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is provided under the MIT License. The Qwen2.5 model follows its own license terms from Alibaba Cloud.

## Acknowledgments

- **Qwen Team** at Alibaba Cloud for the excellent Qwen2.5 models
- **llama.cpp** team for the efficient GGUF inference engine
- **Hugging Face** for the model hosting and libraries