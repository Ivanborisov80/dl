#!/bin/bash

# Qwen2.5 3B Spam Classifier Quick Start Script
# This script sets up everything needed to run the spam classifier

echo "🚀 Qwen2.5 3B Spam Classifier Quick Start"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if model exists, if not run the setup
if [ ! -f "models/qwen2.5-3b-instruct-q4_k_m.gguf" ]; then
    echo "🤖 Model not found. Running setup..."
    python3 qwen_spam_classifier_setup.py
else
    echo "✅ Model found. Starting classifier..."
    # Create a simple interactive script
    python3 -c "
from qwen_spam_classifier_setup import SpamClassifier

print('🎯 Loading Qwen2.5 3B Spam Classifier...')
classifier = SpamClassifier('./models/qwen2.5-3b-instruct-q4_k_m.gguf')

print('\n📝 Interactive Spam Classification')
print('Type your text to classify, or \"quit\" to exit.\n')

while True:
    try:
        text = input('Enter text: ').strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        if text:
            result = classifier.classify(text)
            print(f'🔍 Classification: {result[\"classification\"]}')
            print(f'💭 Explanation: {result[\"explanation\"]}\n')
    except KeyboardInterrupt:
        print('\n👋 Goodbye!')
        break
"
fi

echo "🎉 Setup complete!"