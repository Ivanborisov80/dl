#!/usr/bin/env python3
"""
Qwen2.5 3B Spam/Junk Classification Setup
=========================================

This script sets up and demonstrates how to use a Qwen2.5 3B model for 
binary classification (spam/junk vs not spam/junk) using GGUF format
for efficient inference.

Since the specific RefalMachine model doesn't exist, we'll use the 
official Qwen2.5-3B-Instruct model which is excellent for classification tasks.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

def install_requirements():
    """Install required packages"""
    packages = [
        "huggingface_hub",
        "llama-cpp-python",
        "transformers",
        "torch",
        "datasets",
        "scikit-learn",
        "numpy",
        "pandas"
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def download_model():
    """Download the Qwen2.5-3B-Instruct GGUF model"""
    from huggingface_hub import hf_hub_download
    
    # We'll use a quantized version for better performance
    model_id = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    filename = "qwen2.5-3b-instruct-q4_k_m.gguf"
    
    print(f"Downloading {model_id}/{filename}...")
    
    try:
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir="./models",
            local_dir_use_symlinks=False
        )
        print(f"✓ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return None

class SpamClassifier:
    """Spam/Junk classifier using Qwen2.5-3B"""
    
    def __init__(self, model_path: str):
        """Initialize the classifier with the model path"""
        try:
            from llama_cpp import Llama
            
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,  # Context window
                n_threads=4,  # Adjust based on your CPU
                verbose=False
            )
            print("✓ Model loaded successfully")
            
        except ImportError:
            print("✗ llama-cpp-python not installed. Please run install_requirements()")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def create_prompt(self, text: str) -> str:
        """Create a classification prompt"""
        prompt = f"""<|im_start|>system
You are a spam/junk classifier. Analyze the given text and classify it as either "SPAM" or "NOT_SPAM".

Rules:
- SPAM: Promotional content, phishing attempts, suspicious links, unsolicited offers, repetitive content
- NOT_SPAM: Legitimate messages, personal communication, informational content, business correspondence

Respond with only "SPAM" or "NOT_SPAM" followed by a brief explanation.
<|im_end|>
<|im_start|>user
Classify this text: "{text}"
<|im_end|>
<|im_start|>assistant"""
        
        return prompt
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify a single text as spam or not spam"""
        prompt = self.create_prompt(text)
        
        try:
            response = self.llm(
                prompt,
                max_tokens=100,
                temperature=0.1,  # Low temperature for consistent classification
                top_p=0.9,
                stop=["<|im_end|>"],
                echo=False
            )
            
            response_text = response['choices'][0]['text'].strip()
            
            # Parse the response
            is_spam = "SPAM" in response_text.upper() and "NOT_SPAM" not in response_text.upper()
            
            return {
                "text": text,
                "classification": "SPAM" if is_spam else "NOT_SPAM",
                "confidence": response['choices'][0].get('logprobs', {}).get('top_logprobs', []),
                "explanation": response_text,
                "raw_response": response_text
            }
            
        except Exception as e:
            return {
                "text": text,
                "classification": "ERROR",
                "error": str(e),
                "explanation": f"Error during classification: {e}"
            }
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts"""
        results = []
        for i, text in enumerate(texts, 1):
            print(f"Processing {i}/{len(texts)}...")
            result = self.classify(text)
            results.append(result)
        return results

def create_sample_data() -> List[Dict[str, str]]:
    """Create sample data for testing"""
    return [
        {
            "text": "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
            "expected": "SPAM"
        },
        {
            "text": "Hi John, can we schedule a meeting for tomorrow at 2 PM to discuss the project?",
            "expected": "NOT_SPAM"
        },
        {
            "text": "URGENT: Your account will be suspended! Verify your identity immediately by clicking this link.",
            "expected": "SPAM"
        },
        {
            "text": "Thank you for your purchase. Your order #12345 will be delivered within 3-5 business days.",
            "expected": "NOT_SPAM"
        },
        {
            "text": "Buy cheap medications online! No prescription needed! Discount 90%!!!",
            "expected": "SPAM"
        },
        {
            "text": "Reminder: Your appointment with Dr. Smith is scheduled for Friday at 10:30 AM.",
            "expected": "NOT_SPAM"
        },
        {
            "text": "Make money fast! Work from home! Earn $5000 per week! No experience required!",
            "expected": "SPAM"
        },
        {
            "text": "The quarterly report is ready for review. Please find it attached to this email.",
            "expected": "NOT_SPAM"
        }
    ]

def evaluate_classifier(classifier: SpamClassifier, test_data: List[Dict[str, str]]):
    """Evaluate the classifier on test data"""
    print("\n" + "="*60)
    print("EVALUATING CLASSIFIER")
    print("="*60)
    
    correct = 0
    total = len(test_data)
    
    for item in test_data:
        result = classifier.classify(item["text"])
        expected = item["expected"]
        actual = result["classification"]
        
        is_correct = expected == actual
        if is_correct:
            correct += 1
        
        print(f"\nText: {item['text'][:50]}...")
        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print(f"Explanation: {result['explanation']}")
        print("-" * 40)
    
    accuracy = correct / total
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")

def main():
    """Main function to demonstrate the spam classifier"""
    print("Qwen2.5 3B Spam/Junk Classifier Setup")
    print("=" * 50)
    
    # Install requirements
    print("\n1. Installing requirements...")
    install_requirements()
    
    # Download model
    print("\n2. Downloading model...")
    model_path = download_model()
    if not model_path:
        print("Failed to download model. Exiting.")
        return
    
    # Initialize classifier
    print("\n3. Loading classifier...")
    classifier = SpamClassifier(model_path)
    
    # Test with sample data
    print("\n4. Testing with sample data...")
    test_data = create_sample_data()
    evaluate_classifier(classifier, test_data)
    
    # Interactive mode
    print("\n5. Interactive mode (type 'quit' to exit):")
    while True:
        try:
            user_input = input("\nEnter text to classify: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                result = classifier.classify(user_input)
                print(f"\nClassification: {result['classification']}")
                print(f"Explanation: {result['explanation']}")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()