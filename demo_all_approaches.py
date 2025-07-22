#!/usr/bin/env python3
"""
Demo: All Spam Classification Approaches
=======================================

This script demonstrates all three spam classification approaches:
1. Simple (no dependencies)
2. Lightweight ML (minimal dependencies)
3. Heavy ML (full LLM approach)

Shows performance, accuracy, and resource usage comparison.
"""

import time
import sys
from pathlib import Path

def demo_simple_classifier():
    """Demo the simple rule-based classifier"""
    print("🔧 SIMPLE CLASSIFIER (No Dependencies)")
    print("=" * 50)
    
    try:
        from simple_spam_classifier import SimpleDictSpamClassifier
        
        classifier = SimpleDictSpamClassifier()
        
        test_texts = [
            "Congratulations! You've won $1,000,000! Click here now!",
            "Hi, can we schedule a meeting tomorrow at 2 PM?",
            "URGENT: Account suspended! Verify immediately!",
            "Thank you for your purchase. Ships in 3-5 days.",
            "FREE iPhone! Limited time! Click now!!!",
            "Happy birthday! Have a wonderful day."
        ]
        
        print(f"📊 Testing {len(test_texts)} samples...")
        start_time = time.time()
        
        results = []
        for text in test_texts:
            result = classifier.classify(text)
            results.append(result)
            print(f"   {result['classification']:8} | {result['confidence']:5.1%} | {text[:40]}...")
        
        elapsed = time.time() - start_time
        speed = len(test_texts) / elapsed
        
        print(f"⚡ Speed: {speed:.0f} texts/second")
        print(f"💾 Memory: < 10MB")
        print(f"📦 Dependencies: None")
        print(f"✅ Setup time: 0 seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_lightweight_classifier():
    """Demo the lightweight ML classifier"""
    print("\n🔧 LIGHTWEIGHT ML CLASSIFIER")
    print("=" * 50)
    
    try:
        from lightweight_spam_classifier import LightweightSpamClassifier, create_training_data
        
        print("🏋️ Training model...")
        train_start = time.time()
        
        classifier = LightweightSpamClassifier(approach="ensemble")
        texts, labels = create_training_data()
        classifier.train_ml_model(texts, labels)
        
        train_time = time.time() - train_start
        
        test_texts = [
            "Congratulations! You've won $1,000,000! Click here now!",
            "Hi, can we schedule a meeting tomorrow at 2 PM?",
            "URGENT: Account suspended! Verify immediately!",
            "Thank you for your purchase. Ships in 3-5 days.",
            "FREE iPhone! Limited time! Click now!!!",
            "Happy birthday! Have a wonderful day."
        ]
        
        print(f"📊 Testing {len(test_texts)} samples...")
        start_time = time.time()
        
        results = []
        for text in test_texts:
            result = classifier.classify(text)
            results.append(result)
            print(f"   {result['classification']:8} | {result['confidence']:5.1%} | {text[:40]}...")
        
        elapsed = time.time() - start_time
        speed = len(test_texts) / elapsed
        
        print(f"⚡ Speed: {speed:.0f} texts/second")
        print(f"💾 Memory: ~50-100MB")
        print(f"📦 Dependencies: 3 packages")
        print(f"🏋️ Training time: {train_time:.1f} seconds")
        
        return True
        
    except ImportError:
        print("❌ scikit-learn not available")
        print("📦 Install with: pip install scikit-learn pandas numpy")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_heavy_classifier():
    """Demo the heavy LLM classifier"""
    print("\n🔧 HEAVY ML CLASSIFIER (LLM)")
    print("=" * 50)
    
    try:
        # Check if model exists
        model_path = Path("./models/qwen2.5-3b-instruct-q4_k_m.gguf")
        if not model_path.exists():
            print("❌ Model not found. Run setup first:")
            print("   python qwen_spam_classifier_setup.py")
            return False
        
        from qwen_spam_classifier_setup import SpamClassifier
        
        print("🤖 Loading 3B parameter model...")
        load_start = time.time()
        
        classifier = SpamClassifier(str(model_path))
        
        load_time = time.time() - load_start
        
        test_texts = [
            "Congratulations! You've won $1,000,000! Click here now!",
            "Hi, can we schedule a meeting tomorrow at 2 PM?",
            "URGENT: Account suspended! Verify immediately!"
        ]  # Fewer tests due to slower speed
        
        print(f"📊 Testing {len(test_texts)} samples...")
        start_time = time.time()
        
        results = []
        for text in test_texts:
            result = classifier.classify(text)
            results.append(result)
            print(f"   {result['classification']:8} | Text: {text[:40]}...")
        
        elapsed = time.time() - start_time
        speed = len(test_texts) / elapsed
        
        print(f"⚡ Speed: {speed:.1f} texts/second")
        print(f"💾 Memory: ~3-8GB")
        print(f"📦 Dependencies: Many packages")
        print(f"🤖 Load time: {load_time:.1f} seconds")
        
        return True
        
    except ImportError:
        print("❌ Dependencies not available")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def benchmark_approaches():
    """Benchmark all available approaches"""
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Common test data
    test_data = [
        "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
        "Hi John, can we schedule a meeting for tomorrow at 2 PM to discuss the project?",
        "URGENT: Your account will be suspended! Verify your identity immediately!",
        "Thank you for your purchase. Your order #12345 will be delivered within 3-5 business days.",
        "Buy cheap medications online! No prescription needed! Discount 90%!!!",
        "Reminder: Your appointment with Dr. Smith is scheduled for Friday at 10:30 AM.",
        "Make money fast! Work from home! Earn $5000 per week! No experience required!",
        "The quarterly report is ready for review. Please find it attached to this email.",
        "FREE iPhone 15! Limited time offer! Click now before it's too late!",
        "Could you please review the attached document and provide feedback by Friday?"
    ]
    
    expected_labels = [
        "SPAM", "NOT_SPAM", "SPAM", "NOT_SPAM", "SPAM", 
        "NOT_SPAM", "SPAM", "NOT_SPAM", "SPAM", "NOT_SPAM"
    ]
    
    results = {}
    
    # Test Simple Classifier
    try:
        from simple_spam_classifier import SimpleDictSpamClassifier
        
        classifier = SimpleDictSpamClassifier()
        start_time = time.time()
        
        predictions = []
        for text in test_data:
            result = classifier.classify(text)
            predictions.append(result['classification'])
        
        elapsed = time.time() - start_time
        accuracy = sum(1 for p, e in zip(predictions, expected_labels) if p == e) / len(expected_labels)
        
        results['Simple'] = {
            'speed': len(test_data) / elapsed,
            'accuracy': accuracy,
            'memory': '<10MB',
            'setup_time': '0s',
            'dependencies': 0
        }
        
    except Exception as e:
        print(f"Simple classifier failed: {e}")
    
    # Test Lightweight ML
    try:
        from lightweight_spam_classifier import LightweightSpamClassifier, create_training_data
        
        classifier = LightweightSpamClassifier(approach="ensemble")
        
        # Train
        train_start = time.time()
        texts, labels = create_training_data()
        classifier.train_ml_model(texts, labels)
        train_time = time.time() - train_start
        
        # Test
        start_time = time.time()
        predictions = []
        for text in test_data:
            result = classifier.classify(text)
            predictions.append(result['classification'])
        
        elapsed = time.time() - start_time
        accuracy = sum(1 for p, e in zip(predictions, expected_labels) if p == e) / len(expected_labels)
        
        results['Lightweight ML'] = {
            'speed': len(test_data) / elapsed,
            'accuracy': accuracy,
            'memory': '50-100MB',
            'setup_time': f'{train_time:.1f}s',
            'dependencies': 3
        }
        
    except Exception as e:
        print(f"Lightweight ML failed: {e}")
    
    # Print comparison table
    if results:
        print(f"\n{'Approach':<15} | {'Speed (t/s)':<12} | {'Accuracy':<8} | {'Memory':<10} | {'Setup':<8} | {'Deps':<5}")
        print("-" * 70)
        
        for name, data in results.items():
            print(f"{name:<15} | {data['speed']:>9.0f}   | {data['accuracy']:>6.1%}   | {data['memory']:<10} | {data['setup_time']:<8} | {data['dependencies']:<5}")

def show_resource_recommendations():
    """Show recommendations based on available resources"""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    print("Choose your approach based on available resources:")
    print()
    print("🚀 SIMPLE CLASSIFIER (Recommended for most users)")
    print("   Use when: Limited resources, fast processing needed")
    print("   Command: python3 simple_spam_classifier.py --test")
    print("   Pros: Zero setup, very fast, no dependencies")
    print("   Cons: Lower accuracy than ML approaches")
    print()
    print("⚖️ LIGHTWEIGHT ML")
    print("   Use when: Can install packages, want better accuracy")
    print("   Command: python3 lightweight_spam_classifier.py")
    print("   Pros: Good accuracy, reasonable speed, small memory")
    print("   Cons: Requires scikit-learn installation")
    print()
    print("🎯 HEAVY ML (LLM)")
    print("   Use when: Have powerful hardware, need best accuracy")
    print("   Command: python3 qwen_spam_classifier_setup.py")
    print("   Pros: Highest accuracy, context understanding")
    print("   Cons: Large download, slow, high memory usage")

def main():
    """Main demo function"""
    print("🚀 Spam Classification Approaches Demo")
    print("=" * 60)
    print("This demo shows three different approaches for spam classification")
    print("optimized for different resource constraints.")
    print()
    
    # Demo each approach
    simple_works = demo_simple_classifier()
    lightweight_works = demo_lightweight_classifier()
    heavy_works = demo_heavy_classifier()
    
    # Benchmark if any work
    if simple_works or lightweight_works or heavy_works:
        benchmark_approaches()
    
    # Show recommendations
    show_resource_recommendations()
    
    print("\n🎯 QUICK START")
    print("For immediate use with no setup:")
    print("   python3 simple_spam_classifier.py --interactive")
    print()
    print("For batch processing:")
    print("   python3 simple_spam_classifier.py input.csv output.csv")

if __name__ == "__main__":
    main()