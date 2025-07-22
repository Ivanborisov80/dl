#!/usr/bin/env python3
"""
Simple Spam/Junk Classifier for Limited Resources (Pure Python)
==============================================================

This script provides a lightweight spam classification system that uses only
Python standard library - no external dependencies required!

Perfect for:
- Limited computational resources
- Systems without package installation permissions
- Quick deployment scenarios
- Educational purposes

Features:
- Rule-based classification
- Keyword matching
- Pattern detection
- Fast processing
- Memory efficient
"""

import re
import csv
import json
import sys
import time
from typing import List, Dict, Any, Tuple
from collections import Counter
import os

class SimpleDictSpamClassifier:
    """
    Simple spam classifier using only Python standard library
    """
    
    def __init__(self):
        """Initialize the classifier with predefined rules and keywords"""
        
        # Spam indicators with weights
        self.spam_patterns = {
            # Money/Financial
            'money_keywords': {
                'patterns': ['money', 'cash', 'earn', 'income', 'profit', 'rich', 'wealth', 
                           'salary', 'dollar', '$', 'investment', 'bitcoin', 'crypto'],
                'weight': 2
            },
            
            # Urgency indicators
            'urgency_keywords': {
                'patterns': ['urgent', 'immediate', 'now', 'hurry', 'limited time', 
                           'expires', 'deadline', 'act now', 'don\'t wait'],
                'weight': 2
            },
            
            # Offers and deals
            'offer_keywords': {
                'patterns': ['free', 'discount', 'offer', 'deal', 'sale', 'promotion', 
                           'bonus', 'gift', 'prize', 'winner', 'selected'],
                'weight': 2
            },
            
            # Suspicious actions
            'suspicious_keywords': {
                'patterns': ['click here', 'verify', 'confirm', 'update', 'suspended', 
                           'blocked', 'download', 'install', 'claim'],
                'weight': 3
            },
            
            # Medical/Health spam
            'medical_keywords': {
                'patterns': ['pills', 'medication', 'viagra', 'pharmacy', 'prescription', 
                           'drugs', 'weight loss', 'diet', 'supplement'],
                'weight': 3
            },
            
            # Work from home
            'work_from_home': {
                'patterns': ['work from home', 'make money', 'no experience', 'easy money',
                           'part time', 'full time', 'opportunity'],
                'weight': 2
            },
            
            # Excessive punctuation
            'excessive_punctuation': {
                'patterns': ['!!!', '???', '!!!!!', '?????'],
                'weight': 1
            }
        }
        
        # Legitimate indicators
        self.legitimate_patterns = {
            'business_keywords': {
                'patterns': ['meeting', 'appointment', 'schedule', 'project', 'report', 
                           'document', 'presentation', 'conference'],
                'weight': 2
            },
            
            'personal_keywords': {
                'patterns': ['birthday', 'family', 'friend', 'regards', 'thank you', 
                           'please', 'vacation', 'holiday'],
                'weight': 1
            },
            
            'formal_keywords': {
                'patterns': ['regarding', 'attached', 'review', 'feedback', 'confirm', 
                           'assistance', 'sincerely', 'best regards'],
                'weight': 2
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs (simple pattern)
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'www\.\S+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract various features from text"""
        clean_text = self.preprocess_text(text)
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_count': sum(1 for c in text if c.isupper()),
            'digit_count': sum(1 for c in text if c.isdigit()),
            'has_url': bool(re.search(r'http[s]?://|www\.', text.lower())),
            'has_email': bool(re.search(r'\S+@\S+', text)),
            'has_phone': bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)),
        }
        
        # Calculate ratios
        features['caps_ratio'] = features['caps_count'] / max(len(text), 1)
        features['digit_ratio'] = features['digit_count'] / max(len(text), 1)
        
        return features
    
    def calculate_spam_score(self, text: str) -> Tuple[float, List[str]]:
        """Calculate spam score and provide explanations"""
        clean_text = self.preprocess_text(text)
        features = self.extract_features(text)
        
        spam_score = 0
        explanations = []
        
        # Check spam patterns
        for category, config in self.spam_patterns.items():
            patterns = config['patterns']
            weight = config['weight']
            
            matches = sum(1 for pattern in patterns if pattern in clean_text)
            if matches > 0:
                score = matches * weight
                spam_score += score
                explanations.append(f"{category}: {matches} matches (score: +{score})")
        
        # Check for excessive punctuation
        if features['exclamation_count'] >= 3:
            spam_score += 2
            explanations.append(f"Excessive exclamation marks: {features['exclamation_count']} (score: +2)")
        
        if features['question_count'] >= 3:
            spam_score += 1
            explanations.append(f"Many question marks: {features['question_count']} (score: +1)")
        
        # Check for excessive caps
        if features['caps_ratio'] > 0.3:
            spam_score += 2
            explanations.append(f"Too many capitals: {features['caps_ratio']:.1%} (score: +2)")
        
        # Check for URLs with other spam indicators
        if features['has_url'] and spam_score > 0:
            spam_score += 1
            explanations.append("URL with spam indicators (score: +1)")
        
        return spam_score, explanations
    
    def calculate_legitimate_score(self, text: str) -> Tuple[float, List[str]]:
        """Calculate legitimate score"""
        clean_text = self.preprocess_text(text)
        
        legit_score = 0
        explanations = []
        
        # Check legitimate patterns
        for category, config in self.legitimate_patterns.items():
            patterns = config['patterns']
            weight = config['weight']
            
            matches = sum(1 for pattern in patterns if pattern in clean_text)
            if matches > 0:
                score = matches * weight
                legit_score += score
                explanations.append(f"{category}: {matches} matches (score: +{score})")
        
        return legit_score, explanations
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify text as spam or not spam"""
        if not text or str(text).strip() == "":
            return {
                "text": text,
                "classification": "UNKNOWN",
                "confidence": 0.0,
                "spam_score": 0,
                "legitimate_score": 0,
                "explanation": "Empty or invalid text"
            }
        
        # Calculate scores
        spam_score, spam_explanations = self.calculate_spam_score(text)
        legit_score, legit_explanations = self.calculate_legitimate_score(text)
        
        # Decision logic
        if spam_score >= 5:  # High spam threshold
            classification = "SPAM"
            confidence = min(0.95, 0.6 + (spam_score - 5) * 0.05)
        elif spam_score >= 3 and legit_score < 2:  # Medium spam threshold
            classification = "SPAM"
            confidence = 0.7
        elif legit_score >= 3 and spam_score < 2:  # High legitimate threshold
            classification = "NOT_SPAM"
            confidence = 0.8
        elif legit_score > spam_score:  # More legitimate than spam
            classification = "NOT_SPAM"
            confidence = 0.6
        elif spam_score > legit_score * 1.5:  # Significantly more spam indicators
            classification = "SPAM"
            confidence = 0.65
        else:
            # Default to not spam with low confidence
            classification = "NOT_SPAM"
            confidence = 0.5
        
        # Combine explanations
        all_explanations = []
        if spam_explanations:
            all_explanations.append("SPAM indicators: " + "; ".join(spam_explanations[:3]))
        if legit_explanations:
            all_explanations.append("LEGITIMATE indicators: " + "; ".join(legit_explanations[:2]))
        
        explanation = " | ".join(all_explanations) if all_explanations else "No strong indicators found"
        
        return {
            "text": text,
            "classification": classification,
            "confidence": round(confidence, 3),
            "spam_score": spam_score,
            "legitimate_score": legit_score,
            "explanation": explanation
        }

def process_csv_file(input_file: str, output_file: str, text_column: str = 'text'):
    """Process a CSV file and classify each text entry"""
    classifier = SimpleDictSpamClassifier()
    
    print(f"📁 Processing file: {input_file}")
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"✅ Loaded {len(rows)} rows")
        
        if text_column not in rows[0]:
            print(f"❌ Column '{text_column}' not found!")
            print(f"Available columns: {list(rows[0].keys())}")
            return
        
        # Process each row
        results = []
        start_time = time.time()
        
        for i, row in enumerate(rows, 1):
            text = row.get(text_column, '')
            result = classifier.classify(text)
            
            # Add result columns
            row['predicted_classification'] = result['classification']
            row['confidence_score'] = result['confidence']
            row['spam_score'] = result['spam_score']
            row['legitimate_score'] = result['legitimate_score']
            row['explanation'] = result['explanation']
            
            # Check accuracy if labels exist
            if 'label' in row:
                row['correct'] = row['label'] == result['classification']
            
            results.append(row)
            
            if i % 100 == 0:
                print(f"   Processed {i}/{len(rows)} rows...")
        
        processing_time = time.time() - start_time
        speed = len(rows) / processing_time
        
        print(f"⚡ Processing completed in {processing_time:.1f}s ({speed:.1f} texts/sec)")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        print(f"💾 Results saved to: {output_file}")
        
        # Print statistics
        spam_count = sum(1 for r in results if r['predicted_classification'] == 'SPAM')
        not_spam_count = sum(1 for r in results if r['predicted_classification'] == 'NOT_SPAM')
        unknown_count = len(results) - spam_count - not_spam_count
        
        print(f"\n📈 Classification Summary:")
        print(f"   SPAM: {spam_count} ({spam_count/len(results):.1%})")
        print(f"   NOT_SPAM: {not_spam_count} ({not_spam_count/len(results):.1%})")
        print(f"   UNKNOWN: {unknown_count} ({unknown_count/len(results):.1%})")
        print(f"   Speed: {speed:.1f} texts/second")
        
        # Show accuracy if labels available
        if 'label' in results[0]:
            correct = sum(1 for r in results if r.get('correct', False))
            accuracy = correct / len(results)
            print(f"   Accuracy: {correct}/{len(results)} = {accuracy:.1%}")
            
    except Exception as e:
        print(f"❌ Error processing file: {e}")

def interactive_mode():
    """Interactive classification mode"""
    classifier = SimpleDictSpamClassifier()
    
    print("\n💬 Interactive Spam Classifier")
    print("Enter text to classify (type 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                result = classifier.classify(user_input)
                print(f"\n📋 Classification: {result['classification']}")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Spam Score: {result['spam_score']}")
                print(f"   Legitimate Score: {result['legitimate_score']}")
                print(f"   Explanation: {result['explanation']}")
        
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break

def test_classifier():
    """Test the classifier with sample data"""
    classifier = SimpleDictSpamClassifier()
    
    test_cases = [
        ("Congratulations! You've won $1,000,000! Click here to claim now!", "SPAM"),
        ("Hi John, can we schedule a meeting tomorrow at 2 PM?", "NOT_SPAM"),
        ("URGENT: Account suspended! Verify immediately!", "SPAM"),
        ("Thank you for your purchase. Order #12345 ships in 3-5 days.", "NOT_SPAM"),
        ("FREE iPhone! Limited time! Click now!!!", "SPAM"),
        ("Happy birthday! Hope you have a wonderful day.", "NOT_SPAM"),
        ("Make $5000 weekly from home! No experience needed!", "SPAM"),
        ("The quarterly report is attached for your review.", "NOT_SPAM")
    ]
    
    print("🧪 Testing classifier with sample data:")
    print("=" * 60)
    
    correct = 0
    for text, expected in test_cases:
        result = classifier.classify(text)
        actual = result['classification']
        is_correct = expected == actual
        
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Expected: {expected}, Got: {actual}")
        print(f"   Text: {text[:50]}...")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Explanation: {result['explanation'][:80]}...")
        print()
    
    accuracy = correct / len(test_cases)
    print(f"📊 Test Accuracy: {correct}/{len(test_cases)} = {accuracy:.1%}")

def main():
    """Main function"""
    if len(sys.argv) == 1:
        # No arguments - run tests and interactive mode
        print("🚀 Simple Spam Classifier (Pure Python)")
        print("=" * 50)
        test_classifier()
        interactive_mode()
    
    elif len(sys.argv) == 2 and sys.argv[1] in ['--test', '-t']:
        # Test mode only
        test_classifier()
    
    elif len(sys.argv) == 2 and sys.argv[1] in ['--interactive', '-i']:
        # Interactive mode only
        interactive_mode()
    
    elif len(sys.argv) >= 3:
        # CSV processing mode
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        text_column = sys.argv[3] if len(sys.argv) > 3 else 'text'
        
        if not os.path.exists(input_file):
            print(f"❌ Input file '{input_file}' not found!")
            return
        
        process_csv_file(input_file, output_file, text_column)
    
    else:
        print("Usage:")
        print("  python simple_spam_classifier.py                    # Test + Interactive mode")
        print("  python simple_spam_classifier.py --test             # Test mode only")
        print("  python simple_spam_classifier.py --interactive      # Interactive mode only")
        print("  python simple_spam_classifier.py input.csv output.csv [text_column]")
        print("  python simple_spam_classifier.py example_data.csv results.csv")

if __name__ == "__main__":
    main()