#!/usr/bin/env python3
"""
Lightweight Spam/Junk Classifier for Limited Resources
=====================================================

This script provides multiple lightweight approaches for spam classification
that work well on limited computational resources:

1. Traditional ML with TF-IDF + SVM/Naive Bayes
2. Simple rule-based classifier
3. Keyword-based approach
4. Ensemble method combining all approaches

All methods are designed to be fast, memory-efficient, and work without GPU.
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import string
from collections import Counter

# Traditional ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available. Only rule-based classification will work.")

class LightweightSpamClassifier:
    """
    Resource-efficient spam classifier using multiple lightweight approaches
    """
    
    def __init__(self, approach: str = "ensemble"):
        """
        Initialize classifier with specified approach
        
        Args:
            approach: "rules", "keywords", "ml", or "ensemble"
        """
        self.approach = approach
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        # Spam keywords and patterns
        self.spam_keywords = {
            'money': ['money', 'cash', 'earn', 'income', 'profit', 'rich', 'wealth', 'salary'],
            'urgency': ['urgent', 'immediate', 'now', 'hurry', 'limited time', 'expires', 'deadline'],
            'offers': ['free', 'discount', 'offer', 'deal', 'sale', 'promotion', 'bonus'],
            'suspicious': ['click here', 'verify', 'confirm', 'update', 'suspended', 'blocked'],
            'medical': ['pills', 'medication', 'viagra', 'pharmacy', 'prescription', 'drugs'],
            'winners': ['winner', 'won', 'congratulations', 'prize', 'lottery', 'selected'],
            'work_from_home': ['work from home', 'make money', 'no experience', 'easy money'],
            'excessive_punctuation': ['!!!', '???', '!!!']
        }
        
        # Legitimate keywords
        self.legitimate_keywords = {
            'business': ['meeting', 'appointment', 'schedule', 'project', 'report', 'document'],
            'personal': ['birthday', 'family', 'friend', 'regards', 'thank you', 'please'],
            'formal': ['regarding', 'attached', 'review', 'feedback', 'confirm', 'assistance']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features for rule-based classification"""
        text_lower = text.lower()
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'has_url': bool(re.search(r'http[s]?://', text_lower)),
            'has_email': bool(re.search(r'\S+@\S+', text_lower)),
            'has_phone': bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text_lower)),
        }
        
        # Count spam keywords
        spam_score = 0
        for category, keywords in self.spam_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'spam_{category}_count'] = count
            spam_score += count * 2  # Weight spam keywords heavily
        
        # Count legitimate keywords
        legit_score = 0
        for category, keywords in self.legitimate_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'legit_{category}_count'] = count
            legit_score += count
        
        features['spam_score'] = spam_score
        features['legit_score'] = legit_score
        features['spam_legit_ratio'] = spam_score / max(legit_score, 1)
        
        return features
    
    def classify_by_rules(self, text: str) -> Tuple[str, float, str]:
        """Rule-based classification"""
        features = self.extract_features(text)
        
        reasons = []
        spam_indicators = 0
        
        # Check for strong spam indicators
        if features['spam_score'] >= 3:
            spam_indicators += 2
            reasons.append(f"Multiple spam keywords (score: {features['spam_score']})")
        
        if features['exclamation_count'] >= 3:
            spam_indicators += 1
            reasons.append(f"Excessive exclamation marks ({features['exclamation_count']})")
        
        if features['caps_ratio'] > 0.3:
            spam_indicators += 1
            reasons.append(f"Too many capital letters ({features['caps_ratio']:.1%})")
        
        if features['has_url'] and features['spam_score'] > 0:
            spam_indicators += 1
            reasons.append("Suspicious URL with spam keywords")
        
        if features['spam_legit_ratio'] > 2:
            spam_indicators += 1
            reasons.append(f"High spam-to-legitimate ratio ({features['spam_legit_ratio']:.1f})")
        
        # Check for legitimate indicators
        legit_indicators = 0
        if features['legit_score'] >= 2:
            legit_indicators += 1
            reasons.append(f"Contains legitimate keywords (score: {features['legit_score']})")
        
        # Make decision
        if spam_indicators >= 2:
            confidence = min(0.9, 0.5 + spam_indicators * 0.1)
            return "SPAM", confidence, "; ".join(reasons[:3])
        elif legit_indicators > 0 and spam_indicators == 0:
            confidence = 0.7
            return "NOT_SPAM", confidence, "; ".join(reasons[:2])
        else:
            confidence = 0.5
            return "NOT_SPAM", confidence, "No clear spam indicators found"
    
    def classify_by_keywords(self, text: str) -> Tuple[str, float, str]:
        """Simple keyword-based classification"""
        text_lower = text.lower()
        
        spam_count = 0
        spam_categories = []
        
        for category, keywords in self.spam_keywords.items():
            category_count = sum(1 for keyword in keywords if keyword in text_lower)
            if category_count > 0:
                spam_count += category_count
                spam_categories.append(f"{category}({category_count})")
        
        legit_count = sum(
            sum(1 for keyword in keywords if keyword in text_lower)
            for keywords in self.legitimate_keywords.values()
        )
        
        if spam_count >= 2:
            confidence = min(0.9, 0.6 + spam_count * 0.1)
            explanation = f"Spam keywords found: {', '.join(spam_categories)}"
            return "SPAM", confidence, explanation
        elif legit_count >= 1 and spam_count == 0:
            confidence = 0.8
            return "NOT_SPAM", confidence, f"Legitimate content indicators: {legit_count}"
        else:
            confidence = 0.6
            return "NOT_SPAM", confidence, "No significant spam indicators"
    
    def train_ml_model(self, texts: List[str], labels: List[str]) -> bool:
        """Train traditional ML model"""
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn not available for ML training")
            return False
        
        print("🔧 Training ML model...")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit features for memory efficiency
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Ignore rare terms
            max_df=0.8  # Ignore too common terms
        )
        
        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(labels)
        
        # Create ensemble of lightweight models
        models = [
            ('nb', MultinomialNB(alpha=0.1)),
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
        
        # Only add SVM if dataset is small (SVM is O(n²))
        if len(texts) < 1000:
            models.append(('svm', SVC(kernel='linear', C=1.0, probability=True, random_state=42)))
        
        self.model = VotingClassifier(models, voting='soft')
        self.model.fit(X, y)
        
        print("✅ ML model trained successfully")
        self.is_trained = True
        return True
    
    def classify_by_ml(self, text: str) -> Tuple[str, float, str]:
        """ML-based classification"""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return "NOT_SPAM", 0.5, "ML model not available or trained"
        
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return prediction, confidence, f"ML prediction with {confidence:.1%} confidence"
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Main classification method"""
        if not text or pd.isna(text) or str(text).strip() == "":
            return {
                "text": text,
                "classification": "UNKNOWN",
                "confidence": 0.0,
                "explanation": "Empty or invalid text",
                "method": "validation"
            }
        
        if self.approach == "rules":
            classification, confidence, explanation = self.classify_by_rules(text)
            method = "rule-based"
        elif self.approach == "keywords":
            classification, confidence, explanation = self.classify_by_keywords(text)
            method = "keyword-based"
        elif self.approach == "ml" and self.is_trained:
            classification, confidence, explanation = self.classify_by_ml(text)
            method = "machine-learning"
        elif self.approach == "ensemble":
            # Combine multiple approaches
            results = []
            
            # Rule-based
            rule_class, rule_conf, rule_exp = self.classify_by_rules(text)
            results.append(("rules", rule_class, rule_conf, rule_exp))
            
            # Keyword-based
            kw_class, kw_conf, kw_exp = self.classify_by_keywords(text)
            results.append(("keywords", kw_class, kw_conf, kw_exp))
            
            # ML-based (if available)
            if self.is_trained and SKLEARN_AVAILABLE:
                ml_class, ml_conf, ml_exp = self.classify_by_ml(text)
                results.append(("ml", ml_class, ml_conf, ml_exp))
            
            # Ensemble decision
            spam_votes = sum(1 for _, cls, conf, _ in results if cls == "SPAM" and conf > 0.6)
            total_votes = len(results)
            
            if spam_votes >= total_votes // 2 + 1:  # Majority vote
                classification = "SPAM"
                confidence = sum(conf for _, cls, conf, _ in results if cls == "SPAM") / max(spam_votes, 1)
            else:
                classification = "NOT_SPAM"
                not_spam_votes = sum(1 for _, cls, conf, _ in results if cls == "NOT_SPAM")
                confidence = sum(conf for _, cls, conf, _ in results if cls == "NOT_SPAM") / max(not_spam_votes, 1)
            
            explanation = f"Ensemble decision ({spam_votes}/{total_votes} voted SPAM)"
            method = "ensemble"
        else:
            # Fallback to rules
            classification, confidence, explanation = self.classify_by_rules(text)
            method = "rule-based (fallback)"
        
        return {
            "text": text,
            "classification": classification,
            "confidence": round(confidence, 3),
            "explanation": explanation,
            "method": method
        }
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if self.is_trained and SKLEARN_AVAILABLE:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'approach': self.approach
                }, f)
            print(f"✅ Model saved to {filepath}")
        else:
            print("❌ No trained model to save")
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                self.approach = data['approach']
                self.is_trained = True
            print(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def create_training_data() -> Tuple[List[str], List[str]]:
    """Create expanded training data for better accuracy"""
    spam_texts = [
        "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
        "URGENT: Your account will be suspended! Verify your identity immediately.",
        "Buy cheap medications online! No prescription needed! Discount 90%!!!",
        "Make money fast! Work from home! Earn $5000 per week! No experience required!",
        "FREE iPhone 15! Limited time offer! Click now before it's too late!",
        "Lose 30 pounds in 30 days! Guaranteed results or your money back!",
        "WINNER! You have been selected to receive a $500 gift card. Claim now!",
        "Get rich quick! Secret method revealed! Only $19.99 for limited time!",
        "Click here to increase your income by 500% in just 30 days!",
        "URGENT RESPONSE NEEDED: Confirm your bank details to avoid account closure",
        "Amazing opportunity! Make $3000 weekly from home! No skills needed!",
        "Your computer is infected! Download our antivirus NOW!",
        "Congratulations! You are our 1 millionth visitor! Claim your prize!",
        "Hot singles in your area want to meet you! Click here now!",
        "Enlarge your... muscles with our new supplement! 90% discount!",
    ]
    
    legitimate_texts = [
        "Hi John, can we schedule a meeting for tomorrow at 2 PM to discuss the project?",
        "Thank you for your purchase. Your order #12345 will be delivered within 3-5 business days.",
        "Reminder: Your appointment with Dr. Smith is scheduled for Friday at 10:30 AM.",
        "The quarterly report is ready for review. Please find it attached to this email.",
        "Could you please review the attached document and provide feedback by Friday?",
        "Happy birthday! Hope you have a wonderful day celebrating with family and friends.",
        "The meeting has been rescheduled to Thursday at 3 PM. Please confirm your attendance.",
        "Please find the requested information attached. Let me know if you need anything else.",
        "Thank you for your email. I will respond to your inquiry within 24 hours.",
        "The project deadline has been extended to next Monday. Please adjust your schedule accordingly.",
        "Your subscription renewal is due next month. Please update your payment information.",
        "Regarding our discussion yesterday, I have prepared the following proposal.",
        "The weather forecast shows rain tomorrow. Don't forget your umbrella!",
        "Can you pick up milk on your way home? We're running low.",
        "The conference call is scheduled for 3 PM. Dial-in details are attached.",
    ]
    
    texts = spam_texts + legitimate_texts
    labels = ["SPAM"] * len(spam_texts) + ["NOT_SPAM"] * len(legitimate_texts)
    
    return texts, labels

def evaluate_classifier(classifier: LightweightSpamClassifier, test_file: str = "example_data.csv"):
    """Evaluate classifier performance"""
    if not Path(test_file).exists():
        print(f"❌ Test file {test_file} not found. Using built-in test data.")
        texts, labels = create_training_data()
        # Use a subset for testing
        test_texts = texts[::2]  # Every other sample
        test_labels = labels[::2]
    else:
        df = pd.read_csv(test_file)
        test_texts = df['text'].tolist()
        test_labels = df['label'].tolist()
    
    print(f"\n🔍 Evaluating classifier on {len(test_texts)} samples...")
    
    correct = 0
    results = []
    
    for text, true_label in zip(test_texts, test_labels):
        result = classifier.classify(text)
        predicted = result['classification']
        
        is_correct = true_label == predicted
        if is_correct:
            correct += 1
        
        results.append({
            'text': text[:50] + "..." if len(text) > 50 else text,
            'true': true_label,
            'predicted': predicted,
            'confidence': result['confidence'],
            'correct': is_correct,
            'method': result['method']
        })
    
    accuracy = correct / len(test_texts)
    
    print(f"\n📊 Results:")
    print(f"   Accuracy: {correct}/{len(test_texts)} = {accuracy:.1%}")
    print(f"   Method: {classifier.approach}")
    
    # Show some examples
    print(f"\n📝 Sample predictions:")
    for i, result in enumerate(results[:5]):
        status = "✅" if result['correct'] else "❌"
        print(f"   {status} {result['text']}")
        print(f"      True: {result['true']} | Predicted: {result['predicted']} | Confidence: {result['confidence']:.1%}")
    
    return accuracy, results

def main():
    """Main function demonstrating the lightweight spam classifier"""
    print("🚀 Lightweight Spam Classifier for Limited Resources")
    print("=" * 60)
    
    # Test different approaches
    approaches = ["rules", "keywords", "ensemble"]
    if SKLEARN_AVAILABLE:
        approaches.insert(-1, "ml")
    
    best_accuracy = 0
    best_approach = None
    
    for approach in approaches:
        print(f"\n🧪 Testing {approach} approach...")
        
        classifier = LightweightSpamClassifier(approach=approach)
        
        # Train ML model if needed
        if approach in ["ml", "ensemble"] and SKLEARN_AVAILABLE:
            texts, labels = create_training_data()
            classifier.train_ml_model(texts, labels)
        
        # Evaluate
        accuracy, _ = evaluate_classifier(classifier)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_approach = approach
    
    print(f"\n🏆 Best approach: {best_approach} with {best_accuracy:.1%} accuracy")
    
    # Interactive mode with best classifier
    print(f"\n💬 Interactive mode using {best_approach} approach (type 'quit' to exit):")
    
    best_classifier = LightweightSpamClassifier(approach=best_approach)
    if best_approach in ["ml", "ensemble"] and SKLEARN_AVAILABLE:
        texts, labels = create_training_data()
        best_classifier.train_ml_model(texts, labels)
    
    while True:
        try:
            user_input = input("\nEnter text to classify: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                result = best_classifier.classify(user_input)
                print(f"\n📋 Classification: {result['classification']}")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Method: {result['method']}")
                print(f"   Explanation: {result['explanation']}")
        
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break

if __name__ == "__main__":
    main()