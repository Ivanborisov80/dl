#!/usr/bin/env python3
"""
Lightweight Batch Spam Classification Script
============================================

Process CSV files using the lightweight spam classifier.
Works efficiently on limited resources without requiring large models.

Usage:
    python lightweight_batch_classify.py input.csv output.csv
    python lightweight_batch_classify.py input.csv output.csv --approach rules
    python lightweight_batch_classify.py input.csv output.csv --approach ensemble
"""

import sys
import pandas as pd
import argparse
import time
from pathlib import Path
from lightweight_spam_classifier import LightweightSpamClassifier, create_training_data

def main():
    parser = argparse.ArgumentParser(description='Batch classify text data as spam or not spam (lightweight)')
    parser.add_argument('input_file', help='Input CSV file with text data')
    parser.add_argument('output_file', help='Output CSV file for results')
    parser.add_argument('--text-column', default='text', help='Name of the text column (default: text)')
    parser.add_argument('--approach', default='ensemble', 
                       choices=['rules', 'keywords', 'ml', 'ensemble'],
                       help='Classification approach (default: ensemble)')
    parser.add_argument('--batch-size', type=int, default=100, 
                       help='Number of texts to process at once (default: 100)')
    parser.add_argument('--train-ml', action='store_true',
                       help='Train ML model (only needed for ml/ensemble approaches)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"❌ Input file '{args.input_file}' not found!")
        sys.exit(1)
    
    print(f"📁 Loading data from '{args.input_file}'...")
    try:
        df = pd.read_csv(args.input_file)
        print(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        sys.exit(1)
    
    # Check if text column exists
    if args.text_column not in df.columns:
        print(f"❌ Column '{args.text_column}' not found in CSV!")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"🤖 Initializing {args.approach} classifier...")
    classifier = LightweightSpamClassifier(approach=args.approach)
    
    # Train ML model if needed
    if (args.approach in ["ml", "ensemble"] or args.train_ml):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            print("🔧 Training ML model...")
            texts, labels = create_training_data()
            classifier.train_ml_model(texts, labels)
        except ImportError:
            print("⚠️ scikit-learn not available. Using rule-based approach instead.")
            classifier.approach = "rules"
    
    # Process data
    results = []
    texts = df[args.text_column].tolist()
    
    print(f"🔍 Classifying {len(texts)} texts using {classifier.approach} approach...")
    start_time = time.time()
    
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        batch_start = i + 1
        batch_end = min(i + args.batch_size, len(texts))
        
        print(f"   Processing batch {batch_start}-{batch_end}/{len(texts)}...")
        
        for text in batch:
            if pd.isna(text) or str(text).strip() == "":
                result = {
                    'classification': 'UNKNOWN',
                    'explanation': 'Empty or invalid text',
                    'confidence': 0.0,
                    'method': 'validation'
                }
            else:
                result = classifier.classify(str(text))
            
            results.append({
                'original_text': text,
                'classification': result['classification'],
                'explanation': result['explanation'],
                'confidence': result['confidence'],
                'method': result['method']
            })
    
    processing_time = time.time() - start_time
    texts_per_second = len(texts) / processing_time
    
    print(f"⚡ Processing completed in {processing_time:.1f}s ({texts_per_second:.1f} texts/sec)")
    
    # Create output DataFrame
    output_df = df.copy()
    output_df['predicted_classification'] = [r['classification'] for r in results]
    output_df['explanation'] = [r['explanation'] for r in results]
    output_df['confidence_score'] = [r['confidence'] for r in results]
    output_df['method'] = [r['method'] for r in results]
    
    # Add accuracy if labels are available
    if 'label' in df.columns:
        correct = sum(1 for i, row in output_df.iterrows() 
                     if row['label'] == row['predicted_classification'])
        accuracy = correct / len(output_df)
        print(f"\n📊 Accuracy: {correct}/{len(output_df)} = {accuracy:.1%}")
        
        # Add accuracy column
        output_df['correct'] = output_df['label'] == output_df['predicted_classification']
        
        # Show confusion matrix
        spam_actual = sum(1 for label in df['label'] if label == 'SPAM')
        spam_predicted = sum(1 for pred in output_df['predicted_classification'] if pred == 'SPAM')
        
        true_positives = sum(1 for i, row in output_df.iterrows() 
                           if row['label'] == 'SPAM' and row['predicted_classification'] == 'SPAM')
        false_positives = sum(1 for i, row in output_df.iterrows() 
                            if row['label'] == 'NOT_SPAM' and row['predicted_classification'] == 'SPAM')
        false_negatives = sum(1 for i, row in output_df.iterrows() 
                            if row['label'] == 'SPAM' and row['predicted_classification'] == 'NOT_SPAM')
        
        precision = true_positives / max(spam_predicted, 1)
        recall = true_positives / max(spam_actual, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
        
        print(f"   Precision: {precision:.1%}")
        print(f"   Recall: {recall:.1%}")
        print(f"   F1-Score: {f1_score:.1%}")
    
    # Save results
    print(f"💾 Saving results to '{args.output_file}'...")
    try:
        output_df.to_csv(args.output_file, index=False)
        print(f"✅ Results saved successfully!")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        sys.exit(1)
    
    # Print summary
    spam_count = sum(1 for r in results if r['classification'] == 'SPAM')
    not_spam_count = sum(1 for r in results if r['classification'] == 'NOT_SPAM')
    unknown_count = sum(1 for r in results if r['classification'] == 'UNKNOWN')
    
    print(f"\n📈 Classification Summary:")
    print(f"   SPAM: {spam_count} ({spam_count/len(results):.1%})")
    print(f"   NOT_SPAM: {not_spam_count} ({not_spam_count/len(results):.1%})")
    print(f"   UNKNOWN: {unknown_count} ({unknown_count/len(results):.1%})")
    print(f"   TOTAL: {len(results)}")
    print(f"   Method: {args.approach}")
    print(f"   Speed: {texts_per_second:.1f} texts/second")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python lightweight_batch_classify.py input.csv output.csv")
        print("Example: python lightweight_batch_classify.py example_data.csv results.csv")
        print("Example: python lightweight_batch_classify.py example_data.csv results.csv --approach rules")
        sys.exit(1)
    
    main()