#!/usr/bin/env python3
"""
Batch Spam Classification Script
================================

This script processes CSV files containing text data and classifies each row
as SPAM or NOT_SPAM using the Qwen2.5 3B model.

Usage:
    python batch_classify.py input.csv output.csv
    python batch_classify.py example_data.csv results.csv
"""

import sys
import pandas as pd
import argparse
from pathlib import Path
from qwen_spam_classifier_setup import SpamClassifier

def main():
    parser = argparse.ArgumentParser(description='Batch classify text data as spam or not spam')
    parser.add_argument('input_file', help='Input CSV file with text data')
    parser.add_argument('output_file', help='Output CSV file for results')
    parser.add_argument('--text-column', default='text', help='Name of the text column (default: text)')
    parser.add_argument('--model-path', default='./models/qwen2.5-3b-instruct-q4_k_m.gguf', 
                       help='Path to the GGUF model file')
    parser.add_argument('--batch-size', type=int, default=10, 
                       help='Number of texts to process at once (default: 10)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"❌ Input file '{args.input_file}' not found!")
        sys.exit(1)
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"❌ Model file '{args.model_path}' not found!")
        print("Run 'python qwen_spam_classifier_setup.py' first to download the model.")
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
    
    print(f"🤖 Loading Qwen2.5 3B classifier...")
    classifier = SpamClassifier(args.model_path)
    
    # Process data
    results = []
    texts = df[args.text_column].tolist()
    
    print(f"🔍 Classifying {len(texts)} texts...")
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        batch_start = i + 1
        batch_end = min(i + args.batch_size, len(texts))
        
        print(f"Processing batch {batch_start}-{batch_end}/{len(texts)}...")
        
        for text in batch:
            if pd.isna(text) or text.strip() == "":
                result = {
                    'classification': 'UNKNOWN',
                    'explanation': 'Empty or invalid text',
                    'confidence_score': 0.0
                }
            else:
                result = classifier.classify(str(text))
            
            results.append({
                'original_text': text,
                'classification': result['classification'],
                'explanation': result['explanation'],
                'confidence_score': 1.0 if result['classification'] in ['SPAM', 'NOT_SPAM'] else 0.0
            })
    
    # Create output DataFrame
    output_df = df.copy()
    output_df['predicted_classification'] = [r['classification'] for r in results]
    output_df['explanation'] = [r['explanation'] for r in results]
    output_df['confidence_score'] = [r['confidence_score'] for r in results]
    
    # Add accuracy if labels are available
    if 'label' in df.columns:
        correct = sum(1 for i, row in output_df.iterrows() 
                     if row['label'] == row['predicted_classification'])
        accuracy = correct / len(output_df)
        print(f"\n📊 Accuracy: {correct}/{len(output_df)} = {accuracy:.2%}")
        
        # Add accuracy column
        output_df['correct'] = output_df['label'] == output_df['predicted_classification']
    
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
    error_count = len(results) - spam_count - not_spam_count
    
    print(f"\n📈 Classification Summary:")
    print(f"   SPAM: {spam_count}")
    print(f"   NOT_SPAM: {not_spam_count}")
    print(f"   ERRORS: {error_count}")
    print(f"   TOTAL: {len(results)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python batch_classify.py input.csv output.csv")
        print("Example: python batch_classify.py example_data.csv results.csv")
        sys.exit(1)
    
    main()