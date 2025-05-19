import argparse
import json
from pathlib import Path
from enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Enhanced Sentiment Analysis CLI')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='Input file containing texts to analyze (one per line)')
    parser.add_argument('--output', type=str, default='output', help='Output directory for results')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default='json', help='Output format')
    parser.add_argument('--preprocess', action='store_true', help='Apply text preprocessing')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on benchmark dataset')
    parser.add_argument('--dataset', type=str, default='sst2', help='Dataset for evaluation')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EnhancedSentimentAnalyzer()
    
    # Set output directory
    analyzer.output_dir = Path(args.output)
    analyzer.output_dir.mkdir(exist_ok=True)
    
    if args.evaluate:
        print(f"Evaluating model on {args.dataset} dataset...")
        results = analyzer.evaluate_model(args.dataset)
        if results:
            print("\nEvaluation Results:")
            print(json.dumps(results['metrics'], indent=2))
            print(f"\nResults saved to {analyzer.output_dir}")
        return
    
    if args.text:
        # Analyze single text
        result = analyzer.analyze_sentiment(args.text, args.preprocess)
        print("\nAnalysis Results:")
        print(json.dumps(result, indent=2))
        
        # Export results
        output_file = analyzer.export_results(result, args.format)
        print(f"\nResults exported to {output_file}")
        
    elif args.file:
        # Read texts from file
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        # Analyze texts
        results = analyzer.analyze_batch(texts, args.preprocess)
        
        # Export results
        output_file = analyzer.export_results(results, args.format)
        print(f"\nAnalyzed {len(texts)} texts")
        print(f"Results exported to {output_file}")
        
    else:
        # Interactive mode
        print("Enter text to analyze (press Ctrl+D or Ctrl+Z to finish):")
        texts = []
        try:
            while True:
                text = input("> ")
                if text.strip():
                    texts.append(text)
        except EOFError:
            pass
        
        if texts:
            # Analyze texts
            results = analyzer.analyze_batch(texts, args.preprocess)
            
            # Export results
            output_file = analyzer.export_results(results, args.format)
            print(f"\nAnalyzed {len(texts)} texts")
            print(f"Results exported to {output_file}")
        else:
            print("No texts provided")

if __name__ == "__main__":
    main() 