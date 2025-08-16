#!/usr/bin/env python3
"""
Simple evaluation runner for SGSimEval.
This script provides a simple interface to run evaluations without import issues.
"""

import argparse
import json
import os
import sys

# Add the scripts directory to the path
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, scripts_dir)

# Change working directory to scripts for proper imports
os.chdir(scripts_dir)

# Import after changing directory
try:
    from evaluate import evaluate, batch_evaluate_by_cat, batch_evaluate_by_system
    from cal_similarity import calculate_and_update_similarity_scores, calculate_and_update_similarity_scores_new, batch_calculate_similarity_scores
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running this script from the project root directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='SGSimEval: Survey Generation Evaluation Framework')
    
    # Input arguments
    parser.add_argument('--survey_path', type=str,
                      help='Path to the survey markdown file to evaluate')
    parser.add_argument('--category', type=str,
                      help='Category to evaluate (e.g., cs, econ)')
    parser.add_argument('--system', type=str,
                      help='System name to evaluate')
    parser.add_argument('--model', type=str, default='gpt-4',
                      help='Model name for evaluation')
    
    # Evaluation options
    parser.add_argument('--do_outline', action='store_true', default=True,
                      help='Evaluate outline')
    parser.add_argument('--do_content', action='store_true', default=True,
                      help='Evaluate content')
    parser.add_argument('--do_reference', action='store_true', default=True,
                      help='Evaluate references')
    parser.add_argument('--criteria_type', type=str, default='general',
                      choices=['general', 'domain'],
                      help='Type of evaluation criteria')
    
    # Batch evaluation options
    parser.add_argument('--batch_category', action='store_true',
                      help='Run batch evaluation for a category')
    parser.add_argument('--batch_system', action='store_true',
                      help='Run batch evaluation for a system')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of worker threads for parallel processing')
    
    # Similarity calculation options
    parser.add_argument('--calculate_similarity', action='store_true',
                      help='Calculate similarity scores after evaluation')
    parser.add_argument('--ground_truth_path', type=str,
                      help='Path to ground truth markdown file for similarity calculation')
    parser.add_argument('--similarity_type', type=str, default='balanced',
                      choices=['balanced', 'human_perfect'],
                      help='Type of similarity weighting (balanced or human_perfect)')
    parser.add_argument('--batch_similarity', action='store_true',
                      help='Run batch similarity calculation for multiple systems')
    parser.add_argument('--systems', nargs='+', type=str,
                      help='List of systems for batch similarity calculation')
    parser.add_argument('--domains', nargs='+', type=str,
                      help='List of domains for batch similarity calculation')
    
    args = parser.parse_args()
    
    if args.survey_path:
        # Single file evaluation
        print(f"Evaluating single file: {args.survey_path}")
        results = evaluate(
            args.survey_path,
            model=args.model,
            do_outline=args.do_outline,
            do_content=args.do_content,
            do_reference=args.do_reference,
            criteria_type=args.criteria_type
        )
        print("Evaluation completed!")
        print(f"Results: {json.dumps(results, indent=2)}")
        
        # Calculate similarity if requested
        if args.calculate_similarity and args.ground_truth_path:
            print(f"Calculating similarity scores...")
            if args.similarity_type == 'human_perfect':
                similarity_scores = calculate_and_update_similarity_scores_new(
                    args.model, args.survey_path, args.ground_truth_path
                )
            else:
                similarity_scores = calculate_and_update_similarity_scores(
                    args.model, args.survey_path, args.ground_truth_path
                )
            print("Similarity calculation completed!")
            print(f"Similarity scores: {json.dumps(similarity_scores, indent=2)}")
        
    elif args.batch_category and args.category:
        # Batch evaluation by category
        print(f"Running batch evaluation for category: {args.category}")
        batch_evaluate_by_cat(
            cats=[args.category],
            model=args.model,
            do_outline=args.do_outline,
            do_content=args.do_content,
            do_reference=args.do_reference,
            num_workers=args.num_workers,
            criteria_type=args.criteria_type
        )
        print("Batch evaluation completed!")
        
    elif args.batch_system and args.system:
        # Batch evaluation by system
        print(f"Running batch evaluation for system: {args.system}")
        batch_evaluate_by_system(
            system_list=[args.system],
            model=args.model,
            do_outline=args.do_outline,
            do_content=args.do_content,
            do_reference=args.do_reference,
            num_workers=args.num_workers,
            criteria_type=args.criteria_type
        )
        print("Batch evaluation completed!")
        
    elif args.batch_similarity and args.systems and args.domains:
        # Batch similarity calculation
        print(f"Running batch similarity calculation for systems: {args.systems}")
        print(f"Domains: {args.domains}")
        batch_calculate_similarity_scores(
            args.systems, args.model, args.domains
        )
        print("Batch similarity calculation completed!")
        
    else:
        print("Please provide one of the following options:")
        print("1. --survey_path for single file evaluation")
        print("2. --batch_category with --category for category batch evaluation")
        print("3. --batch_system with --system for system batch evaluation")
        print("4. --batch_similarity with --systems and --domains for batch similarity calculation")
        print("\nFor similarity calculation, also provide:")
        print("  --calculate_similarity --ground_truth_path [path] --similarity_type [balanced|human_perfect]")
        parser.print_help()

if __name__ == '__main__':
    main()
