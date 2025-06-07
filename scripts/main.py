"""
Main entry point for SGSimEval.
Provides a command-line interface for the evaluation framework.
"""

import argparse
import json
import os
from typing import Optional

from core.evaluator import EvaluationConfig, evaluate_survey
from core.similarity import SimilarityConfig, calculate_similarity
from visualization.plotter import PlotConfig, plot_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SGSimEval: Survey Generation Evaluation Framework')
    
    # Input/output arguments
    parser.add_argument('--survey_path', type=str, required=True,
                      help='Path to the survey to evaluate')
    parser.add_argument('--reference_path', type=str,
                      help='Path to the reference survey for similarity calculation')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save evaluation results')
    
    # Evaluation configuration
    parser.add_argument('--do_outline', action='store_true', default=True,
                      help='Evaluate outline')
    parser.add_argument('--do_content', action='store_true', default=True,
                      help='Evaluate content')
    parser.add_argument('--do_reference', action='store_true', default=True,
                      help='Evaluate references')
    parser.add_argument('--criteria_type', type=str, default='general',
                      choices=['general', 'domain'],
                      help='Type of evaluation criteria to use')
    
    # Similarity configuration
    parser.add_argument('--use_human_as_perfect', action='store_true', default=True,
                      help='Use human-authored content as perfect reference')
    parser.add_argument('--use_balanced_weighting', action='store_true', default=True,
                      help='Use balanced weighting for similarity calculation')
    
    # Plotting configuration
    parser.add_argument('--plot_results', action='store_true', default=True,
                      help='Generate plots for evaluation results')
    parser.add_argument('--plot_style', type=str, default='seaborn',
                      help='Style for plots')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure evaluation
    eval_config = EvaluationConfig(
        do_outline=args.do_outline,
        do_content=args.do_content,
        do_reference=args.do_reference,
        criteria_type=args.criteria_type
    )
    
    # Evaluate survey
    results = evaluate_survey(args.survey_path, eval_config)
    
    # Calculate similarity if reference is provided
    if args.reference_path:
        sim_config = SimilarityConfig(
            use_human_as_perfect=args.use_human_as_perfect,
            use_balanced_weighting=args.use_balanced_weighting
        )
        similarity_results = calculate_similarity(
            results,
            evaluate_survey(args.reference_path, eval_config),
            sim_config
        )
        results['similarity'] = similarity_results
    
    # Save results
    output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results if requested
    if args.plot_results:
        plot_config = PlotConfig(
            save_path=args.output_dir,
            style=args.plot_style
        )
        plot_results(results, os.path.basename(args.survey_path), plot_config)

if __name__ == '__main__':
    main() 