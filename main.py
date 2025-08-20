#!/usr/bin/env python3
"""
Main command-line interface for SGSimEval evaluation system.
Combines functionality from evaluate.py and cal_similarity.py.
"""

import argparse
import json
import os
import sys
from typing import List, Optional

# Import functions from evaluate.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from evaluate import (
    evaluate, batch_evaluate_by_cat, batch_evaluate_by_system,
    calculate_average_score, calculate_average_score_by_cat, calculate_average_score_by_system,
    clear_scores, clear_all_scores, delete_system,
    aggregate_results_to_csv, calculate_category_average_from_csv,
    aggregate_all_categories_average, calculate_all_scores,
    reorganize_results_columns, convert_to_latex,
    compare_with_pdfs, aggregate_comparison_results,
    supplement_missing_scores
)

# Import functions from cal_similarity.py
from cal_similarity import (
    cal_outline_embedding, cal_content_embedding, cal_reference_embedding,
    cal_similarity, calculate_and_update_similarity_scores_new, batch_calculate_similarity_scores,
    clear_similarity, clear_all_similarity,
    generate_evaluation_tex, generate_evaluation_csv
)

def main() -> None:
    """
    Main entry point for SGSimEval command-line interface.
    
    Parses command-line arguments and routes to appropriate handler functions
    for evaluation, similarity calculation, aggregation, and other operations.
    """
    parser = argparse.ArgumentParser(
        description="SGSimEval - Survey Generation Similarity Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single markdown file
  python main.py evaluate --md-path "surveys/cs/topic/system/file.md" --model "qwen-plus-latest"

  # Batch evaluate by category
  python main.py batch-evaluate --mode category --categories cs econ --model "qwen-plus-latest" --workers 4

  # Batch evaluate by system
  python main.py batch-evaluate --mode system --systems AutoSurvey SurveyForge --model "qwen-plus-latest" --workers 4

  # Calculate similarity scores (both regular and human-perfect)
  python main.py similarity --systems AutoSurvey SurveyForge --model "qwen-plus-latest"

  # Calculate average scores
  python main.py average --mode category --categories cs econ

  # Aggregate results
  python main.py aggregate --mode category --categories cs econ

  # Clear scores
  python main.py clear --mode scores --categories cs --systems AutoSurvey --model "qwen-plus-latest"

  # Generate evaluation tables
  python main.py generate --mode evaluation --systems AutoSurvey SurveyForge --model "qwen-plus-latest"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a single markdown file')
    eval_parser.add_argument('--md-path', required=True, help='Path to markdown file')
    eval_parser.add_argument('--model', default='qwen-plus-latest', help='Model name for evaluation')
    eval_parser.add_argument('--outline', action='store_true', default=True, help='Evaluate outline')
    eval_parser.add_argument('--content', action='store_true', default=True, help='Evaluate content')
    eval_parser.add_argument('--reference', action='store_true', default=True, help='Evaluate references')
    
    # Batch evaluate command
    batch_parser = subparsers.add_parser('batch-evaluate', help='Batch evaluate multiple files')
    batch_parser.add_argument('--mode', choices=['category', 'system'], required=True, help='Evaluation mode')
    batch_parser.add_argument('--categories', nargs='+', help='Categories to evaluate (for category mode)')
    batch_parser.add_argument('--systems', nargs='+', help='Systems to evaluate (for system mode)')
    batch_parser.add_argument('--model', default='qwen-plus-latest', help='Model name for evaluation')
    batch_parser.add_argument('--workers', type=int, default=1, help='Number of worker threads')
    batch_parser.add_argument('--outline', action='store_true', default=True, help='Evaluate outline')
    batch_parser.add_argument('--content', action='store_true', default=True, help='Evaluate content')
    batch_parser.add_argument('--reference', action='store_true', default=True, help='Evaluate references')
    
    # Similarity command
    sim_parser = subparsers.add_parser('similarity', help='Calculate similarity scores')
    sim_parser.add_argument('--systems', nargs='+', required=True, help='Systems to process')
    sim_parser.add_argument('--model', default='qwen-plus-latest', help='Model name for evaluation')
    sim_parser.add_argument('--domains', nargs='+', default=['cs', 'econ', 'eess', 'math', 'physics', 'q-bio', 'q-fin', 'stat'], 
                           help='Domains to process')
    sim_parser.add_argument('--mode', choices=['single', 'batch'], default='batch', help='Processing mode')
    sim_parser.add_argument('--md-path', help='Path to markdown file (for single mode)')
    sim_parser.add_argument('--gt-path', help='Path to ground truth file (for single mode)')
    sim_parser.add_argument('--type', choices=['new'], default='new', help='Similarity calculation type (new function calculates both regular and human-perfect scores)')
    
    # Average command
    avg_parser = subparsers.add_parser('average', help='Calculate average scores')
    avg_parser.add_argument('--mode', choices=['category', 'system', 'all'], required=True, help='Average calculation mode')
    avg_parser.add_argument('--categories', nargs='+', help='Categories to process (for category mode)')
    avg_parser.add_argument('--systems', nargs='+', help='Systems to process (for system mode)')
    avg_parser.add_argument('--model', help='Model name (for system mode)')
    
    # Aggregate command
    agg_parser = subparsers.add_parser('aggregate', help='Aggregate results')
    agg_parser.add_argument('--mode', choices=['category', 'all'], required=True, help='Aggregation mode')
    agg_parser.add_argument('--categories', nargs='+', help='Categories to aggregate (for category mode)')
    agg_parser.add_argument('--systems', nargs='+', help='Systems to process')
    agg_parser.add_argument('--models', nargs='+', help='Models to process')
    agg_parser.add_argument('--workers', type=int, default=1, help='Number of worker threads')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear scores and data')
    clear_parser.add_argument('--mode', choices=['scores', 'similarity', 'all'], required=True, help='Clear mode')
    clear_parser.add_argument('--categories', nargs='+', help='Categories to clear')
    clear_parser.add_argument('--systems', nargs='+', help='Systems to clear')
    clear_parser.add_argument('--model', help='Model name')
    clear_parser.add_argument('--target', help='Target metrics to clear (comma-separated)')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate evaluation tables and reports')
    gen_parser.add_argument('--mode', choices=['evaluation', 'csv', 'tex'], required=True, help='Generation mode')
    gen_parser.add_argument('--systems', nargs='+', required=True, help='Systems to include')
    gen_parser.add_argument('--model', default='qwen-plus-latest', help='Model name')
    gen_parser.add_argument('--output', help='Output file path')
    
    # Supplement command
    supp_parser = subparsers.add_parser('supplement', help='Supplement missing scores')
    supp_parser.add_argument('--categories', nargs='+', help='Categories to process')
    supp_parser.add_argument('--systems', nargs='+', help='Systems to process')
    supp_parser.add_argument('--models', nargs='+', help='Models to process')
    supp_parser.add_argument('--workers', type=int, default=1, help='Number of worker threads')
    
    # Calculate all scores command
    calc_parser = subparsers.add_parser('calculate-all', help='Calculate all scores comprehensively')
    calc_parser.add_argument('--categories', nargs='+', help='Categories to process')
    calc_parser.add_argument('--systems', nargs='+', help='Systems to process')
    calc_parser.add_argument('--models', nargs='+', help='Models to process')
    calc_parser.add_argument('--workers', type=int, default=1, help='Number of worker threads')
    
    # Compare command
    comp_parser = subparsers.add_parser('compare', help='Compare systems with PDFs')
    comp_parser.add_argument('--systems', nargs='+', required=True, help='Systems to compare')
    comp_parser.add_argument('--metrics', nargs='+', default=['outline', 'content', 'reference'], 
                           help='Metrics to compare')
    comp_parser.add_argument('--model', default='qwen-plus-latest', help='Model name')
    comp_parser.add_argument('--mode', choices=['single', 'aggregate'], default='aggregate', help='Comparison mode')
    comp_parser.add_argument('--topic-dir', help='Topic directory (for single mode)')
    comp_parser.add_argument('--system', help='System name (for single mode)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'evaluate':
            handle_evaluate(args)
        elif args.command == 'batch-evaluate':
            handle_batch_evaluate(args)
        elif args.command == 'similarity':
            handle_similarity(args)
        elif args.command == 'average':
            handle_average(args)
        elif args.command == 'aggregate':
            handle_aggregate(args)
        elif args.command == 'clear':
            handle_clear(args)
        elif args.command == 'generate':
            handle_generate(args)
        elif args.command == 'supplement':
            handle_supplement(args)
        elif args.command == 'calculate-all':
            handle_calculate_all(args)
        elif args.command == 'compare':
            handle_compare(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def handle_evaluate(args) -> None:
    """Handle single file evaluation"""
    print(f"Evaluating: {args.md_path}")
    print(f"Model: {args.model}")
    
    results = evaluate(
        md_path=args.md_path,
        model=args.model,
        do_outline=args.outline,
        do_content=args.content,
        do_reference=args.reference
    )
    
    print("Evaluation completed!")
    print("Results:", results)

def handle_batch_evaluate(args) -> None:
    """Handle batch evaluation"""
    if args.mode == 'category':
        if not args.categories:
            print("Error: Categories must be specified for category mode")
            return
        
        print(f"Batch evaluating categories: {args.categories}")
        print(f"Model: {args.model}")
        print(f"Workers: {args.workers}")
        
        batch_evaluate_by_cat(
            cats=args.categories,
            model=args.model,
            do_outline=args.outline,
            do_content=args.content,
            do_reference=args.reference,
            num_workers=args.workers
        )
    
    elif args.mode == 'system':
        if not args.systems:
            print("Error: Systems must be specified for system mode")
            return
        
        print(f"Batch evaluating systems: {args.systems}")
        print(f"Model: {args.model}")
        print(f"Workers: {args.workers}")
        
        batch_evaluate_by_system(
            system_list=args.systems,
            model=args.model,
            do_outline=args.outline,
            do_content=args.content,
            do_reference=args.reference,
            num_workers=args.workers
        )

def handle_similarity(args) -> None:
    """Handle similarity calculation"""
    if args.mode == 'single':
        if not args.md_path or not args.gt_path:
            print("Error: Both --md-path and --gt-path must be specified for single mode")
            return
        
        print(f"Calculating similarity for: {args.md_path}")
        print(f"Ground truth: {args.gt_path}")
        print(f"Model: {args.model}")
        print(f"Type: {args.type}")
        
        # Always use the new function which calculates both regular and human-perfect scores
        calculate_and_update_similarity_scores_new(args.model, args.md_path, args.gt_path)
    
    elif args.mode == 'batch':
        print(f"Batch calculating similarity for systems: {args.systems}")
        print(f"Model: {args.model}")
        print(f"Domains: {args.domains}")
        
        batch_calculate_similarity_scores(
            system_list=args.systems,
            model=args.model,
            domains=args.domains
        )

def handle_average(args) -> None:
    """Handle average score calculation"""
    if args.mode == 'category':
        if not args.categories:
            print("Error: Categories must be specified for category mode")
            return
        
        print(f"Calculating averages for categories: {args.categories}")
        for cat in args.categories:
            print(f"Processing category: {cat}")
            calculate_average_score_by_cat(cat)
    
    elif args.mode == 'system':
        if not args.systems or not args.model:
            print("Error: Both --systems and --model must be specified for system mode")
            return
        
        print(f"Calculating averages for systems: {args.systems}")
        print(f"Model: {args.model}")
        for system in args.systems:
            print(f"Processing system: {system}")
            calculate_average_score_by_system(system, args.model)
    
    elif args.mode == 'all':
        print("Calculating averages for all categories")
        calculate_all_cats_average_scores()

def handle_aggregate(args) -> None:
    """Handle result aggregation"""
    if args.mode == 'category':
        if not args.categories:
            print("Error: Categories must be specified for category mode")
            return
        
        print(f"Aggregating results for categories: {args.categories}")
        for cat in args.categories:
            print(f"Processing category: {cat}")
            aggregate_results_to_csv(cat)
            calculate_category_average_from_csv(cat)
    
    elif args.mode == 'all':
        print("Aggregating all categories")
        aggregate_all_categories_average()
        
        if args.systems and args.models:
            print("Reorganizing results columns")
            reorganize_results_columns(systems=args.systems, models=args.models)
            
            print("Converting to LaTeX format")
            convert_to_latex()

def handle_clear(args) -> None:
    """Handle clearing data"""
    if args.mode == 'scores':
        if args.categories and args.systems and args.model:
            print(f"Clearing scores for categories: {args.categories}")
            print(f"Systems: {args.systems}")
            print(f"Model: {args.model}")
            
            target = None
            if args.target:
                target = args.target.split(',')
            
            for cat in args.categories:
                for system in args.systems:
                    clear_scores(cat, system, args.model, target)
        else:
            print("Clearing all scores")
            target = None
            if args.target:
                target = args.target.split(',')
            clear_all_scores(model=args.model, target=target)
    
    elif args.mode == 'similarity':
        if args.systems:
            print(f"Clearing similarity for systems: {args.systems}")
            for system in args.systems:
                clear_similarity("cs", system)  # Default to cs domain
        else:
            print("Clearing all similarity files")
            clear_all_similarity()
    
    elif args.mode == 'all':
        print("Clearing all data")
        clear_all_scores()
        clear_all_similarity()

def handle_generate(args) -> None:
    """Handle generation of evaluation tables"""
    print(f"Generating {args.mode} for systems: {args.systems}")
    print(f"Model: {args.model}")
    
    if args.mode == 'evaluation':
        output_path = args.output or "evaluation_results.tex"
        generate_evaluation_tex(args.systems, args.model, output_path)
    
    elif args.mode == 'csv':
        output_path = args.output or "evaluation_results.csv"
        generate_evaluation_csv(args.systems, args.model, output_path)
    
    elif args.mode == 'tex':
        output_path = args.output or "evaluation_results.tex"
        generate_evaluation_tex(args.systems, args.model, output_path)

def handle_supplement(args) -> None:
    """Handle supplementing missing scores"""
    print("Supplementing missing scores")
    
    if args.categories and args.systems and args.models:
        print(f"Categories: {args.categories}")
        print(f"Systems: {args.systems}")
        print(f"Models: {args.models}")
        print(f"Workers: {args.workers}")
        
        for cat in args.categories:
            for system in args.systems:
                for model in args.models:
                    print(f"Processing {cat}/{system}/{model}")
                    supplement_missing_scores(cat, model, system)
    else:
        print("Supplementing all missing scores")
        supplement_missing_scores()

def handle_calculate_all(args) -> None:
    """Handle comprehensive score calculation"""
    print("Starting comprehensive score calculation")
    
    cats = args.categories
    systems = args.systems
    models = args.models
    num_workers = args.workers
    
    print(f"Categories: {cats}")
    print(f"Systems: {systems}")
    print(f"Models: {models}")
    print(f"Workers: {num_workers}")
    
    # Use a simplified version that doesn't call the non-existent function
    try:
        calculate_all_scores(
            cats=cats,
            systems=systems,
            models=models,
            num_workers=num_workers
        )
    except Exception as e:
        print(f"Error in comprehensive score calculation: {e}")
        print("Note: Some domain-specific functions may not be available")

def handle_compare(args) -> None:
    """Handle system comparison with PDFs"""
    if args.mode == 'single':
        if not args.topic_dir or not args.system:
            print("Error: Both --topic-dir and --system must be specified for single mode")
            return
        
        print(f"Comparing {args.system} with PDFs in {args.topic_dir}")
        print(f"Metrics: {args.metrics}")
        print(f"Model: {args.model}")
        
        compare_with_pdfs(args.topic_dir, args.system, args.metrics, args.model)
    
    elif args.mode == 'aggregate':
        print(f"Aggregating comparison results for systems: {args.systems}")
        print(f"Metrics: {args.metrics}")
        print(f"Model: {args.model}")
        
        aggregate_comparison_results(
            root_dir="surveys",
            systems=args.systems,
            metrics=args.metrics,
            modelname=args.model
        )

if __name__ == "__main__":
    main()
