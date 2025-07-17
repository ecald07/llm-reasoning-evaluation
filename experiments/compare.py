#!/usr/bin/env python3
"""
Results Comparison Script

Compares results from different evaluation runs and generates reports.

Usage:
    python3 compare.py results/eval1.json results/eval2.json
    python3 compare.py --directory results/ --model gpt-3.5-turbo
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics


def load_results(filename: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def compare_modes(results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two sets of results (e.g., verbose vs compact)."""
    comparison = {
        'config1': results1.get('config', {}),
        'config2': results2.get('config', {}),
        'metrics_comparison': {}
    }
    
    # Extract metrics from summaries
    summary1 = results1.get('summary', {}).get('overall_metrics', {})
    summary2 = results2.get('summary', {}).get('overall_metrics', {})
    
    for mode in set(list(summary1.keys()) + list(summary2.keys())):
        if mode in summary1 and mode in summary2:
            metrics1 = summary1[mode]
            metrics2 = summary2[mode]
            
            comparison['metrics_comparison'][mode] = {
                'success_rate_change': metrics2['success_rate'] - metrics1['success_rate'],
                'token_change': metrics2['avg_tokens'] - metrics1['avg_tokens'],
                'time_change': metrics2['avg_time'] - metrics1['avg_time'],
                'correctness_change': metrics2['correctness_rate'] - metrics1['correctness_rate']
            }
    
    return comparison


def print_comparison(comparison: Dict[str, Any]):
    """Print a formatted comparison report."""
    print("=== RESULTS COMPARISON ===\n")
    
    config1 = comparison['config1']
    config2 = comparison['config2']
    
    print(f"Configuration 1: {config1.get('model', 'Unknown')} on {config1.get('puzzle_type', 'Unknown')} ({config1.get('count', 0)} puzzles)")
    print(f"Configuration 2: {config2.get('model', 'Unknown')} on {config2.get('puzzle_type', 'Unknown')} ({config2.get('count', 0)} puzzles)")
    
    for mode, metrics in comparison['metrics_comparison'].items():
        print(f"\n{mode.upper()} MODE CHANGES:")
        print(f"  Success Rate: {metrics['success_rate_change']:+.1%}")
        print(f"  Correctness Rate: {metrics['correctness_change']:+.1%}")
        print(f"  Token Usage: {metrics['token_change']:+.0f}")
        print(f"  Time: {metrics['time_change']:+.1f}s")


def find_result_files(directory: str, model: Optional[str] = None, puzzle_type: Optional[str] = None) -> List[str]:
    """Find result files in directory matching criteria."""
    results_dir = Path(directory)
    files = []
    
    for file_path in results_dir.glob("evaluation_*.json"):
        if model and model not in file_path.name:
            continue
        if puzzle_type and puzzle_type not in file_path.name:
            continue
        files.append(str(file_path))
    
    return sorted(files)


def aggregate_metrics_across_files(files: List[str]) -> Dict[str, Any]:
    """Aggregate metrics across multiple result files."""
    all_results = []
    
    for file_path in files:
        results = load_results(file_path)
        all_results.append(results)
    
    # Combine metrics
    aggregated = {
        'total_files': len(files),
        'combined_metrics': {},
        'files': files
    }
    
    # Group by mode
    by_mode = {}
    for results in all_results:
        summary = results.get('summary', {}).get('overall_metrics', {})
        for mode, metrics in summary.items():
            if mode not in by_mode:
                by_mode[mode] = []
            by_mode[mode].append(metrics)
    
    # Calculate averages
    for mode, metrics_list in by_mode.items():
        if metrics_list:
            aggregated['combined_metrics'][mode] = {
                'avg_success_rate': statistics.mean([m['success_rate'] for m in metrics_list]),
                'avg_correctness_rate': statistics.mean([m['correctness_rate'] for m in metrics_list]),
                'avg_tokens': statistics.mean([m['avg_tokens'] for m in metrics_list]),
                'avg_time': statistics.mean([m['avg_time'] for m in metrics_list]),
                'std_success_rate': statistics.stdev([m['success_rate'] for m in metrics_list]) if len(metrics_list) > 1 else 0,
                'count': len(metrics_list)
            }
    
    return aggregated


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='Compare LLM evaluation results')
    parser.add_argument('files', nargs='*', help='Result files to compare')
    parser.add_argument('--directory', '-d', help='Directory to search for result files')
    parser.add_argument('--model', '-m', help='Filter by model name')
    parser.add_argument('--puzzle-type', '-p', help='Filter by puzzle type')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate metrics across files')
    
    args = parser.parse_args()
    
    # Determine files to process
    if args.directory:
        files = find_result_files(args.directory, args.model, args.puzzle_type)
        if not files:
            print(f"No result files found in {args.directory}")
            sys.exit(1)
        print(f"Found {len(files)} result files")
    elif args.files:
        files = args.files
    else:
        parser.print_help()
        sys.exit(1)
    
    if args.aggregate:
        # Aggregate metrics across multiple files
        aggregated = aggregate_metrics_across_files(files)
        
        print("=== AGGREGATED METRICS ===\n")
        print(f"Total files: {aggregated['total_files']}")
        
        for mode, metrics in aggregated['combined_metrics'].items():
            print(f"\n{mode.upper()} MODE (across {metrics['count']} runs):")
            print(f"  Success Rate: {metrics['avg_success_rate']:.1%} ± {metrics['std_success_rate']:.1%}")
            print(f"  Correctness Rate: {metrics['avg_correctness_rate']:.1%}")
            print(f"  Avg Tokens: {metrics['avg_tokens']:.0f}")
            print(f"  Avg Time: {metrics['avg_time']:.1f}s")
    
    elif len(files) == 2:
        # Compare two files
        results1 = load_results(files[0])
        results2 = load_results(files[1])
        
        comparison = compare_modes(results1, results2)
        print_comparison(comparison)
    
    else:
        # Show summary of each file
        print("=== RESULTS SUMMARY ===\n")
        
        for file_path in files:
            print(f"File: {file_path}")
            results = load_results(file_path)
            
            config = results.get('config', {})
            summary = results.get('summary', {}).get('overall_metrics', {})
            
            print(f"  Model: {config.get('model', 'Unknown')}")
            print(f"  Puzzle Type: {config.get('puzzle_type', 'Unknown')}")
            print(f"  Count: {config.get('count', 0)}")
            
            for mode, metrics in summary.items():
                print(f"  {mode.capitalize()}: {metrics['success_rate']:.1%} success, {metrics['avg_tokens']:.0f} tokens")
            print()


if __name__ == "__main__":
    main() 