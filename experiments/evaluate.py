#!/usr/bin/env python3
"""
Complete LLM Reasoning Evaluation Harness

This script orchestrates the full evaluation pipeline:
1. Generate and verify solvable puzzles
2. Prompt the model in either verbose or compact mode
3. Call the LLM API
4. Parse and execute the model's output
5. Verify correctness and record metrics
6. Generate comparison reports

Usage:
    python3 evaluate.py --model gpt-3.5-turbo --puzzles hanoi --count 10 --mode verbose
    python3 evaluate.py --model gpt-3.5-turbo --puzzles river --count 10 --mode compact
    python3 evaluate.py --config experiments/config.json
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "evaluators"))

from generators.hanoi_generator import HanoiGenerator
from generators.river_generator import RiverCrossingGenerator
from evaluators.llm_client import create_llm_client, LLMClient
from evaluators.prompts import create_evaluation_prompt
from evaluators.executor import CodeExecutor
from evaluators.metrics import MetricsCollector


class EvaluationHarness:
    """Main evaluation harness that coordinates all components."""
    
    def __init__(self, 
                 model: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 mock: bool = False,
                 timeout: float = 30.0):
        """
        Initialize the evaluation harness.
        
        Args:
            model: LLM model name
            api_key: API key (optional, can use env var)
            mock: If True, use mock LLM client for testing
            timeout: Timeout for code execution
        """
        self.model = model
        self.llm_client = create_llm_client(model, api_key, mock)
        self.code_executor = CodeExecutor(timeout_seconds=timeout)
        self.metrics_collector = MetricsCollector()
        
        # Generators
        self.hanoi_generator = HanoiGenerator()
        self.river_generator = RiverCrossingGenerator()
        
        print(f"Initialized evaluation harness with model: {model}")
        if mock:
            print("Using mock LLM client for testing")
    
    def generate_puzzles(self, 
                        puzzle_type: str, 
                        count: int, 
                        difficulty_range: tuple = (3, 5),
                        seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate a set of puzzles for evaluation.
        
        Args:
            puzzle_type: "hanoi" or "river"
            count: Number of puzzles to generate
            difficulty_range: (min, max) difficulty range
            seed: Random seed for reproducibility
            
        Returns:
            List of puzzle dictionaries
        """
        print(f"Generating {count} {puzzle_type} puzzles...")
        
        if puzzle_type == "hanoi":
            if seed is not None:
                self.hanoi_generator = HanoiGenerator(seed)
            
            puzzles = self.hanoi_generator.generate_batch(
                count=count,
                min_disks=difficulty_range[0],
                max_disks=difficulty_range[1]
            )
        
        elif puzzle_type == "river":
            if seed is not None:
                self.river_generator = RiverCrossingGenerator(seed)
            
            puzzles = self.river_generator.generate_batch(
                count=count,
                variants=['classic', 'random_start', 'partial_cross', 'custom']
            )
        
        else:
            raise ValueError(f"Unknown puzzle type: {puzzle_type}")
        
        print(f"Generated {len(puzzles)} valid puzzles")
        return puzzles
    
    def evaluate_single_puzzle(self, 
                              puzzle: Dict[str, Any], 
                              mode: str = "verbose") -> Dict[str, Any]:
        """
        Evaluate a single puzzle with the LLM.
        
        Args:
            puzzle: Puzzle dictionary
            mode: "verbose" or "compact"
            
        Returns:
            Evaluation result dictionary
        """
        start_time = time.time()
        
        # Create prompt
        try:
            system_prompt, user_prompt = create_evaluation_prompt(puzzle, mode)
        except Exception as e:
            return {
                'success': False,
                'error': f"Prompt creation failed: {e}",
                'timing': {'total_time': time.time() - start_time}
            }
        
        # Get LLM response
        try:
            llm_start = time.time()
            response, token_usage = self.llm_client.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for more consistent reasoning
                max_tokens=2000   # Reasonable limit
            )
            llm_time = time.time() - llm_start
            
        except Exception as e:
            return {
                'success': False,
                'error': f"LLM API call failed: {e}",
                'timing': {'total_time': time.time() - start_time}
            }
        
        # Extract moves from response
        try:
            exec_start = time.time()
            
            if mode == "compact":
                # Execute code to get moves
                extraction_result = self.code_executor.extract_moves_from_response(response)
                success = extraction_result['success']
                moves = extraction_result['moves']
                exec_error = extraction_result.get('error', '')
            else:
                # Parse moves directly from verbose response
                moves = self._parse_verbose_moves(response)
                success = moves is not None
                exec_error = '' if success else 'Failed to parse moves from verbose response'
            
            exec_time = time.time() - exec_start
            
        except Exception as e:
            exec_time = time.time() - exec_start
            return {
                'success': False,
                'error': f"Move extraction failed: {e}",
                'timing': {
                    'response_time': llm_time,
                    'execution_time': exec_time,
                    'total_time': time.time() - start_time
                }
            }
        
        total_time = time.time() - start_time
        
        # Record result in metrics
        error_info = None
        if not success:
            error_info = ("extraction_failed", exec_error)
        
        result = self.metrics_collector.add_result(
            puzzle=puzzle,
            mode=mode,
            model=self.model,
            success=success,
            moves=moves,
            token_usage=token_usage,
            timing={
                'response_time': llm_time,
                'execution_time': exec_time,
                'total_time': total_time
            },
            error_info=error_info
        )
        
        return {
            'success': success,
            'moves': moves,
            'response': response,
            'token_usage': token_usage,
            'timing': {
                'response_time': llm_time,
                'execution_time': exec_time,
                'total_time': total_time
            },
            'result': result
        }
    
    def _parse_verbose_moves(self, response: str) -> Optional[List[tuple]]:
        """Parse moves from verbose LLM response."""
        import re
        import ast
        
        # Look for moves = [...] pattern
        patterns = [
            r'moves\s*=\s*(\[.*?\])',
            r'solution\s*=\s*(\[.*?\])',
            r'answer\s*=\s*(\[.*?\])',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # Clean up the match
                    cleaned = re.sub(r'#.*?\n', '\n', match)  # Remove comments
                    parsed = ast.literal_eval(cleaned)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return parsed
                except (ValueError, SyntaxError):
                    continue
        
        return None
    
    def run_evaluation(self, 
                      puzzle_type: str,
                      count: int,
                      modes: List[str] = ["verbose", "compact"],
                      difficulty_range: tuple = (3, 5),
                      seed: Optional[int] = None,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Run a complete evaluation experiment.
        
        Args:
            puzzle_type: "hanoi" or "river"
            count: Number of puzzles to evaluate
            modes: List of modes to test ["verbose", "compact"]
            difficulty_range: (min, max) difficulty range
            seed: Random seed for reproducibility
            save_results: Whether to save results to file
            
        Returns:
            Complete evaluation results
        """
        print(f"\n=== Starting Evaluation ===")
        print(f"Model: {self.model}")
        print(f"Puzzle Type: {puzzle_type}")
        print(f"Count: {count}")
        print(f"Modes: {modes}")
        print(f"Difficulty Range: {difficulty_range}")
        
        # Generate puzzles
        puzzles = self.generate_puzzles(puzzle_type, count, difficulty_range, seed)
        
        if not puzzles:
            print("No puzzles generated. Exiting.")
            return {'error': 'No puzzles generated'}
        
        # Run evaluation for each mode
        results = {}
        
        for mode in modes:
            print(f"\n--- Evaluating {mode} mode ---")
            mode_results = []
            
            for i, puzzle in enumerate(puzzles):
                print(f"Puzzle {i+1}/{len(puzzles)}: ", end='', flush=True)
                
                result = self.evaluate_single_puzzle(puzzle, mode)
                mode_results.append(result)
                
                if result['success']:
                    moves_count = len(result['moves']) if result['moves'] else 0
                    print(f"✓ {moves_count} moves, {result['token_usage']['total_tokens']} tokens")
                else:
                    print(f"✗ Failed")
            
            results[mode] = mode_results
        
        # Generate summary
        summary = self.generate_summary(results, puzzle_type)
        
        # Save results if requested
        if save_results:
            timestamp = int(time.time())
            filename = f"results/evaluation_{puzzle_type}_{self.model}_{timestamp}.json"
            os.makedirs("results", exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump({
                    'config': {
                        'model': self.model,
                        'puzzle_type': puzzle_type,
                        'count': count,
                        'modes': modes,
                        'difficulty_range': difficulty_range,
                        'seed': seed
                    },
                    'puzzles': puzzles,
                    'results': results,
                    'summary': summary
                }, f, indent=2)
            
            print(f"\nResults saved to: {filename}")
        
        return {
            'config': {
                'model': self.model,
                'puzzle_type': puzzle_type,
                'count': count,
                'modes': modes
            },
            'summary': summary
        }
    
    def generate_summary(self, results: Dict[str, List], puzzle_type: str) -> Dict[str, Any]:
        """Generate evaluation summary and comparison."""
        summary = {
            'overall_metrics': {},
            'mode_comparison': {},
            'detailed_metrics': {}
        }
        
        # Get metrics for each mode
        for mode in results.keys():
            metrics = self.metrics_collector.get_aggregated_metrics(
                filter_mode=mode,
                filter_model=self.model,
                filter_puzzle_type=puzzle_type
            )
            summary['detailed_metrics'][mode] = metrics
            summary['overall_metrics'][mode] = {
                'success_rate': metrics.success_rate,
                'correctness_rate': metrics.correctness_rate,
                'avg_tokens': metrics.avg_total_tokens,
                'avg_time': metrics.avg_total_time
            }
        
        # Compare modes if both verbose and compact were tested
        if 'verbose' in results and 'compact' in results:
            comparison = self.metrics_collector.compare_modes(self.model, puzzle_type)
            summary['mode_comparison'] = comparison
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary of results."""
        print("\n=== EVALUATION SUMMARY ===")
        
        for mode, metrics in summary['overall_metrics'].items():
            print(f"\n{mode.upper()} MODE:")
            print(f"  Success Rate: {metrics['success_rate']:.1%}")
            print(f"  Correctness Rate: {metrics['correctness_rate']:.1%}")
            print(f"  Avg Tokens: {metrics['avg_tokens']:.0f}")
            print(f"  Avg Time: {metrics['avg_time']:.1f}s")
        
        if 'mode_comparison' in summary and summary['mode_comparison']:
            comp = summary['mode_comparison']['improvements']
            print(f"\nCOMPACT vs VERBOSE:")
            print(f"  Token Savings: {comp['token_savings_pct']:.1f}%")
            print(f"  Time Savings: {comp['time_savings_pct']:.1f}%")
            print(f"  Accuracy Change: {comp['accuracy_change_pct']:+.1f}%")


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='LLM Reasoning Evaluation Harness')
    parser.add_argument('--model', '-m', default='gpt-3.5-turbo', help='LLM model name')
    parser.add_argument('--puzzles', '-p', choices=['hanoi', 'river'], default='hanoi', help='Puzzle type')
    parser.add_argument('--count', '-c', type=int, default=5, help='Number of puzzles')
    parser.add_argument('--mode', choices=['verbose', 'compact', 'both'], default='both', help='Evaluation mode')
    parser.add_argument('--min-difficulty', type=int, default=3, help='Minimum difficulty')
    parser.add_argument('--max-difficulty', type=int, default=5, help='Maximum difficulty')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--mock', action='store_true', help='Use mock LLM client for testing')
    parser.add_argument('--timeout', type=float, default=30.0, help='Code execution timeout')
    parser.add_argument('--config', help='Load configuration from JSON file')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results to file')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Start with the original args to preserve all parser defaults
        merged_config = vars(args).copy()
        # Override with config file values
        merged_config.update(config)
        # Override with command line args (highest priority)
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                merged_config[key] = value
        args = argparse.Namespace(**merged_config)
    
    # Set up modes
    if args.mode == 'both':
        modes = ['verbose', 'compact']
    else:
        modes = [args.mode]
    
    try:
        # Initialize harness
        harness = EvaluationHarness(
            model=args.model,
            api_key=args.api_key,
            mock=args.mock,
            timeout=args.timeout
        )
        
        # Run evaluation
        results = harness.run_evaluation(
            puzzle_type=args.puzzles,
            count=args.count,
            modes=modes,
            difficulty_range=(args.min_difficulty, args.max_difficulty),
            seed=args.seed,
            save_results=not args.no_save
        )
        
        # Print summary
        if 'summary' in results:
            harness.print_summary(results['summary'])
        
        print("\n=== Evaluation Complete ===")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 