#!/usr/bin/env python3
"""
Metrics and Evaluation Framework

Tracks accuracy, token usage, execution time, and other metrics for comparing
verbose vs compact LLM evaluation modes.
"""

import json
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.hanoi_solver import HanoiSolver
from solvers.river_solver import RiverCrossingSolver


@dataclass
class EvaluationResult:
    """Single evaluation result for one puzzle attempt."""
    puzzle_id: str
    puzzle_type: str  # "hanoi" or "river"
    mode: str  # "verbose" or "compact"
    model: str
    success: bool
    correct_moves: bool
    reaches_goal: bool
    
    # Token usage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # Timing
    llm_response_time: float
    code_execution_time: float
    total_time: float
    
    # Solution quality
    expected_moves: int
    actual_moves: int
    move_efficiency: float  # actual/expected ratio
    
    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # Additional metadata
    difficulty: str = "unknown"
    puzzle_size: int = 0  # Number of disks or entities


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple evaluations."""
    total_attempts: int
    successful_attempts: int
    success_rate: float
    
    # Token statistics
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_total_tokens: float
    total_token_cost: int
    
    # Timing statistics
    avg_response_time: float
    avg_execution_time: float
    avg_total_time: float
    
    # Accuracy statistics
    correctness_rate: float  # Percentage of solutions that reach goal
    avg_move_efficiency: float
    
    # By difficulty
    metrics_by_difficulty: Dict[str, Dict[str, float]]
    
    # By puzzle size
    metrics_by_size: Dict[int, Dict[str, float]]


class MetricsCollector:
    """Collects and analyzes evaluation metrics."""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
        self.hanoi_solver = HanoiSolver()
        self.river_solver = RiverCrossingSolver()
    
    def verify_solution(self, puzzle: Dict[str, Any], moves: List[Tuple]) -> Tuple[bool, bool]:
        """
        Verify if a solution is correct and reaches the goal.
        
        Args:
            puzzle: Original puzzle dictionary
            moves: List of move tuples from LLM
            
        Returns:
            Tuple of (moves_are_valid, reaches_goal)
        """
        try:
            # Determine puzzle type
            if 'num_disks' in puzzle.get('metadata', {}):
                return self._verify_hanoi_solution(puzzle, moves)
            elif 'entities' in puzzle.get('metadata', {}):
                return self._verify_river_solution(puzzle, moves)
            else:
                return False, False
        except Exception:
            return False, False
    
    def _verify_hanoi_solution(self, puzzle: Dict[str, Any], moves: List[Tuple]) -> Tuple[bool, bool]:
        """Verify Tower of Hanoi solution."""
        initial_state = puzzle['initial_state']
        goal_state = puzzle['goal_state']
        
        # Simulate the moves
        current_state = {
            'pegs': {peg: list(disks) for peg, disks in initial_state['pegs'].items()},
            'num_disks': initial_state['num_disks']
        }
        
        try:
            for from_peg, to_peg in moves:
                # Validate move
                if from_peg not in current_state['pegs'] or to_peg not in current_state['pegs']:
                    return False, False
                
                if not current_state['pegs'][from_peg]:
                    return False, False  # No disk to move
                
                # Get the disk to move (top disk)
                disk = current_state['pegs'][from_peg].pop()
                
                # Check if move is valid (no larger disk on smaller)
                if (current_state['pegs'][to_peg] and 
                    current_state['pegs'][to_peg][-1] < disk):
                    return False, False
                
                # Make the move
                current_state['pegs'][to_peg].append(disk)
            
            # Check if goal is reached
            reaches_goal = (current_state['pegs'] == goal_state['pegs'])
            return True, reaches_goal
            
        except (IndexError, KeyError, TypeError):
            return False, False
    
    def _verify_river_solution(self, puzzle: Dict[str, Any], moves: List[Tuple]) -> Tuple[bool, bool]:
        """Verify River Crossing solution."""
        initial_state = puzzle['initial_state']
        goal_state = puzzle['goal_state']
        entities = puzzle['metadata']['entities']
        
        # Use the river solver to verify
        try:
            solver = RiverCrossingSolver(entities)
            
            # Simulate the moves
            current_state = initial_state.copy()
            
            for direction, moving_entities in moves:
                # Validate the move format
                if direction not in ['left', 'right']:
                    return False, False
                
                if not isinstance(moving_entities, list):
                    return False, False
                
                # Check that farmer is included (required for river crossing)
                if entities[0] not in moving_entities:  # Farmer is usually first entity
                    return False, False
                
                # Apply the move
                for entity in moving_entities:
                    if entity not in current_state:
                        return False, False
                    current_state[entity] = direction
                
                # Check if the state is valid using solver's validation
                if not solver._is_valid_state(solver._dict_to_tuple(current_state)):
                    return False, False
            
            # Check if goal is reached
            reaches_goal = (current_state == goal_state)
            return True, reaches_goal
            
        except Exception:
            return False, False
    
    def get_expected_moves(self, puzzle: Dict[str, Any]) -> int:
        """Get the expected number of moves for optimal solution."""
        try:
            if 'num_disks' in puzzle.get('metadata', {}):
                # Tower of Hanoi: 2^n - 1 for simple cases
                n = puzzle['metadata']['num_disks']
                return (2 ** n) - 1
            elif 'entities' in puzzle.get('metadata', {}):
                # River crossing: try to solve and count moves
                solver = RiverCrossingSolver()
                moves = solver.solve(puzzle['initial_state'], puzzle['goal_state'])
                return len(moves)
            else:
                return 0
        except Exception:
            return 0
    
    def add_result(self, 
                   puzzle: Dict[str, Any],
                   mode: str,
                   model: str,
                   success: bool,
                   moves: Optional[List[Tuple]],
                   token_usage: Dict[str, int],
                   timing: Dict[str, float],
                   error_info: Optional[Tuple[str, str]] = None) -> EvaluationResult:
        """
        Add an evaluation result.
        
        Args:
            puzzle: Original puzzle
            mode: "verbose" or "compact"
            model: Model name
            success: Whether extraction succeeded
            moves: Extracted moves (None if failed)
            token_usage: Dict with prompt_tokens, completion_tokens, total_tokens
            timing: Dict with response_time, execution_time, total_time
            error_info: Optional (error_type, error_message) tuple
            
        Returns:
            Created EvaluationResult
        """
        # Verify solution if moves were extracted
        correct_moves = False
        reaches_goal = False
        actual_moves = 0
        
        if success and moves:
            correct_moves, reaches_goal = self.verify_solution(puzzle, moves)
            actual_moves = len(moves)
        
        # Get expected moves for efficiency calculation
        expected_moves = self.get_expected_moves(puzzle)
        move_efficiency = actual_moves / expected_moves if expected_moves > 0 else float('inf')
        
        # Determine puzzle type
        if 'num_disks' in puzzle.get('metadata', {}):
            puzzle_type = "hanoi"
            puzzle_size = puzzle['metadata']['num_disks']
        else:
            puzzle_type = "river"
            puzzle_size = len(puzzle['metadata'].get('entities', []))
        
        # Create result
        result = EvaluationResult(
            puzzle_id=puzzle.get('metadata', {}).get('puzzle_id', f"{puzzle_type}_{len(self.results)}"),
            puzzle_type=puzzle_type,
            mode=mode,
            model=model,
            success=success,
            correct_moves=correct_moves,
            reaches_goal=reaches_goal,
            prompt_tokens=token_usage.get('prompt_tokens', 0),
            completion_tokens=token_usage.get('completion_tokens', 0),
            total_tokens=token_usage.get('total_tokens', 0),
            llm_response_time=timing.get('response_time', 0),
            code_execution_time=timing.get('execution_time', 0),
            total_time=timing.get('total_time', 0),
            expected_moves=expected_moves,
            actual_moves=actual_moves,
            move_efficiency=move_efficiency,
            difficulty=puzzle.get('metadata', {}).get('difficulty', 'unknown'),
            puzzle_size=puzzle_size
        )
        
        if error_info:
            result.error_type, result.error_message = error_info
        
        self.results.append(result)
        return result
    
    def get_aggregated_metrics(self, 
                             filter_mode: Optional[str] = None,
                             filter_model: Optional[str] = None,
                             filter_puzzle_type: Optional[str] = None) -> AggregatedMetrics:
        """
        Calculate aggregated metrics with optional filtering.
        
        Args:
            filter_mode: Only include results with this mode
            filter_model: Only include results with this model
            filter_puzzle_type: Only include results with this puzzle type
            
        Returns:
            AggregatedMetrics object
        """
        # Apply filters
        filtered_results = self.results
        
        if filter_mode:
            filtered_results = [r for r in filtered_results if r.mode == filter_mode]
        
        if filter_model:
            filtered_results = [r for r in filtered_results if r.model == filter_model]
        
        if filter_puzzle_type:
            filtered_results = [r for r in filtered_results if r.puzzle_type == filter_puzzle_type]
        
        if not filtered_results:
            # Return empty metrics
            return AggregatedMetrics(
                total_attempts=0, successful_attempts=0, success_rate=0.0,
                avg_prompt_tokens=0.0, avg_completion_tokens=0.0, avg_total_tokens=0.0,
                total_token_cost=0, avg_response_time=0.0, avg_execution_time=0.0,
                avg_total_time=0.0, correctness_rate=0.0, avg_move_efficiency=0.0,
                metrics_by_difficulty={}, metrics_by_size={}
            )
        
        # Calculate basic metrics
        total_attempts = len(filtered_results)
        successful_attempts = sum(1 for r in filtered_results if r.success)
        success_rate = successful_attempts / total_attempts
        
        # Token statistics
        prompt_tokens = [r.prompt_tokens for r in filtered_results]
        completion_tokens = [r.completion_tokens for r in filtered_results]
        total_tokens = [r.total_tokens for r in filtered_results]
        
        # Timing statistics
        response_times = [r.llm_response_time for r in filtered_results]
        execution_times = [r.code_execution_time for r in filtered_results]
        total_times = [r.total_time for r in filtered_results]
        
        # Accuracy statistics
        correct_results = [r for r in filtered_results if r.success]
        correctness_rate = sum(1 for r in correct_results if r.reaches_goal) / len(correct_results) if correct_results else 0
        
        # Move efficiency (only for successful results that reach goal)
        successful_efficient = [r.move_efficiency for r in correct_results if r.reaches_goal and r.move_efficiency != float('inf')]
        avg_move_efficiency = statistics.mean(successful_efficient) if successful_efficient else 0.0
        
        # Group by difficulty
        by_difficulty = defaultdict(list)
        for result in filtered_results:
            by_difficulty[result.difficulty].append(result)
        
        metrics_by_difficulty = {}
        for difficulty, results in by_difficulty.items():
            success_count = sum(1 for r in results if r.reaches_goal)
            metrics_by_difficulty[difficulty] = {
                'count': len(results),
                'success_rate': success_count / len(results),
                'avg_tokens': statistics.mean([r.total_tokens for r in results])
            }
        
        # Group by puzzle size
        by_size = defaultdict(list)
        for result in filtered_results:
            by_size[result.puzzle_size].append(result)
        
        metrics_by_size = {}
        for size, results in by_size.items():
            success_count = sum(1 for r in results if r.reaches_goal)
            metrics_by_size[size] = {
                'count': len(results),
                'success_rate': success_count / len(results),
                'avg_tokens': statistics.mean([r.total_tokens for r in results])
            }
        
        return AggregatedMetrics(
            total_attempts=total_attempts,
            successful_attempts=successful_attempts,
            success_rate=success_rate,
            avg_prompt_tokens=statistics.mean(prompt_tokens) if prompt_tokens else 0,
            avg_completion_tokens=statistics.mean(completion_tokens) if completion_tokens else 0,
            avg_total_tokens=statistics.mean(total_tokens) if total_tokens else 0,
            total_token_cost=sum(total_tokens),
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            avg_execution_time=statistics.mean(execution_times) if execution_times else 0,
            avg_total_time=statistics.mean(total_times) if total_times else 0,
            correctness_rate=correctness_rate,
            avg_move_efficiency=avg_move_efficiency,
            metrics_by_difficulty=metrics_by_difficulty,
            metrics_by_size=metrics_by_size
        )
    
    def compare_modes(self, model: str, puzzle_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare verbose vs compact modes for a given model.
        
        Args:
            model: Model name to compare
            puzzle_type: Optional puzzle type filter
            
        Returns:
            Comparison dictionary
        """
        verbose_metrics = self.get_aggregated_metrics("verbose", model, puzzle_type)
        compact_metrics = self.get_aggregated_metrics("compact", model, puzzle_type)
        
        # Calculate improvement ratios
        token_savings = 1 - (compact_metrics.avg_total_tokens / verbose_metrics.avg_total_tokens) if verbose_metrics.avg_total_tokens > 0 else 0
        time_savings = 1 - (compact_metrics.avg_total_time / verbose_metrics.avg_total_time) if verbose_metrics.avg_total_time > 0 else 0
        accuracy_change = compact_metrics.correctness_rate - verbose_metrics.correctness_rate
        
        return {
            'model': model,
            'puzzle_type': puzzle_type or 'all',
            'verbose_metrics': asdict(verbose_metrics),
            'compact_metrics': asdict(compact_metrics),
            'improvements': {
                'token_savings_pct': token_savings * 100,
                'time_savings_pct': time_savings * 100,
                'accuracy_change_pct': accuracy_change * 100
            }
        }
    
    def export_results(self, filename: str):
        """Export all results to JSON file."""
        data = {
            'results': [asdict(result) for result in self.results],
            'summary': {
                'total_evaluations': len(self.results),
                'models': list(set(r.model for r in self.results)),
                'modes': list(set(r.mode for r in self.results)),
                'puzzle_types': list(set(r.puzzle_type for r in self.results))
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self):
        """Print a summary of all results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("=== LLM Reasoning Evaluation Summary ===")
        print(f"Total evaluations: {len(self.results)}")
        
        # Overall success rate
        successful = sum(1 for r in self.results if r.reaches_goal)
        print(f"Overall success rate: {successful}/{len(self.results)} ({successful/len(self.results)*100:.1f}%)")
        
        # By mode
        modes = set(r.mode for r in self.results)
        for mode in modes:
            mode_results = [r for r in self.results if r.mode == mode]
            mode_successful = sum(1 for r in mode_results if r.reaches_goal)
            avg_tokens = statistics.mean([r.total_tokens for r in mode_results])
            print(f"  {mode.capitalize()} mode: {mode_successful}/{len(mode_results)} ({mode_successful/len(mode_results)*100:.1f}%) success, {avg_tokens:.0f} avg tokens")
        
        # By puzzle type
        puzzle_types = set(r.puzzle_type for r in self.results)
        for ptype in puzzle_types:
            type_results = [r for r in self.results if r.puzzle_type == ptype]
            type_successful = sum(1 for r in type_results if r.reaches_goal)
            print(f"  {ptype.capitalize()}: {type_successful}/{len(type_results)} ({type_successful/len(type_results)*100:.1f}%) success")


if __name__ == "__main__":
    # Test the metrics system
    print("Testing metrics system...")
    
    collector = MetricsCollector()
    
    # Example puzzle for testing
    example_puzzle = {
        'initial_state': {
            'pegs': {'peg_0': [3, 2, 1], 'peg_1': [], 'peg_2': []},
            'num_disks': 3
        },
        'goal_state': {
            'pegs': {'peg_0': [], 'peg_1': [], 'peg_2': [3, 2, 1]},
            'num_disks': 3
        },
        'metadata': {'num_disks': 3, 'difficulty': 'easy', 'puzzle_id': 'test_1'}
    }
    
    # Example moves (correct 3-disk Tower of Hanoi solution)
    correct_moves = [("peg_0", "peg_2"), ("peg_0", "peg_1"), ("peg_2", "peg_1"), 
                    ("peg_0", "peg_2"), ("peg_1", "peg_0"), ("peg_1", "peg_2"), ("peg_0", "peg_2")]
    
    # Add a test result
    collector.add_result(
        puzzle=example_puzzle,
        mode="verbose",
        model="gpt-3.5-turbo",
        success=True,
        moves=correct_moves,
        token_usage={'prompt_tokens': 150, 'completion_tokens': 80, 'total_tokens': 230},
        timing={'response_time': 2.1, 'execution_time': 0.1, 'total_time': 2.2}
    )
    
    # Print summary
    collector.print_summary()
    
    # Get aggregated metrics
    metrics = collector.get_aggregated_metrics()
    print(f"\nDetailed metrics:")
    print(f"Success rate: {metrics.success_rate:.2f}")
    print(f"Average tokens: {metrics.avg_total_tokens:.0f}")
    print(f"Correctness rate: {metrics.correctness_rate:.2f}")
    print(f"Average move efficiency: {metrics.avg_move_efficiency:.2f}") 