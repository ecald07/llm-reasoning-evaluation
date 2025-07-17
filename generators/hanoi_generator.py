#!/usr/bin/env python3
"""
Tower of Hanoi Puzzle Generator

Generates randomized Tower of Hanoi puzzle instances and verifies solvability.
"""

import sys
import json
import random
import argparse
from typing import Dict, List, Any, Optional
from solvers.hanoi_solver import HanoiSolver


class HanoiGenerator:
    """Generator for Tower of Hanoi puzzles."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.solver = HanoiSolver()
    
    def generate_puzzle(self, num_disks: int = 3, num_pegs: int = 3) -> Dict[str, Any]:
        """
        Generate a random Tower of Hanoi puzzle.
        
        Args:
            num_disks: Number of disks in the puzzle (default: 3)
            num_pegs: Number of pegs in the puzzle (default: 3)
        
        Returns:
            Dict containing the puzzle with initial_state and goal_state
        """
        if num_disks < 1:
            raise ValueError("Number of disks must be at least 1")
        if num_pegs < 3:
            raise ValueError("Number of pegs must be at least 3")
        
        # Create peg names
        peg_names = [f"peg_{i}" for i in range(num_pegs)]
        
        # Create initial state - all disks on a random peg
        initial_pegs = {peg: [] for peg in peg_names}
        start_peg = random.choice(peg_names)
        initial_pegs[start_peg] = list(range(num_disks, 0, -1))  # Largest disk at bottom
        
        # Create goal state - all disks on a different random peg
        goal_pegs = {peg: [] for peg in peg_names}
        available_pegs = [p for p in peg_names if p != start_peg]
        end_peg = random.choice(available_pegs)
        goal_pegs[end_peg] = list(range(num_disks, 0, -1))  # Largest disk at bottom
        
        initial_state = {
            'pegs': initial_pegs,
            'num_disks': num_disks
        }
        
        goal_state = {
            'pegs': goal_pegs,
            'num_disks': num_disks
        }
        
        puzzle = {
            'initial_state': initial_state,
            'goal_state': goal_state,
            'metadata': {
                'num_disks': num_disks,
                'num_pegs': num_pegs,
                'start_peg': start_peg,
                'end_peg': end_peg,
                'difficulty': self._estimate_difficulty(num_disks)
            }
        }
        
        return puzzle
    
    def generate_complex_puzzle(self, num_disks: int = 4, num_pegs: int = 4) -> Dict[str, Any]:
        """
        Generate a more complex Tower of Hanoi puzzle with partially distributed disks.
        
        Args:
            num_disks: Number of disks in the puzzle
            num_pegs: Number of pegs in the puzzle
        
        Returns:
            Dict containing the puzzle with initial_state and goal_state
        """
        if num_disks < 2:
            raise ValueError("Number of disks must be at least 2 for complex puzzles")
        if num_pegs < 3:
            raise ValueError("Number of pegs must be at least 3")
        
        peg_names = [f"peg_{i}" for i in range(num_pegs)]
        
        # Create a valid initial state with disks distributed across pegs
        initial_pegs = {peg: [] for peg in peg_names}
        
        # Distribute disks randomly while maintaining valid stacking
        disks = list(range(1, num_disks + 1))
        random.shuffle(disks)
        
        for disk in disks:
            # Find valid pegs where this disk can be placed
            valid_pegs = []
            for peg_name in peg_names:
                peg_stack = initial_pegs[peg_name]
                if not peg_stack or peg_stack[-1] > disk:  # Can place if peg empty or top disk is larger
                    valid_pegs.append(peg_name)
            
            if valid_pegs:
                chosen_peg = random.choice(valid_pegs)
                initial_pegs[chosen_peg].append(disk)
            else:
                # Fallback: place on first available peg (shouldn't happen with proper generation)
                initial_pegs[peg_names[0]].append(disk)
        
        # Create goal state - all disks on one peg
        goal_pegs = {peg: [] for peg in peg_names}
        end_peg = random.choice(peg_names)
        goal_pegs[end_peg] = list(range(num_disks, 0, -1))
        
        initial_state = {
            'pegs': initial_pegs,
            'num_disks': num_disks
        }
        
        goal_state = {
            'pegs': goal_pegs,
            'num_disks': num_disks
        }
        
        puzzle = {
            'initial_state': initial_state,
            'goal_state': goal_state,
            'metadata': {
                'num_disks': num_disks,
                'num_pegs': num_pegs,
                'puzzle_type': 'complex',
                'end_peg': end_peg,
                'difficulty': self._estimate_difficulty(num_disks, complex_puzzle=True)
            }
        }
        
        return puzzle
    
    def verify_solvability(self, puzzle: Dict[str, Any]) -> bool:
        """
        Verify that a puzzle is solvable using the hanoi_solver.
        
        Args:
            puzzle: The puzzle to verify
        
        Returns:
            True if solvable, False otherwise
        """
        try:
            moves = self.solver.solve(puzzle['initial_state'], puzzle['goal_state'])
            return len(moves) > 0 or self._states_equal(puzzle['initial_state'], puzzle['goal_state'])
        except (ValueError, KeyError) as e:
            # These are expected errors for unsolvable puzzles
            return False
    
    def _states_equal(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> bool:
        """Check if two states are equal."""
        return (state1['num_disks'] == state2['num_disks'] and 
                state1['pegs'] == state2['pegs'])
    
    def _estimate_difficulty(self, num_disks: int, complex_puzzle: bool = False) -> str:
        """Estimate puzzle difficulty based on number of disks."""
        base_difficulty = num_disks
        if complex_puzzle:
            base_difficulty += 1
        
        if base_difficulty <= 3:
            return "easy"
        elif base_difficulty <= 5:
            return "medium"
        elif base_difficulty <= 7:
            return "hard"
        else:
            return "expert"
    
    def generate_batch(self, count: int, min_disks: int = 3, max_disks: int = 5, 
                      complex_prob: float = 0.3) -> List[Dict[str, Any]]:
        """
        Generate a batch of puzzles with varying difficulty.
        
        Args:
            count: Number of puzzles to generate
            min_disks: Minimum number of disks
            max_disks: Maximum number of disks
            complex_prob: Probability of generating complex puzzles
        
        Returns:
            List of verified solvable puzzles
        """
        puzzles = []
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loops
        
        while len(puzzles) < count and attempts < max_attempts:
            attempts += 1
            
            num_disks = random.randint(min_disks, max_disks)
            num_pegs = random.randint(3, 4)  # Keep it reasonable
            
            try:
                if random.random() < complex_prob and num_disks >= 2:
                    puzzle = self.generate_complex_puzzle(num_disks, num_pegs)
                else:
                    puzzle = self.generate_puzzle(num_disks, num_pegs)
                
                if self.verify_solvability(puzzle):
                    puzzle['metadata']['puzzle_id'] = len(puzzles) + 1
                    puzzles.append(puzzle)
            
            except (ValueError, KeyError, TypeError) as e:
                # Skip puzzles that can't be generated or verified
                continue
        
        return puzzles


def main():
    """Command-line interface for the Tower of Hanoi generator."""
    parser = argparse.ArgumentParser(description='Generate Tower of Hanoi puzzles')
    parser.add_argument('--count', '-c', type=int, default=1, help='Number of puzzles to generate')
    parser.add_argument('--disks', '-d', type=int, default=3, help='Number of disks (or max for batch)')
    parser.add_argument('--min-disks', type=int, default=3, help='Minimum disks for batch generation')
    parser.add_argument('--pegs', '-p', type=int, default=3, help='Number of pegs')
    parser.add_argument('--complex', action='store_true', help='Generate complex puzzles')
    parser.add_argument('--complex-prob', type=float, default=0.3, help='Probability of complex puzzles in batch')
    parser.add_argument('--seed', '-s', type=int, help='Random seed for reproducibility')
    parser.add_argument('--output', '-o', type=str, help='Output file (default: stdout)')
    parser.add_argument('--verify', action='store_true', help='Verify solvability (default: True)')
    
    args = parser.parse_args()
    
    generator = HanoiGenerator(seed=args.seed)
    
    try:
        if args.count == 1:
            # Generate single puzzle
            if args.complex:
                puzzle = generator.generate_complex_puzzle(args.disks, args.pegs)
            else:
                puzzle = generator.generate_puzzle(args.disks, args.pegs)
            
            if args.verify and not generator.verify_solvability(puzzle):
                print("Warning: Generated puzzle may not be solvable", file=sys.stderr)
            
            result = json.dumps(puzzle, indent=2)
        else:
            # Generate batch
            puzzles = generator.generate_batch(
                args.count, 
                args.min_disks, 
                args.disks,
                args.complex_prob
            )
            result = json.dumps(puzzles, indent=2)
    
    except ValueError as e:
        result = json.dumps({"error": f"Puzzle generation error: {e}"}, indent=2)
    except (IOError, OSError) as e:
        result = json.dumps({"error": f"File operation error: {e}"}, indent=2)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
    else:
        print(result)


if __name__ == '__main__':
    main() 