#!/usr/bin/env python3
"""
Tower of Hanoi Solver

Solves Tower of Hanoi puzzles using recursive algorithm.
Returns a compact, executable plan to reach the goal state.
"""

import sys
import json
import argparse
from typing import List, Tuple, Dict, Any


class HanoiSolver:
    """Solver for Tower of Hanoi puzzles."""
    
    def __init__(self):
        pass
    
    def solve(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Solve a Tower of Hanoi puzzle using BFS for general cases.
        
        Args:
            initial_state: Dict with 'pegs' (dict of peg_name -> list of disks) and 'num_disks'
            goal_state: Dict with 'pegs' (dict of peg_name -> list of disks) and 'num_disks'
        
        Returns:
            List of moves as (from_peg, to_peg) tuples
        """
        # Validate inputs
        if not self._validate_state(initial_state) or not self._validate_state(goal_state):
            raise ValueError("Invalid puzzle state")
        
        # Check if already solved
        if self._states_equal(initial_state, goal_state):
            return []
        
        # Try simple recursive approach first for efficiency
        if self._is_simple_case(initial_state, goal_state):
            return self._solve_simple_case(initial_state, goal_state)
        
        # Use BFS for complex cases
        return self._solve_with_bfs(initial_state, goal_state)
    
    def _hanoi_recursive(self, n: int, source: str, dest: str, aux: str, moves: List[Tuple[str, str]]):
        """Recursive Tower of Hanoi algorithm."""
        if n == 1:
            moves.append((source, dest))
        else:
            # Move n-1 disks from source to auxiliary
            self._hanoi_recursive(n-1, source, aux, dest, moves)
            # Move the largest disk from source to destination
            moves.append((source, dest))
            # Move n-1 disks from auxiliary to destination
            self._hanoi_recursive(n-1, aux, dest, source, moves)
    
    def _validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate that a state is properly formatted."""
        if 'pegs' not in state or 'num_disks' not in state:
            return False
        
        if not isinstance(state['pegs'], dict) or not isinstance(state['num_disks'], int):
            return False
        
        # Check that each peg contains a valid stack (larger disks at bottom)
        for peg_name, disks in state['pegs'].items():
            if not isinstance(disks, list):
                return False
            for i in range(len(disks) - 1):
                if disks[i] <= disks[i + 1]:  # Larger disks should have higher numbers
                    return False
        
        return True

    def _states_equal(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> bool:
        """Check if two states are equal."""
        return (state1['num_disks'] == state2['num_disks'] and 
                state1['pegs'] == state2['pegs'])
    
    def _is_simple_case(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> bool:
        """Check if this is a simple case (all disks start on one peg, need to end on another)."""
        # Find if there's a peg with all disks in initial state
        source_peg = None
        for peg_name, disks in initial_state['pegs'].items():
            if len(disks) == initial_state['num_disks']:
                source_peg = peg_name
                break
        
        # Find if there's a peg that should have all disks in goal state
        dest_peg = None
        for peg_name, disks in goal_state['pegs'].items():
            if len(disks) == goal_state['num_disks']:
                dest_peg = peg_name
                break
        
        return source_peg is not None and dest_peg is not None and source_peg != dest_peg
    
    def _solve_simple_case(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Solve simple case using recursive algorithm."""
        moves = []
        
        # Find source and destination pegs
        source_peg = None
        dest_peg = None
        
        for peg_name, disks in initial_state['pegs'].items():
            if len(disks) == initial_state['num_disks']:
                source_peg = peg_name
                break
        
        for peg_name, disks in goal_state['pegs'].items():
            if len(disks) == goal_state['num_disks']:
                dest_peg = peg_name
                break
        
        if source_peg is None or dest_peg is None:
            raise ValueError("Cannot solve simple case: source or destination peg not found")
        
        # Find auxiliary peg
        all_pegs = list(initial_state['pegs'].keys())
        aux_peg = [p for p in all_pegs if p not in [source_peg, dest_peg]][0]
        
        # Solve recursively
        self._hanoi_recursive(initial_state['num_disks'], source_peg, dest_peg, aux_peg, moves)
        
        return moves
    
    def _solve_with_bfs(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Solve complex cases using BFS."""
        from collections import deque
        
        # Convert states to hashable format
        start_state = self._state_to_tuple(initial_state)
        goal_state_tuple = self._state_to_tuple(goal_state)
        
        queue = deque([(start_state, [])])
        visited = {start_state}
        
        while queue:
            current_state, moves = queue.popleft()
            
            # Generate all possible moves
            for move in self._generate_valid_moves(current_state):
                new_state = self._apply_move_to_tuple(current_state, move)
                
                if new_state == goal_state_tuple:
                    return moves + [move]
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, moves + [move]))
        
        raise ValueError("No solution found")
    
    def _state_to_tuple(self, state: Dict[str, Any]) -> tuple:
        """Convert state to hashable tuple."""
        pegs = state['pegs']
        peg_names = sorted(pegs.keys())
        return tuple(tuple(pegs[peg]) for peg in peg_names)
    
    def _tuple_to_state(self, state_tuple: tuple, peg_names: List[str]) -> Dict[str, Any]:
        """Convert tuple back to state dict."""
        pegs = {peg_name: list(peg_disks) for peg_name, peg_disks in zip(peg_names, state_tuple)}
        num_disks = sum(len(disks) for disks in pegs.values())
        return {'pegs': pegs, 'num_disks': num_disks}
    
    def _generate_valid_moves(self, state_tuple: tuple) -> List[Tuple[str, str]]:
        """Generate all valid moves from current state."""
        # Convert back to dict for easier manipulation
        peg_names = [f"peg_{i}" for i in range(len(state_tuple))]
        state = self._tuple_to_state(state_tuple, peg_names)
        
        moves = []
        pegs = state['pegs']
        
        # Try moving top disk from each non-empty peg to each other peg
        for from_peg, from_disks in pegs.items():
            if not from_disks:  # Skip empty pegs
                continue
            
            top_disk = from_disks[-1]  # Top disk (smallest number at top)
            
            for to_peg, to_disks in pegs.items():
                if from_peg == to_peg:
                    continue
                
                # Check if move is valid (can only place smaller disk on larger disk)
                if not to_disks or to_disks[-1] > top_disk:
                    moves.append((from_peg, to_peg))
        
        return moves
    
    def _apply_move_to_tuple(self, state_tuple: tuple, move: Tuple[str, str]) -> tuple:
        """Apply a move to a state tuple."""
        peg_names = [f"peg_{i}" for i in range(len(state_tuple))]
        state = self._tuple_to_state(state_tuple, peg_names)
        
        from_peg, to_peg = move
        
        # Move top disk from source to destination
        disk = state['pegs'][from_peg].pop()
        state['pegs'][to_peg].append(disk)
        
        return self._state_to_tuple(state)

    def format_solution(self, moves: List[Tuple[str, str]]) -> str:
        """Format solution as a compact, executable plan."""
        if not moves:
            return "No moves needed"
        
        formatted_moves = []
        for i, (from_peg, to_peg) in enumerate(moves, 1):
            formatted_moves.append(f"{i}. Move disk from {from_peg} to {to_peg}")
        
        return "\n".join(formatted_moves)


def solve_from_json(puzzle_json: str) -> str:
    """Solve a puzzle from JSON input and return formatted solution."""
    try:
        puzzle = json.loads(puzzle_json)
        solver = HanoiSolver()
        moves = solver.solve(puzzle['initial_state'], puzzle['goal_state'])
        return solver.format_solution(moves)
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"
    except (KeyError, TypeError) as e:
        return f"Error in puzzle format: {e}"
    except ValueError as e:
        return f"Error solving puzzle: {e}"


def main():
    """Command-line interface for the Tower of Hanoi solver."""
    parser = argparse.ArgumentParser(description='Solve Tower of Hanoi puzzles')
    parser.add_argument('--input', '-i', type=str, help='Input JSON file (default: stdin)')
    parser.add_argument('--output', '-o', type=str, help='Output file (default: stdout)')
    parser.add_argument('--format', choices=['moves', 'json'], default='moves',
                       help='Output format: moves (human-readable) or json (machine-readable)')
    
    args = parser.parse_args()
    
    # Read input
    if args.input:
        with open(args.input, 'r') as f:
            input_data = f.read()
    else:
        input_data = sys.stdin.read()
    
    # Solve puzzle
    try:
        puzzle = json.loads(input_data)
        solver = HanoiSolver()
        moves = solver.solve(puzzle['initial_state'], puzzle['goal_state'])
        
        if args.format == 'json':
            result = json.dumps({
                'moves': moves,
                'num_moves': len(moves),
                'solution': solver.format_solution(moves)
            }, indent=2)
        else:
            result = solver.format_solution(moves)
    
    except json.JSONDecodeError as e:
        result = f"Error parsing JSON: {e}"
    except (KeyError, TypeError) as e:
        result = f"Error in puzzle format: {e}"
    except ValueError as e:
        result = f"Error solving puzzle: {e}"
    except (IOError, OSError) as e:
        result = f"Error reading/writing files: {e}"
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
    else:
        print(result)


if __name__ == '__main__':
    main() 