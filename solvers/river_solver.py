#!/usr/bin/env python3
"""
Wolf-Goat-Cabbage River Crossing Solver

Solves the classic river crossing puzzle using breadth-first search.
Returns a compact, executable plan to reach the goal state.
"""

import sys
import json
import argparse
from typing import List, Tuple, Dict, Any, Set, Optional
from collections import deque


class RiverCrossingSolver:
    """Solver for Wolf-Goat-Cabbage river crossing puzzles."""
    
    def __init__(self, entities: Optional[List[str]] = None):
        if entities is None:
            self.entities = ['farmer', 'wolf', 'goat', 'cabbage']
        else:
            self.entities = entities
    
    def solve(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
        """
        Solve a river crossing puzzle using BFS.
        
        Args:
            initial_state: Dict with entity positions {'farmer': 'left', 'wolf': 'left', ...}
            goal_state: Dict with target entity positions
        
        Returns:
            List of moves as (direction, [entities]) tuples
        """
        # Auto-detect entities from the input state
        self.entities = list(initial_state.keys())
        
        if not self._validate_state(initial_state) or not self._validate_state(goal_state):
            raise ValueError("Invalid puzzle state")
        
        # Convert to internal representation
        start_state = self._dict_to_tuple(initial_state)
        goal_state_tuple = self._dict_to_tuple(goal_state)
        
        if start_state == goal_state_tuple:
            return []  # Already at goal
        
        # BFS to find shortest solution
        queue = deque([(start_state, [])])
        visited: Set[Tuple[str, ...]] = {start_state}
        
        while queue:
            current_state, moves = queue.popleft()
            
            # Generate all possible moves
            for move in self._generate_moves(current_state):
                new_state = self._apply_move(current_state, move)
                
                if new_state == goal_state_tuple:
                    return moves + [move]
                
                if new_state not in visited and self._is_valid_state(new_state):
                    visited.add(new_state)
                    queue.append((new_state, moves + [move]))
        
        raise ValueError("No solution found")
    
    def _dict_to_tuple(self, state_dict: Dict[str, str]) -> Tuple[str, ...]:
        """Convert state dict to tuple for hashing."""
        return tuple(state_dict[entity] for entity in self.entities)
    
    def _tuple_to_dict(self, state_tuple: Tuple[str, ...]) -> Dict[str, str]:
        """Convert state tuple back to dict."""
        return {entity: side for entity, side in zip(self.entities, state_tuple)}
    
    def _validate_state(self, state: Dict[str, str]) -> bool:
        """Validate that a state is properly formatted."""
        if not isinstance(state, dict):
            return False
        
        for entity in self.entities:
            if entity not in state or state[entity] not in ['left', 'right']:
                return False
        
        return True
    
    def _is_valid_state(self, state_tuple: Tuple[str, ...]) -> bool:
        """Check if a state is valid (no forbidden combinations)."""
        state = self._tuple_to_dict(state_tuple)
        
        # Assume first entity is the controller (farmer)
        if len(self.entities) < 4:
            return True  # Not enough entities for standard constraints
        
        controller = self.entities[0]
        entity1 = self.entities[1]  # predator (wolf/cat/etc)
        entity2 = self.entities[2]  # prey/middle (goat/mouse/etc)
        entity3 = self.entities[3]  # food (cabbage/cheese/etc)
        
        controller_side = state[controller]
        
        # Check each side of the river
        for side in ['left', 'right']:
            entities_on_side = [entity for entity in [entity1, entity2, entity3] 
                              if state[entity] == side]
            
            # If controller is on this side, any combination is safe
            if controller_side == side:
                continue
            
            # If controller is NOT on this side, check for forbidden combinations
            if entity1 in entities_on_side and entity2 in entities_on_side:
                return False  # Predator would eat prey
            if entity2 in entities_on_side and entity3 in entities_on_side:
                return False  # Prey would eat food
        
        return True
    
    def _generate_moves(self, state_tuple: Tuple[str, ...]) -> List[Tuple[str, List[str]]]:
        """Generate all possible moves from current state."""
        state = self._tuple_to_dict(state_tuple)
        
        # Use first entity as controller
        controller = self.entities[0]
        controller_side = state[controller]
        target_side = 'right' if controller_side == 'left' else 'left'
        
        moves = []
        
        # Controller goes alone
        moves.append((target_side, [controller]))
        
        # Controller takes one other entity
        other_entities = self.entities[1:]  # All entities except controller
        for entity in other_entities:
            if state[entity] == controller_side:  # Entity is on same side as controller
                moves.append((target_side, [controller, entity]))
        
        return moves
    
    def _apply_move(self, state_tuple: Tuple[str, ...], move: Tuple[str, List[str]]) -> Tuple[str, ...]:
        """Apply a move to a state and return new state."""
        state = self._tuple_to_dict(state_tuple)
        target_side, entities = move
        
        for entity in entities:
            state[entity] = target_side
        
        return self._dict_to_tuple(state)
    
    def format_solution(self, moves: List[Tuple[str, List[str]]]) -> str:
        """Format solution as a compact, executable plan."""
        if not moves:
            return "No moves needed"
        
        controller = self.entities[0]
        controller_title = controller.capitalize()
        
        formatted_moves = []
        for i, (direction, entities) in enumerate(moves, 1):
            if len(entities) == 1:
                action = f"{controller_title} goes {direction}"
            else:
                other_entities = [e for e in entities if e != controller]
                action = f"{controller_title} takes {', '.join(other_entities)} {direction}"
            
            formatted_moves.append(f"{i}. {action}")
        
        return "\n".join(formatted_moves)
    
    def get_state_description(self, state: Dict[str, str]) -> str:
        """Get a human-readable description of the current state."""
        left_entities = [entity for entity in self.entities if state[entity] == 'left']
        right_entities = [entity for entity in self.entities if state[entity] == 'right']
        
        return f"Left: {', '.join(left_entities) if left_entities else 'empty'} | Right: {', '.join(right_entities) if right_entities else 'empty'}"


def solve_from_json(puzzle_json: str) -> str:
    """Solve a puzzle from JSON input and return formatted solution."""
    try:
        puzzle = json.loads(puzzle_json)
        solver = RiverCrossingSolver()
        moves = solver.solve(puzzle['initial_state'], puzzle['goal_state'])
        return solver.format_solution(moves)
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"
    except (KeyError, TypeError) as e:
        return f"Error in puzzle format: {e}"
    except ValueError as e:
        return f"Error solving puzzle: {e}"


def main():
    """Command-line interface for the river crossing solver."""
    parser = argparse.ArgumentParser(description='Solve Wolf-Goat-Cabbage river crossing puzzles')
    parser.add_argument('--input', '-i', type=str, help='Input JSON file (default: stdin)')
    parser.add_argument('--output', '-o', type=str, help='Output file (default: stdout)')
    parser.add_argument('--format', choices=['moves', 'json'], default='moves',
                       help='Output format: moves (human-readable) or json (machine-readable)')
    parser.add_argument('--show-states', action='store_true', help='Show intermediate states')
    
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
        solver = RiverCrossingSolver()
        moves = solver.solve(puzzle['initial_state'], puzzle['goal_state'])
        
        if args.format == 'json':
            result_data = {
                'moves': moves,
                'num_moves': len(moves),
                'solution': solver.format_solution(moves)
            }
            
            if args.show_states:
                # Show intermediate states
                current_state = puzzle['initial_state'].copy()
                states = [current_state.copy()]
                
                for direction, entities in moves:
                    for entity in entities:
                        current_state[entity] = direction
                    states.append(current_state.copy())
                
                result_data['states'] = states
                result_data['state_descriptions'] = [solver.get_state_description(state) for state in states]
            
            result = json.dumps(result_data, indent=2)
        else:
            result = solver.format_solution(moves)
            
            if args.show_states:
                # Add state descriptions
                current_state = puzzle['initial_state'].copy()
                result += f"\n\nInitial state: {solver.get_state_description(current_state)}"
                
                for i, (direction, entities) in enumerate(moves, 1):
                    for entity in entities:
                        current_state[entity] = direction
                    result += f"\nAfter move {i}: {solver.get_state_description(current_state)}"
    
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