#!/usr/bin/env python3
"""
Wolf-Goat-Cabbage River Crossing Puzzle Generator

Generates randomized river crossing puzzle instances and verifies solvability.
"""

import sys
import json
import random
import argparse
from typing import Dict, List, Any, Optional
from solvers.river_solver import RiverCrossingSolver


class RiverCrossingGenerator:
    """Generator for Wolf-Goat-Cabbage river crossing puzzles."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.solver = RiverCrossingSolver()
        self.entities = ['farmer', 'wolf', 'goat', 'cabbage']
        
        # Alternative entity sets for variety
        self.entity_variants = [
            ['farmer', 'wolf', 'goat', 'cabbage'],
            ['farmer', 'fox', 'chicken', 'grain'],
            ['farmer', 'lion', 'rabbit', 'carrots'],
            ['farmer', 'cat', 'mouse', 'cheese'],
            ['farmer', 'snake', 'bird', 'seeds']
        ]
    
    def _is_valid_initial_state(self, state: Dict[str, str]) -> bool:
        """
        Check if an initial state is valid (no forbidden combinations without farmer).
        
        Args:
            state: Dictionary with entity positions
            
        Returns:
            True if state is valid, False if predator/prey are alone without farmer
        """
        if len(state) < 4:
            return True  # Not enough entities for standard constraints
        
        entities = list(state.keys())
        controller = entities[0]  # Assume first entity is the farmer
        
        if len(entities) >= 4:
            entity1 = entities[1]  # predator (wolf/cat/etc)
            entity2 = entities[2]  # prey/middle (goat/mouse/etc)
            entity3 = entities[3]  # food (cabbage/cheese/etc)
            
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
    
    def is_valid_start(self, state: Dict[str, str], entities: List[str]) -> bool:
        """
        Check if a start state is valid (no predator-prey pair alone without farmer).
        
        Args:
            state: The state to validate
            entities: List of entities in order [farmer, predator, prey, food]
        
        Returns:
            True if valid, False if invalid
        """
        if len(entities) < 4:
            return True  # Not enough entities for standard constraints
        
        farmer, predator, prey, food = entities[:4]
        
        # Check if predator and prey are alone without farmer
        if (state.get(predator) == state.get(prey) and 
            state.get(farmer) != state.get(predator)):
            return False
        
        # Check if prey and food are alone without farmer
        if (state.get(prey) == state.get(food) and 
            state.get(farmer) != state.get(prey)):
            return False
        
        return True
    
    def generate_puzzle(self, variant: str = 'classic') -> Dict[str, Any]:
        """
        Generate a river crossing puzzle.
        
        Args:
            variant: Type of puzzle ('classic', 'random_start', 'partial_cross', 'custom')
        
        Returns:
            Dict containing the puzzle with initial_state and goal_state
        """
        # Choose entity set
        if variant == 'custom':
            entities = random.choice(self.entity_variants)
        else:
            entities = self.entities
        
        if variant == 'classic':
            return self._generate_classic_puzzle(entities)
        elif variant == 'random_start':
            return self._generate_random_start_puzzle(entities)
        elif variant == 'partial_cross':
            return self._generate_partial_crossing_puzzle(entities)
        elif variant == 'custom':
            return self._generate_custom_puzzle(entities)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    
    def _generate_classic_puzzle(self, entities: List[str]) -> Dict[str, Any]:
        """Generate the classic version: all start left, all end right."""
        initial_state = {entity: 'left' for entity in entities}
        goal_state = {entity: 'right' for entity in entities}
        
        return {
            'initial_state': initial_state,
            'goal_state': goal_state,
            'metadata': {
                'variant': 'classic',
                'entities': entities,
                'difficulty': 'easy',
                'description': 'Classic river crossing: transport all entities from left to right'
            }
        }
    
    def _generate_random_start_puzzle(self, entities: List[str]) -> Dict[str, Any]:
        """Generate puzzle with random starting positions."""
        initial_state = {}
        goal_state = {}
        
        # Randomly assign initial positions, ensuring validity
        max_attempts = 50
        for attempt in range(max_attempts):
            initial_state = {}
            for entity in entities:
                initial_state[entity] = random.choice(['left', 'right'])
            
            # Check if this starting state is valid
            if self.is_valid_start(initial_state, entities):
                break
        else:
            # Fallback to classic start if we can't generate a valid random start
            initial_state = {entity: 'left' for entity in entities}
        
        # Goal is usually the opposite side for most entities
        for entity in entities:
            if random.random() < 0.8:  # 80% chance to flip side
                goal_state[entity] = 'right' if initial_state[entity] == 'left' else 'left'
            else:
                goal_state[entity] = initial_state[entity]
        
        # Ensure farmer position makes sense (needs to move if others need to move)
        entities_need_moving = [e for e in entities if initial_state[e] != goal_state[e]]
        if entities_need_moving and 'farmer' not in entities_need_moving:
            # Farmer must move to help others
            goal_state['farmer'] = 'right' if initial_state['farmer'] == 'left' else 'left'
        
        return {
            'initial_state': initial_state,
            'goal_state': goal_state,
            'metadata': {
                'variant': 'random_start',
                'entities': entities,
                'difficulty': 'medium',
                'description': 'Random starting positions'
            }
        }
    
    def _generate_partial_crossing_puzzle(self, entities: List[str]) -> Dict[str, Any]:
        """Generate puzzle where some entities start on different sides."""
        initial_state = {}
        goal_state = {}
        
        # Generate valid partial crossing state
        max_attempts = 50
        for attempt in range(max_attempts):
            initial_state = {}
            
            # Start with farmer on left (standard)
            initial_state['farmer'] = 'left'
            
            # Randomly distribute other entities
            other_entities = [e for e in entities if e != 'farmer']
            for entity in other_entities:
                initial_state[entity] = random.choice(['left', 'right'])
            
            # Check if this starting state is valid
            if self.is_valid_start(initial_state, entities):
                break
        else:
            # Fallback to classic start if we can't generate a valid partial start
            initial_state = {entity: 'left' for entity in entities}
        
        # Goal: get specific entities to specific sides
        # Ensure at least one entity needs to move
        goal_state['farmer'] = random.choice(['left', 'right'])
        
        other_entities = [e for e in entities if e != 'farmer']
        for entity in other_entities:
            # 70% chance to be on opposite side from start
            if random.random() < 0.7:
                goal_state[entity] = 'right' if initial_state[entity] == 'left' else 'left'
            else:
                goal_state[entity] = initial_state[entity]
        
        return {
            'initial_state': initial_state,
            'goal_state': goal_state,
            'metadata': {
                'variant': 'partial_crossing',
                'entities': entities,
                'difficulty': 'medium',
                'description': 'Partial river crossing with mixed starting positions'
            }
        }
    
    def _generate_custom_puzzle(self, entities: List[str]) -> Dict[str, Any]:
        """Generate puzzle with custom entity names but same constraint structure."""
        # Generate random positions, ensuring validity
        initial_state = {}
        goal_state = {}
        
        max_attempts = 50
        for attempt in range(max_attempts):
            initial_state = {}
            for entity in entities:
                initial_state[entity] = random.choice(['left', 'right'])
            
            # Check if this starting state is valid
            if self.is_valid_start(initial_state, entities):
                break
        else:
            # Fallback to classic start if we can't generate a valid custom start
            initial_state = {entity: 'left' for entity in entities}
        
        for entity in entities:
            goal_state[entity] = random.choice(['left', 'right'])
        
        # Ensure puzzle is not trivial (at least farmer needs to move)
        if initial_state == goal_state:
            goal_state['farmer'] = 'right' if initial_state['farmer'] == 'left' else 'left'
        
        return {
            'initial_state': initial_state,
            'goal_state': goal_state,
            'metadata': {
                'variant': 'custom',
                'entities': entities,
                'difficulty': 'hard',
                'description': f'Custom entities: {", ".join(entities)}'
            }
        }
    
    def verify_solvability(self, puzzle: Dict[str, Any]) -> bool:
        """
        Verify that a puzzle is solvable using the river_solver.
        
        Args:
            puzzle: The puzzle to verify
        
        Returns:
            True if solvable, False otherwise
        """
        try:
            # Create solver with appropriate entities
            solver = RiverCrossingSolver()
            solver.entities = puzzle['metadata']['entities']
            
            moves = solver.solve(puzzle['initial_state'], puzzle['goal_state'])
            return True
        except (ValueError, KeyError) as e:
            # These are expected errors for unsolvable puzzles
            return False
    
    def generate_batch(self, count: int, variants: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate a batch of puzzles with varying types.
        
        Args:
            count: Number of puzzles to generate
            variants: List of variant types to include
        
        Returns:
            List of verified solvable puzzles
        """
        if variants is None:
            variants = ['classic', 'random_start', 'partial_cross', 'custom']
        
        puzzles = []
        attempts = 0
        max_attempts = count * 20  # Prevent infinite loops
        
        while len(puzzles) < count and attempts < max_attempts:
            attempts += 1
            
            try:
                variant = random.choice(variants)
                puzzle = self.generate_puzzle(variant)
                
                if self.verify_solvability(puzzle):
                    puzzle['metadata']['puzzle_id'] = len(puzzles) + 1
                    puzzles.append(puzzle)
            
            except (ValueError, KeyError, TypeError) as e:
                # Skip puzzles that can't be generated or verified
                continue
        
        return puzzles
    
    def create_difficulty_progression(self, count: int = 5) -> List[Dict[str, Any]]:
        """Create a series of puzzles with increasing difficulty."""
        puzzles = []
        
        # Easy: Classic puzzle
        if count >= 1:
            puzzle = self.generate_puzzle('classic')
            puzzle['metadata']['puzzle_id'] = 1
            puzzle['metadata']['progression_level'] = 1
            puzzles.append(puzzle)
        
        # Medium: Random start positions
        if count >= 2:
            puzzle = self.generate_puzzle('random_start')
            puzzle['metadata']['puzzle_id'] = 2
            puzzle['metadata']['progression_level'] = 2
            puzzles.append(puzzle)
        
        # Medium-Hard: Partial crossing
        if count >= 3:
            puzzle = self.generate_puzzle('partial_cross')
            puzzle['metadata']['puzzle_id'] = 3
            puzzle['metadata']['progression_level'] = 3
            puzzles.append(puzzle)
        
        # Hard: Custom entities
        remaining = count - len(puzzles)
        for i in range(remaining):
            puzzle = self.generate_puzzle('custom')
            if self.verify_solvability(puzzle):
                puzzle['metadata']['puzzle_id'] = len(puzzles) + 1
                puzzle['metadata']['progression_level'] = 4 + i
                puzzles.append(puzzle)
        
        return puzzles


def main():
    """Command-line interface for the river crossing generator."""
    parser = argparse.ArgumentParser(description='Generate Wolf-Goat-Cabbage river crossing puzzles')
    parser.add_argument('--count', '-c', type=int, default=1, help='Number of puzzles to generate')
    parser.add_argument('--variant', '-v', choices=['classic', 'random_start', 'partial_cross', 'custom'], 
                       default='classic', help='Puzzle variant type')
    parser.add_argument('--batch', action='store_true', help='Generate batch with mixed variants')
    parser.add_argument('--progression', action='store_true', help='Generate difficulty progression')
    parser.add_argument('--seed', '-s', type=int, help='Random seed for reproducibility')
    parser.add_argument('--output', '-o', type=str, help='Output file (default: stdout)')
    parser.add_argument('--verify', action='store_true', default=True, help='Verify solvability (default: True)')
    
    args = parser.parse_args()
    
    generator = RiverCrossingGenerator(seed=args.seed)
    
    try:
        if args.progression:
            puzzles = generator.create_difficulty_progression(args.count)
            result = json.dumps(puzzles, indent=2)
        elif args.batch or args.count > 1:
            puzzles = generator.generate_batch(args.count)
            result = json.dumps(puzzles, indent=2)
        else:
            # Generate single puzzle
            puzzle = generator.generate_puzzle(args.variant)
            
            if args.verify and not generator.verify_solvability(puzzle):
                print("Warning: Generated puzzle may not be solvable", file=sys.stderr)
            
            result = json.dumps(puzzle, indent=2)
    
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