#!/usr/bin/env python3
"""
Smoke tests for Puzzle Generators

Tests generators to ensure they create valid, solvable puzzles.
"""

import sys
import os

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.hanoi_generator import HanoiGenerator
from generators.river_generator import RiverCrossingGenerator


def test_hanoi_generator_basic():
    """Test basic Hanoi puzzle generation."""
    generator = HanoiGenerator(seed=42)
    
    puzzle = generator.generate_puzzle(num_disks=3, num_pegs=3)
    
    # Verify puzzle structure
    assert 'initial_state' in puzzle
    assert 'goal_state' in puzzle
    assert 'metadata' in puzzle
    
    # Verify states have correct structure
    for state in [puzzle['initial_state'], puzzle['goal_state']]:
        assert 'pegs' in state
        assert 'num_disks' in state
        assert state['num_disks'] == 3
        assert len(state['pegs']) == 3
    
    # Verify puzzle is solvable
    assert generator.verify_solvability(puzzle)


def test_hanoi_generator_complex():
    """Test complex Hanoi puzzle generation."""
    generator = HanoiGenerator(seed=42)
    
    puzzle = generator.generate_complex_puzzle(num_disks=4, num_pegs=4)
    
    # Verify puzzle structure
    assert 'initial_state' in puzzle
    assert 'goal_state' in puzzle
    assert 'metadata' in puzzle
    assert puzzle['metadata']['puzzle_type'] == 'complex'
    
    # Verify puzzle is solvable
    assert generator.verify_solvability(puzzle)


def test_hanoi_generator_batch():
    """Test Hanoi batch generation."""
    generator = HanoiGenerator(seed=42)
    
    puzzles = generator.generate_batch(count=3, min_disks=2, max_disks=4)
    
    assert len(puzzles) <= 3  # May be fewer if some fail verification
    
    for puzzle in puzzles:
        assert 'puzzle_id' in puzzle['metadata']
        assert generator.verify_solvability(puzzle)


def test_river_generator_classic():
    """Test classic river puzzle generation."""
    generator = RiverCrossingGenerator(seed=42)
    
    puzzle = generator.generate_puzzle('classic')
    
    # Verify puzzle structure
    assert 'initial_state' in puzzle
    assert 'goal_state' in puzzle
    assert 'metadata' in puzzle
    assert puzzle['metadata']['variant'] == 'classic'
    
    # Classic puzzle: all start left, all end right
    for entity in ['farmer', 'wolf', 'goat', 'cabbage']:
        assert puzzle['initial_state'][entity] == 'left'
        assert puzzle['goal_state'][entity] == 'right'
    
    # Verify puzzle is solvable
    assert generator.verify_solvability(puzzle)


def test_river_generator_random_start():
    """Test random start river puzzle generation."""
    generator = RiverCrossingGenerator(seed=42)
    
    puzzle = generator.generate_puzzle('random_start')
    
    # Verify puzzle structure
    assert 'initial_state' in puzzle
    assert 'goal_state' in puzzle
    assert 'metadata' in puzzle
    assert puzzle['metadata']['variant'] == 'random_start'
    
    # Verify valid start state (no invalid combinations)
    entities = puzzle['metadata']['entities']
    assert generator.is_valid_start(puzzle['initial_state'], entities)
    
    # Verify puzzle is solvable
    assert generator.verify_solvability(puzzle)


def test_river_generator_partial_crossing():
    """Test partial crossing river puzzle generation."""
    generator = RiverCrossingGenerator(seed=42)
    
    puzzle = generator.generate_puzzle('partial_cross')
    
    # Verify puzzle structure
    assert 'initial_state' in puzzle
    assert 'goal_state' in puzzle
    assert 'metadata' in puzzle
    assert puzzle['metadata']['variant'] == 'partial_crossing'
    
    # Verify valid start state
    entities = puzzle['metadata']['entities']
    assert generator.is_valid_start(puzzle['initial_state'], entities)
    
    # Verify puzzle is solvable
    assert generator.verify_solvability(puzzle)


def test_river_generator_custom():
    """Test custom entity river puzzle generation."""
    generator = RiverCrossingGenerator(seed=42)
    
    puzzle = generator.generate_puzzle('custom')
    
    # Verify puzzle structure
    assert 'initial_state' in puzzle
    assert 'goal_state' in puzzle
    assert 'metadata' in puzzle
    assert puzzle['metadata']['variant'] == 'custom'
    
    # Verify valid start state
    entities = puzzle['metadata']['entities']
    assert generator.is_valid_start(puzzle['initial_state'], entities)
    
    # Verify puzzle is solvable
    assert generator.verify_solvability(puzzle)


def test_river_generator_batch():
    """Test river batch generation."""
    generator = RiverCrossingGenerator(seed=42)
    
    puzzles = generator.generate_batch(count=3)
    
    assert len(puzzles) <= 3  # May be fewer if some fail verification
    
    for puzzle in puzzles:
        assert 'puzzle_id' in puzzle['metadata']
        assert generator.verify_solvability(puzzle)


def test_river_generator_difficulty_progression():
    """Test river difficulty progression generation."""
    generator = RiverCrossingGenerator(seed=42)
    
    puzzles = generator.create_difficulty_progression(count=3)
    
    assert len(puzzles) == 3
    
    for i, puzzle in enumerate(puzzles):
        assert puzzle['metadata']['puzzle_id'] == i + 1
        assert puzzle['metadata']['progression_level'] == i + 1
        assert generator.verify_solvability(puzzle)


def test_river_validation_function():
    """Test the input validation function."""
    generator = RiverCrossingGenerator()
    
    entities = ['farmer', 'wolf', 'goat', 'cabbage']
    
    # Valid state: farmer with wolf and goat
    valid_state = {
        'farmer': 'left',
        'wolf': 'left',
        'goat': 'left',
        'cabbage': 'right'
    }
    assert generator.is_valid_start(valid_state, entities)
    
    # Invalid state: wolf and goat alone without farmer
    invalid_state1 = {
        'farmer': 'right',
        'wolf': 'left',
        'goat': 'left',
        'cabbage': 'right'
    }
    assert not generator.is_valid_start(invalid_state1, entities)
    
    # Invalid state: goat and cabbage alone without farmer
    invalid_state2 = {
        'farmer': 'right',
        'wolf': 'right',
        'goat': 'left',
        'cabbage': 'left'
    }
    assert not generator.is_valid_start(invalid_state2, entities)
    
    # Valid state: entities distributed with farmer present on each side
    valid_state2 = {
        'farmer': 'left',
        'wolf': 'right',
        'goat': 'left',
        'cabbage': 'right'
    }
    assert generator.is_valid_start(valid_state2, entities)


def test_hanoi_difficulty_estimation():
    """Test Hanoi difficulty estimation."""
    generator = HanoiGenerator()
    
    # Test difficulty levels
    assert generator._estimate_difficulty(1) == "easy"
    assert generator._estimate_difficulty(3) == "easy"
    assert generator._estimate_difficulty(4) == "medium"
    assert generator._estimate_difficulty(6) == "hard"
    assert generator._estimate_difficulty(8) == "expert"
    
    # Test complex puzzle difficulty (should be +1)
    assert generator._estimate_difficulty(3, complex_puzzle=True) == "medium"
    assert generator._estimate_difficulty(6, complex_puzzle=True) == "hard"
    assert generator._estimate_difficulty(7, complex_puzzle=True) == "expert"


def test_generators_with_invalid_inputs():
    """Test generators handle invalid inputs gracefully."""
    hanoi_gen = HanoiGenerator()
    river_gen = RiverCrossingGenerator()
    
    # Test Hanoi with invalid disk count
    try:
        hanoi_gen.generate_puzzle(num_disks=0)
        assert False, "Should raise ValueError for 0 disks"
    except ValueError:
        pass  # Expected
    
    # Test Hanoi with invalid peg count  
    try:
        hanoi_gen.generate_puzzle(num_disks=3, num_pegs=2)
        assert False, "Should raise ValueError for < 3 pegs"
    except ValueError:
        pass  # Expected
    
    # Test River with invalid variant
    try:
        river_gen.generate_puzzle('invalid_variant')
        assert False, "Should raise ValueError for invalid variant"
    except ValueError:
        pass  # Expected


if __name__ == '__main__':
    # Run tests manually if pytest not available
    test_hanoi_generator_basic()
    test_hanoi_generator_complex()
    test_hanoi_generator_batch()
    test_river_generator_classic()
    test_river_generator_random_start()
    test_river_generator_partial_crossing()
    test_river_generator_custom()
    test_river_generator_batch()
    test_river_generator_difficulty_progression()
    test_river_validation_function()
    test_hanoi_difficulty_estimation()
    test_generators_with_invalid_inputs()
    print("All generator tests passed!") 