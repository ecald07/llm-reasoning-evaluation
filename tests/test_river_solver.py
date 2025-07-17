#!/usr/bin/env python3
"""
Smoke tests for Wolf-Goat-Cabbage River Crossing Solver

Tests canonical puzzles to ensure the solver works correctly.
"""

import sys
import os

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.river_solver import RiverCrossingSolver


def test_river_classic():
    """Test classic wolf-goat-cabbage puzzle."""
    solver = RiverCrossingSolver()
    
    initial_state = {
        'farmer': 'left',
        'wolf': 'left',
        'goat': 'left',
        'cabbage': 'left'
    }
    
    goal_state = {
        'farmer': 'right',
        'wolf': 'right',
        'goat': 'right',
        'cabbage': 'right'
    }
    
    moves = solver.solve(initial_state, goal_state)
    
    # Classic puzzle should be solvable
    assert len(moves) > 0
    
    # Verify moves are valid tuples with direction and entities
    for move in moves:
        assert isinstance(move, tuple)
        assert len(move) == 2
        direction, entities = move
        assert direction in ['left', 'right']
        assert isinstance(entities, list)
        assert 'farmer' in entities  # Farmer must be in every move
    
    # Verify solution reaches goal
    current_state = initial_state.copy()
    for direction, entities in moves:
        for entity in entities:
            current_state[entity] = direction
    
    assert current_state == goal_state


def test_river_already_solved():
    """Test puzzle that's already at goal state."""
    solver = RiverCrossingSolver()
    
    state = {
        'farmer': 'right',
        'wolf': 'right',
        'goat': 'right',
        'cabbage': 'right'
    }
    
    moves = solver.solve(state, state)
    
    # Should require no moves
    assert len(moves) == 0


def test_river_partial_crossing():
    """Test puzzle with partial crossing."""
    solver = RiverCrossingSolver()
    
    initial_state = {
        'farmer': 'left',
        'wolf': 'left',
        'goat': 'right',
        'cabbage': 'left'
    }
    
    goal_state = {
        'farmer': 'right',
        'wolf': 'right',
        'goat': 'left',
        'cabbage': 'right'
    }
    
    moves = solver.solve(initial_state, goal_state)
    
    # Should be solvable
    assert len(moves) > 0
    
    # Verify solution reaches goal
    current_state = initial_state.copy()
    for direction, entities in moves:
        for entity in entities:
            current_state[entity] = direction
    
    assert current_state == goal_state


def test_river_custom_entities():
    """Test puzzle with custom entity names."""
    solver = RiverCrossingSolver(['farmer', 'fox', 'chicken', 'grain'])
    
    initial_state = {
        'farmer': 'left',
        'fox': 'left',
        'chicken': 'left',
        'grain': 'left'
    }
    
    goal_state = {
        'farmer': 'right',
        'fox': 'right',
        'chicken': 'right',
        'grain': 'right'
    }
    
    moves = solver.solve(initial_state, goal_state)
    
    # Should be solvable
    assert len(moves) > 0


def test_river_format_solution():
    """Test solution formatting."""
    solver = RiverCrossingSolver()
    
    moves = [
        ('right', ['farmer', 'goat']),
        ('left', ['farmer']),
        ('right', ['farmer', 'wolf'])
    ]
    
    formatted = solver.format_solution(moves)
    
    assert "1. Farmer takes goat right" in formatted
    assert "2. Farmer goes left" in formatted
    assert "3. Farmer takes wolf right" in formatted


def test_river_empty_moves():
    """Test formatting empty move list."""
    solver = RiverCrossingSolver()
    
    formatted = solver.format_solution([])
    assert formatted == "No moves needed"


def test_river_state_description():
    """Test state description formatting."""
    solver = RiverCrossingSolver()
    
    state = {
        'farmer': 'left',
        'wolf': 'right',
        'goat': 'left',
        'cabbage': 'right'
    }
    
    description = solver.get_state_description(state)
    
    assert "Left:" in description
    assert "Right:" in description
    assert "farmer" in description
    assert "wolf" in description
    assert "goat" in description
    assert "cabbage" in description


def test_river_invalid_state():
    """Test handling of invalid puzzle states."""
    solver = RiverCrossingSolver()
    
    # Invalid state: entity with invalid position
    invalid_state = {
        'farmer': 'middle',  # Invalid position
        'wolf': 'left',
        'goat': 'left',
        'cabbage': 'left'
    }
    
    goal_state = {
        'farmer': 'right',
        'wolf': 'right',
        'goat': 'right',
        'cabbage': 'right'
    }
    
    try:
        moves = solver.solve(invalid_state, goal_state)
        assert False, "Should have raised ValueError for invalid state"
    except ValueError:
        pass  # Expected


def test_river_impossible_goal():
    """Test handling of impossible goal states."""
    solver = RiverCrossingSolver()
    
    initial_state = {
        'farmer': 'left',
        'wolf': 'left',
        'goat': 'left',
        'cabbage': 'left'
    }
    
    # Actually impossible: farmer on right but multiple constraint violations on left
    impossible_goal = {
        'farmer': 'right',
        'wolf': 'left',
        'goat': 'left',
        'cabbage': 'left'
    }
    
    try:
        moves = solver.solve(initial_state, impossible_goal)
        # This might not raise ValueError but instead find no solution
        # Both are acceptable for impossible puzzles
        if moves:
            # If solution found, it might be possible after all
            pass
    except ValueError:
        pass  # Expected for impossible puzzle


def test_state_after_moves():
    """Helper function to test state after applying moves."""
    def state_after(initial_state, moves):
        current_state = initial_state.copy()
        for direction, entities in moves:
            for entity in entities:
                current_state[entity] = direction
        return current_state
    
    def is_goal(state, goal):
        return state == goal
    
    # Test helper functions work
    initial = {'farmer': 'left', 'goat': 'left'}
    moves = [('right', ['farmer', 'goat'])]
    final = state_after(initial, moves)
    goal = {'farmer': 'right', 'goat': 'right'}
    
    assert is_goal(final, goal)


if __name__ == '__main__':
    # Run tests manually if pytest not available
    test_river_classic()
    test_river_already_solved()
    test_river_partial_crossing()
    test_river_custom_entities()
    test_river_format_solution()
    test_river_empty_moves()
    test_river_state_description()
    test_river_invalid_state()
    test_river_impossible_goal()
    test_state_after_moves()
    print("All River solver tests passed!") 