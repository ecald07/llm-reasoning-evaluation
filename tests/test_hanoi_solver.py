#!/usr/bin/env python3
"""
Smoke tests for Tower of Hanoi Solver

Tests canonical puzzles to ensure the solver works correctly.
"""

import sys
import os

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.hanoi_solver import HanoiSolver


def test_hanoi_3_disks():
    """Test classic 3-disk Tower of Hanoi puzzle."""
    solver = HanoiSolver()
    
    initial_state = {
        'pegs': {
            'A': [3, 2, 1],  # All disks on peg A (largest to smallest)
            'B': [],
            'C': []
        },
        'num_disks': 3
    }
    
    goal_state = {
        'pegs': {
            'A': [],
            'B': [],
            'C': [3, 2, 1]  # All disks on peg C
        },
        'num_disks': 3
    }
    
    moves = solver.solve(initial_state, goal_state)
    
    # 3-disk Hanoi should take exactly 2^3 - 1 = 7 moves
    assert len(moves) == 7
    
    # Verify moves are valid tuples
    for move in moves:
        assert isinstance(move, tuple)
        assert len(move) == 2
        assert move[0] in ['A', 'B', 'C']
        assert move[1] in ['A', 'B', 'C']
        assert move[0] != move[1]


def test_hanoi_1_disk():
    """Test trivial 1-disk Tower of Hanoi puzzle."""
    solver = HanoiSolver()
    
    initial_state = {
        'pegs': {
            'A': [1],
            'B': [],
            'C': []
        },
        'num_disks': 1
    }
    
    goal_state = {
        'pegs': {
            'A': [],
            'B': [1],
            'C': []
        },
        'num_disks': 1
    }
    
    moves = solver.solve(initial_state, goal_state)
    
    # 1-disk should take exactly 1 move
    assert len(moves) == 1
    assert moves[0] == ('A', 'B')


def test_hanoi_2_disks():
    """Test 2-disk Tower of Hanoi puzzle."""
    solver = HanoiSolver()
    
    initial_state = {
        'pegs': {
            'A': [2, 1],
            'B': [],
            'C': []
        },
        'num_disks': 2
    }
    
    goal_state = {
        'pegs': {
            'A': [],
            'B': [],
            'C': [2, 1]
        },
        'num_disks': 2
    }
    
    moves = solver.solve(initial_state, goal_state)
    
    # 2-disk Hanoi should take exactly 2^2 - 1 = 3 moves
    assert len(moves) == 3


def test_hanoi_already_solved():
    """Test puzzle that's already at goal state."""
    solver = HanoiSolver()
    
    state = {
        'pegs': {
            'A': [],
            'B': [],
            'C': [3, 2, 1]
        },
        'num_disks': 3
    }
    
    moves = solver.solve(state, state)
    
    # Should require no moves
    assert len(moves) == 0


def test_hanoi_different_pegs():
    """Test Hanoi with different source and destination pegs."""
    solver = HanoiSolver()
    
    initial_state = {
        'pegs': {
            'A': [],
            'B': [2, 1],
            'C': []
        },
        'num_disks': 2
    }
    
    goal_state = {
        'pegs': {
            'A': [2, 1],
            'B': [],
            'C': []
        },
        'num_disks': 2
    }
    
    moves = solver.solve(initial_state, goal_state)
    
    # Should still take 3 moves for 2 disks
    assert len(moves) == 3


def test_hanoi_format_solution():
    """Test solution formatting."""
    solver = HanoiSolver()
    
    moves = [('A', 'C'), ('A', 'B'), ('C', 'B')]
    formatted = solver.format_solution(moves)
    
    assert "1. Move disk from A to C" in formatted
    assert "2. Move disk from A to B" in formatted
    assert "3. Move disk from C to B" in formatted


def test_hanoi_empty_moves():
    """Test formatting empty move list."""
    solver = HanoiSolver()
    
    formatted = solver.format_solution([])
    assert formatted == "No moves needed"


def test_hanoi_invalid_state():
    """Test handling of invalid puzzle states."""
    solver = HanoiSolver()
    
    # Invalid state: disk 1 on top of disk 2 (should be other way around)
    invalid_state = {
        'pegs': {
            'A': [1, 2],  # Invalid order
            'B': [],
            'C': []
        },
        'num_disks': 2
    }
    
    goal_state = {
        'pegs': {
            'A': [],
            'B': [],
            'C': [2, 1]
        },
        'num_disks': 2
    }
    
    try:
        moves = solver.solve(invalid_state, goal_state)
        assert False, "Should have raised ValueError for invalid state"
    except ValueError:
        pass  # Expected


if __name__ == '__main__':
    # Run tests manually if pytest not available
    test_hanoi_3_disks()
    test_hanoi_1_disk()
    test_hanoi_2_disks()
    test_hanoi_already_solved()
    test_hanoi_different_pegs()
    test_hanoi_format_solution()
    test_hanoi_empty_moves()
    test_hanoi_invalid_state()
    print("All Hanoi solver tests passed!") 