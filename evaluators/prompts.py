#!/usr/bin/env python3
"""
Prompt Templates for LLM Reasoning Evaluation

Contains system prompts and user prompt templates for both verbose and compact
evaluation modes for Tower of Hanoi and River Crossing puzzles.
"""

from typing import Dict, Any


class PromptTemplates:
    """Collection of prompt templates for different puzzle types and modes."""
    
    # Tower of Hanoi Prompts
    HANOI_SYSTEM_VERBOSE = """You are an expert puzzle solver. You will be given a Tower of Hanoi puzzle to solve.

Tower of Hanoi Rules:
1. Only one disk can be moved at a time
2. Only the top disk from any stack can be moved
3. A larger disk may never be placed on top of a smaller disk

Your task is to find the sequence of moves to transform the initial configuration into the goal configuration.

Format your final answer as a list of moves:
moves = [(from_peg, to_peg), (from_peg, to_peg), ...]

For example: moves = [("peg_0", "peg_2"), ("peg_0", "peg_1"), ("peg_2", "peg_1")]

Show your reasoning step by step, then provide the complete move list."""

    HANOI_SYSTEM_COMPACT = """You are an expert puzzle solver. You will be given a Tower of Hanoi puzzle to solve.

Instead of listing all individual moves, provide a Python function that generates the solution efficiently.

Your function should:
1. Use recursion or iteration to solve the puzzle
2. Yield or return move tuples in the format (from_peg, to_peg)
3. Be executable Python code that produces the correct sequence

Format your answer as executable Python code within triple backticks.
Example:
```python
def solve_hanoi(n, source, dest, aux):
    if n == 1:
        yield (source, dest)
    else:
        yield from solve_hanoi(n-1, source, aux, dest)
        yield (source, dest)
        yield from solve_hanoi(n-1, aux, dest, source)

# Generate the solution
moves = list(solve_hanoi(3, 'peg_0', 'peg_2', 'peg_1'))
print(moves)
```"""

    # River Crossing Prompts
    RIVER_SYSTEM_VERBOSE = """You are an expert puzzle solver. You will be given a river crossing puzzle to solve.

River Crossing Rules:
1. The farmer must accompany any entity across the river
2. Certain entities cannot be left alone together without the farmer:
   - Predator and prey (e.g., wolf and goat)
   - Prey and food (e.g., goat and cabbage)
3. The goal is to get all entities to their target positions

Your task is to find the sequence of moves to transform the initial state into the goal state.

Format your final answer as a list of moves:
moves = [(direction, [entities]), (direction, [entities]), ...]

Where direction is "left" or "right" and entities is a list of who travels.
For example: moves = [("right", ["farmer", "goat"]), ("left", ["farmer"]), ("right", ["farmer", "wolf"])]

Show your reasoning step by step, then provide the complete move list."""

    RIVER_SYSTEM_COMPACT = """You are an expert puzzle solver. You will be given a river crossing puzzle to solve.

Instead of manually working through each move, provide a Python function that solves the puzzle efficiently.

Your function should:
1. Use search algorithms (BFS, DFS) or logical reasoning
2. Return move tuples in the format (direction, [entities])
3. Be executable Python code that produces the correct sequence
4. Handle the constraint validation automatically

Format your answer as executable Python code within triple backticks.
Example:
```python
def solve_river_crossing(initial_state, goal_state):
    # Your solution logic here
    moves = [
        ("right", ["farmer", "goat"]),
        ("left", ["farmer"]),
        ("right", ["farmer", "wolf"]),
        ("left", ["farmer", "goat"]),
        ("right", ["farmer", "cabbage"]),
        ("left", ["farmer"]),
        ("right", ["farmer", "goat"])
    ]
    return moves

solution = solve_river_crossing(initial_state, goal_state)
print(solution)
```"""

    @staticmethod
    def format_hanoi_puzzle(puzzle: Dict[str, Any]) -> str:
        """Format a Tower of Hanoi puzzle for the prompt."""
        initial_state = puzzle['initial_state']
        goal_state = puzzle['goal_state']
        
        prompt = f"""Tower of Hanoi Puzzle:

Initial state:
"""
        
        # Format initial state
        for peg_name, disks in initial_state['pegs'].items():
            if disks:
                disk_str = ', '.join(map(str, disks))
                prompt += f"  {peg_name}: [{disk_str}] (bottom to top)\n"
            else:
                prompt += f"  {peg_name}: [empty]\n"
        
        prompt += f"\nGoal state:\n"
        
        # Format goal state
        for peg_name, disks in goal_state['pegs'].items():
            if disks:
                disk_str = ', '.join(map(str, disks))
                prompt += f"  {peg_name}: [{disk_str}] (bottom to top)\n"
            else:
                prompt += f"  {peg_name}: [empty]\n"
        
        prompt += f"\nNumber of disks: {initial_state['num_disks']}"
        
        return prompt

    @staticmethod
    def format_river_puzzle(puzzle: Dict[str, Any]) -> str:
        """Format a River Crossing puzzle for the prompt."""
        initial_state = puzzle['initial_state']
        goal_state = puzzle['goal_state']
        entities = puzzle['metadata']['entities']
        
        prompt = f"""River Crossing Puzzle:

Entities: {', '.join(entities)}

Initial state:
"""
        
        # Group entities by side
        left_entities = [e for e in entities if initial_state[e] == 'left']
        right_entities = [e for e in entities if initial_state[e] == 'right']
        
        prompt += f"  Left side: {', '.join(left_entities) if left_entities else 'empty'}\n"
        prompt += f"  Right side: {', '.join(right_entities) if right_entities else 'empty'}\n"
        
        prompt += f"\nGoal state:\n"
        
        # Group goal entities by side
        left_goal = [e for e in entities if goal_state[e] == 'left']
        right_goal = [e for e in entities if goal_state[e] == 'right']
        
        prompt += f"  Left side: {', '.join(left_goal) if left_goal else 'empty'}\n"
        prompt += f"  Right side: {', '.join(right_goal) if right_goal else 'empty'}\n"
        
        # Add constraint information
        if len(entities) >= 4:
            farmer = entities[0]
            predator = entities[1]
            prey = entities[2]
            food = entities[3]
            
            prompt += f"\nConstraints:\n"
            prompt += f"  - {predator} and {prey} cannot be alone together without {farmer}\n"
            prompt += f"  - {prey} and {food} cannot be alone together without {farmer}\n"
            prompt += f"  - {farmer} must be present for any river crossing\n"
        
        return prompt

    @staticmethod
    def get_system_prompt(puzzle_type: str, mode: str) -> str:
        """
        Get the appropriate system prompt.
        
        Args:
            puzzle_type: "hanoi" or "river"
            mode: "verbose" or "compact"
            
        Returns:
            System prompt string
        """
        if puzzle_type == "hanoi":
            if mode == "verbose":
                return PromptTemplates.HANOI_SYSTEM_VERBOSE
            else:
                return PromptTemplates.HANOI_SYSTEM_COMPACT
        elif puzzle_type == "river":
            if mode == "verbose":
                return PromptTemplates.RIVER_SYSTEM_VERBOSE
            else:
                return PromptTemplates.RIVER_SYSTEM_COMPACT
        else:
            raise ValueError(f"Unknown puzzle type: {puzzle_type}")

    @staticmethod
    def create_prompt(puzzle: Dict[str, Any], mode: str = "verbose") -> tuple[str, str]:
        """
        Create a complete prompt for a puzzle.
        
        Args:
            puzzle: Puzzle dictionary
            mode: "verbose" or "compact"
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Determine puzzle type from metadata
        if 'num_disks' in puzzle.get('metadata', {}):
            puzzle_type = "hanoi"
            user_prompt = PromptTemplates.format_hanoi_puzzle(puzzle)
        elif 'entities' in puzzle.get('metadata', {}):
            puzzle_type = "river"
            user_prompt = PromptTemplates.format_river_puzzle(puzzle)
        else:
            raise ValueError("Cannot determine puzzle type from metadata")
        
        system_prompt = PromptTemplates.get_system_prompt(puzzle_type, mode)
        
        return system_prompt, user_prompt


# Additional utility functions
def create_evaluation_prompt(puzzle: Dict[str, Any], mode: str = "verbose") -> tuple[str, str]:
    """
    Convenience function to create evaluation prompts.
    
    Args:
        puzzle: Puzzle dictionary from generator
        mode: "verbose" for step-by-step solutions, "compact" for function-based solutions
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return PromptTemplates.create_prompt(puzzle, mode)


if __name__ == "__main__":
    # Test the prompt templates
    print("Testing prompt templates...")
    
    # Example Tower of Hanoi puzzle
    hanoi_puzzle = {
        'initial_state': {
            'pegs': {
                'peg_0': [3, 2, 1],
                'peg_1': [],
                'peg_2': []
            },
            'num_disks': 3
        },
        'goal_state': {
            'pegs': {
                'peg_0': [],
                'peg_1': [],
                'peg_2': [3, 2, 1]
            },
            'num_disks': 3
        },
        'metadata': {
            'num_disks': 3,
            'num_pegs': 3,
            'difficulty': 'easy'
        }
    }
    
    # Example River Crossing puzzle
    river_puzzle = {
        'initial_state': {
            'farmer': 'left',
            'wolf': 'left', 
            'goat': 'left',
            'cabbage': 'left'
        },
        'goal_state': {
            'farmer': 'right',
            'wolf': 'right',
            'goat': 'right', 
            'cabbage': 'right'
        },
        'metadata': {
            'entities': ['farmer', 'wolf', 'goat', 'cabbage'],
            'variant': 'classic',
            'difficulty': 'easy'
        }
    }
    
    # Test Hanoi prompts
    print("\n=== Tower of Hanoi Verbose ===")
    system, user = create_evaluation_prompt(hanoi_puzzle, "verbose")
    print("System:", system[:100] + "...")
    print("User:", user[:200] + "...")
    
    print("\n=== Tower of Hanoi Compact ===")
    system, user = create_evaluation_prompt(hanoi_puzzle, "compact")
    print("System:", system[:100] + "...")
    print("User:", user[:200] + "...")
    
    # Test River prompts
    print("\n=== River Crossing Verbose ===")
    system, user = create_evaluation_prompt(river_puzzle, "verbose")
    print("System:", system[:100] + "...")
    print("User:", user[:200] + "...")
    
    print("\n=== River Crossing Compact ===")
    system, user = create_evaluation_prompt(river_puzzle, "compact")
    print("System:", system[:100] + "...")
    print("User:", user[:200] + "...") 