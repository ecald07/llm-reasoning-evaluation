# LLM Reasoning Evaluation

A fair-play testbed for evaluating how well small-to-mid-size language models perform on classic planning puzzles. This project implements two foundational reasoning tasks: the **Tower of Hanoi** and the **Wolf-Goat-Cabbage River Crossing** puzzle.

## Overview

Inspired by Apple's "Illusion of Thinking" paper (June 2025) and Anthropic's rebuttal, this testbed provides a systematic way to evaluate whether language models can truly "reason" or are simply pattern matching on training data.

## Project Structure

```
llm-reasoning-evaluation/
├── hanoi_generator.py    # Generate Tower of Hanoi puzzles
├── hanoi_solver.py       # Solve Tower of Hanoi puzzles
├── river_generator.py    # Generate River Crossing puzzles
├── river_solver.py       # Solve River Crossing puzzles
└── README.md            # This file
```

## Features

### Tower of Hanoi
- **Randomized Generation**: Create puzzles with 3+ disks and pegs
- **Complex Variants**: Distributed starting positions for advanced reasoning
- **Optimal Solutions**: Uses recursive algorithm for simple cases, BFS for complex ones
- **Verification**: All puzzles verified for solvability before output

### River Crossing
- **Multiple Variants**: Classic, random start, partial crossing, custom entities
- **Flexible Entities**: Supports different entity sets (wolf/goat/cabbage, cat/mouse/cheese, etc.)
- **Constraint Validation**: Proper predator-prey relationship enforcement
- **Optimal Pathfinding**: BFS ensures shortest solution paths

## Installation

This project requires Python 3.7+ with only standard library dependencies.

```bash
# Clone or download the project files
cd llm-reasoning-evaluation

# Make scripts executable (optional)
chmod +x *.py
```

## Usage

### Basic Pipeline

The core design allows piping generators into solvers:

```bash
# Generate and solve a Tower of Hanoi puzzle
python3 hanoi_generator.py --disks 3 | python3 hanoi_solver.py

# Generate and solve a River Crossing puzzle  
python3 river_generator.py --variant classic | python3 river_solver.py
```

### Tower of Hanoi Examples

```bash
# Simple 3-disk puzzle
python3 hanoi_generator.py --disks 3 --seed 42

# Complex 4-disk puzzle with distributed start
python3 hanoi_generator.py --disks 4 --complex --seed 123

# Batch generation (5 puzzles, 3-5 disks)
python3 hanoi_generator.py --count 5 --min-disks 3 --disks 5

# Generate and solve with JSON output
python3 hanoi_generator.py --disks 3 | python3 hanoi_solver.py --format json
```

### River Crossing Examples

```bash
# Classic wolf-goat-cabbage puzzle
python3 river_generator.py --variant classic

# Random starting positions
python3 river_generator.py --variant random_start --seed 456

# Custom entities (cat, mouse, cheese, etc.)
python3 river_generator.py --variant custom --seed 789

# Difficulty progression (5 puzzles of increasing difficulty)
python3 river_generator.py --progression --count 5

# Solve with state visualization
python3 river_generator.py --variant classic | python3 river_solver.py --show-states
```

### Command Line Options

#### Generators
Both generators support:
- `--count, -c`: Number of puzzles to generate
- `--seed, -s`: Random seed for reproducibility  
- `--output, -o`: Output file (default: stdout)
- `--verify`: Verify solvability (default: true)

#### Solvers
Both solvers support:
- `--input, -i`: Input file (default: stdin)
- `--output, -o`: Output file (default: stdout)
- `--format`: Output format (`moves` or `json`)

#### Tower of Hanoi Specific
```bash
# Generator options
--disks, -d DISKS        # Number of disks (or max for batch)
--min-disks MIN_DISKS    # Minimum disks for batch generation  
--pegs, -p PEGS          # Number of pegs (default: 3)
--complex                # Generate complex distributed puzzles
--complex-prob PROB      # Probability of complex puzzles in batch
```

#### River Crossing Specific
```bash
# Generator options
--variant, -v VARIANT    # Puzzle variant: classic, random_start, partial_cross, custom
--batch                  # Generate batch with mixed variants
--progression            # Generate difficulty progression

# Solver options  
--show-states            # Show intermediate states in solution
```

## Output Formats

### Human-Readable (Default)
```
1. Move disk from peg_0 to peg_1
2. Move disk from peg_0 to peg_2
3. Move disk from peg_1 to peg_2
```

### JSON Format
```json
{
  "moves": [["peg_0", "peg_1"], ["peg_0", "peg_2"], ["peg_1", "peg_2"]],
  "num_moves": 3,
  "solution": "1. Move disk from peg_0 to peg_1\n2. Move disk from peg_0 to peg_2\n3. Move disk from peg_1 to peg_2"
}
```

## Evaluation Workflow

### For LLM Testing
1. **Generate Test Set**: Create a variety of puzzles with known solutions
2. **Prompt LLM**: Provide puzzle description and ask for solution
3. **Validate Response**: Use solvers to verify LLM's proposed solution
4. **Analyze Results**: Compare solution quality, path length, correctness

### Example Evaluation Pipeline
```bash
# Generate test set
python3 hanoi_generator.py --count 10 --min-disks 3 --disks 6 --seed 2024 > test_hanoi.json
python3 river_generator.py --progression --count 10 --seed 2024 > test_river.json

# Get reference solutions
cat test_hanoi.json | python3 hanoi_solver.py --format json > solutions_hanoi.json
cat test_river.json | python3 river_solver.py --format json > solutions_river.json

# Test LLM outputs against reference solutions
# (implement your own comparison script)
```

## Puzzle Complexity

### Tower of Hanoi Difficulty Levels
- **Easy (3 disks)**: 7 moves, basic recursive thinking
- **Medium (4-5 disks)**: 15-31 moves, multi-step planning
- **Hard (6+ disks)**: 63+ moves, extended reasoning chains
- **Complex variants**: Distributed starting positions, non-standard configurations

### River Crossing Difficulty Levels
- **Easy**: Classic puzzle, clear constraints
- **Medium**: Random starting positions, partial crossings
- **Hard**: Custom entities, mixed constraints
- **Expert**: Multiple valid solution paths

## Technical Notes

### Algorithms Used
- **Tower of Hanoi**: Recursive solution for simple cases, BFS for complex distributions
- **River Crossing**: Breadth-first search for optimal pathfinding
- **Validation**: Complete state-space verification for puzzle solvability

### Performance Characteristics
- **Memory**: O(3^n) for Tower of Hanoi BFS, O(2^4) for River Crossing
- **Time**: Sub-second for puzzles up to 6 disks / 4 entities
- **Scalability**: Tested up to 8 disks, 5 pegs; 5+ entity river crossings

## Contributing

This is a research testbed. Contributions welcome for:
- Additional puzzle variants
- Performance optimizations  
- New constraint types
- Evaluation metrics
- LLM integration examples

## License

Open source for research and educational use.

## Citation

If you use this testbed in research, please cite:
```
LLM Reasoning Evaluation Testbed
Tower of Hanoi and River Crossing Puzzles
GitHub: [repository-url]
``` 