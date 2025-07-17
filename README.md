# LLM Reasoning Evaluation

A fair-play testbed for evaluating how well small-to-mid-size language models perform on classic planning puzzles. This project implements two foundational reasoning tasks: the **Tower of Hanoi** and the **Wolf-Goat-Cabbage River Crossing** puzzle.

## Overview

Inspired by Apple's "Illusion of Thinking" paper and Alex Lawsen's "The Illusion of the Illusion of Thinking: A Comment on Shojaee et al. (2025)" from Open Philanthropy, this testbed provides a systematic way to evaluate whether language models can truly "reason" or are simply pattern matching on training data.

## Project Structure

```
llm-reasoning-evaluation/
├── hanoi_generator.py      # Generate Tower of Hanoi puzzles
├── hanoi_solver.py         # Solve Tower of Hanoi puzzles
├── river_generator.py      # Generate River Crossing puzzles
├── river_solver.py         # Solve River Crossing puzzles
├── evaluators/             # LLM evaluation framework
│   ├── llm_client.py       # GPT-3.5-turbo API integration
│   ├── prompts.py          # Verbose vs compact prompt templates
│   ├── executor.py         # Safe code execution sandbox
│   └── metrics.py          # Token counting and accuracy measurement
├── experiments/            # Evaluation pipeline scripts
│   ├── evaluate.py         # Main evaluation harness
│   ├── compare.py          # Results comparison and analysis
│   └── config.json         # Example configuration
├── results/                # Evaluation results (auto-generated)
├── tests/                  # Comprehensive test suite
└── README.md              # This file
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

## 🚀 Quick Start (First Time Users)

**Step 1: Install dependencies**
```bash
pip3 install -r requirements.txt
```

**Step 2: Test that everything works (no API key needed)**
```bash
python3 experiments/evaluate.py --mock --puzzles hanoi --count 3 --mode both
```
You should see: `✓ 7 moves, 271 tokens` and `Results saved to: results/...`

**Step 3: Run with a real model (needs OpenAI API key)**
```bash
export OPENAI_API_KEY="your-key-here"
python3 experiments/evaluate.py --model gpt-3.5-turbo --puzzles hanoi --count 5 --mode both
```

**Step 4: Check your results**
```bash
ls results/
python3 experiments/compare.py --directory results/
```

**That's it!** You now have evaluation results comparing verbose vs compact prompting approaches.

---

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

## LLM Evaluation Framework

### Quick Start

**Basic evaluation with mock client (for testing):**
```bash
# Test verbose mode with 3 Tower of Hanoi puzzles
python3 experiments/evaluate.py --mock --puzzles hanoi --count 3 --mode verbose

# Test compact mode with 3 River Crossing puzzles  
python3 experiments/evaluate.py --mock --puzzles river --count 3 --mode compact

# Compare both modes
python3 experiments/evaluate.py --mock --puzzles hanoi --count 5 --mode both
```

**Real evaluation with GPT-3.5-turbo:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run full evaluation comparing verbose vs compact modes
python3 experiments/evaluate.py --model gpt-3.5-turbo --puzzles hanoi --count 10 --mode both

# Use configuration file for complex experiments
python3 experiments/evaluate.py --config experiments/config.json
```

### Key Features

**🎯 Solves the "Impossible Puzzle Problem"**
- Only generates verified solvable puzzles
- River crossing avoids N≥6 boat capacity issues from Shojaee et al.
- Built-in solvability verification before evaluation

**🔄 Compact vs Verbose Evaluation**
- **Verbose Mode**: Traditional step-by-step move enumeration
- **Compact Mode**: Function-based solutions to avoid token limits
- Direct comparison of accuracy and efficiency

**🔒 Safe Code Execution**
- Sandboxed execution environment for LLM-generated code
- Timeout protection and input validation
- Prevents dangerous operations while allowing puzzle solving

**📊 Comprehensive Metrics**
- Token usage tracking (prompt + completion)
- Solution correctness verification
- Move efficiency analysis
- Statistical comparisons across modes

### Evaluation Pipeline

The evaluation harness coordinates these steps:

1. **Generate Puzzles**: Create verified solvable instances
2. **Create Prompts**: Format for verbose or compact mode
3. **Query LLM**: Call GPT-3.5-turbo with appropriate prompts
4. **Execute Solutions**: Run generated code safely or parse moves
5. **Verify Correctness**: Check if solution reaches goal state
6. **Collect Metrics**: Record tokens, timing, and accuracy
7. **Generate Reports**: Compare modes and analyze results

### Example Results

```
=== EVALUATION SUMMARY ===

VERBOSE MODE:
  Success Rate: 85.0%
  Correctness Rate: 78.0%
  Avg Tokens: 342
  Avg Time: 2.1s

COMPACT MODE:
  Success Rate: 90.0%
  Correctness Rate: 82.0%
  Avg Tokens: 156
  Avg Time: 1.8s

COMPACT vs VERBOSE:
  Token Savings: 54.4%
  Time Savings: 14.3%
  Accuracy Change: +4.0%
```

### Configuration

Use `experiments/config.json` for complex experiments:

```json
{
  "model": "gpt-3.5-turbo",
  "puzzles": "hanoi",
  "count": 20,
  "mode": "both",
  "min_difficulty": 3,
  "max_difficulty": 6,
  "seed": 42,
  "timeout": 30.0
}
```

### Results Analysis

Compare evaluation runs:

```bash
# Compare two result files
python3 experiments/compare.py results/eval1.json results/eval2.json

# Aggregate metrics across multiple runs
python3 experiments/compare.py --directory results/ --model gpt-3.5-turbo --aggregate

# Filter by puzzle type
python3 experiments/compare.py --directory results/ --puzzle-type hanoi
```

### Research Applications

This framework enables systematic study of:

- **Token Efficiency**: How much can compact representations save?
- **Reasoning vs Execution**: Do models understand algorithms or just pattern match?
- **Scaling Behavior**: How does performance change with puzzle complexity?
- **Model Comparison**: Which models excel at different reasoning tasks?

### Supported Models

- **GPT-3.5-turbo** (tested)
- **GPT-4** (compatible API)
- Any OpenAI-compatible API endpoint
- Mock client for testing and development

## Contributing

This is a research testbed. Contributions welcome for:
- Additional puzzle variants
- Performance optimizations  
- New constraint types
- Evaluation metrics
- Additional LLM model integrations
- Advanced analysis tools

## License

Open source for research and educational use.

## Citation

If you use this testbed in research, please cite:
```
LLM Reasoning Evaluation Testbed
Tower of Hanoi and River Crossing Puzzles with Compact vs Verbose Evaluation
GitHub: [repository-url]
``` 