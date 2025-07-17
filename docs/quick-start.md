# 🚀 Quick Start Guide

## What This Does
Tests if language models can actually "reason" by giving them puzzle-solving tasks.
Compares two approaches: asking for step-by-step moves vs asking for code that solves it.

## Step-by-Step Instructions

### 1. Install Python packages
```bash
pip install -r requirements.txt
```

### 2. Test without any API (takes 30 seconds)
```bash
python3 experiments/evaluate.py --mock --puzzles hanoi --count 3 --mode both
```

**What you should see:**
```
✓ 7 moves, 271 tokens
✓ 7 moves, 318 tokens  
Results saved to: results/evaluation_hanoi_gpt-3.5-turbo_XXXXX.json
```

### 3. Run with real OpenAI models (optional)
```bash
# Set your API key (get one from https://openai.com/api/)
export OPENAI_API_KEY="sk-your-key-here"

# Run evaluation 
python3 experiments/evaluate.py --model gpt-3.5-turbo --puzzles hanoi --count 5 --mode both
```

### 4. Check your results
```bash
# See what files were created
ls results/

# Compare the results  
python3 experiments/compare.py --directory results/
```

### 5. Understanding the Output

**VERBOSE MODE**: Model lists every single move step-by-step
**COMPACT MODE**: Model writes code that generates the moves

The evaluation shows:
- **Success Rate**: How often the model gave valid solutions
- **Token Usage**: How many words/tokens each approach used  
- **Accuracy**: Whether solutions actually solve the puzzle

## Quick Examples

**Different puzzles:**
```bash
python3 experiments/evaluate.py --mock --puzzles river --count 3 --mode both
```

**More challenging:**
```bash
python3 experiments/evaluate.py --mock --puzzles hanoi --count 10 --min-difficulty 4 --max-difficulty 6
```

**Just compact mode:**
```bash
python3 experiments/evaluate.py --mock --puzzles hanoi --count 5 --mode compact
```

## What Next?

- Check the `results/` folder for detailed JSON files with all the data
- Look at `README.md` for complete documentation
- The framework addresses methodological issues in recent AI reasoning research

**That's it! You're now evaluating LLM reasoning capabilities.** 