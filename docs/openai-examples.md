# OpenAI Model Usage Examples

This project uses the official OpenAI Python library for better reliability and maintainability.

## Quick Start with Different Models

### Using GPT-4o Mini (Latest and Cost-Effective)
```bash
# Command line
python3 experiments/evaluate.py --model gpt-4o-mini-2024-07-18 --count 10

# Config file approach
# Edit experiments/templates/config.json and set:
# "model": "gpt-4o-mini-2024-07-18"
```

### Using Other Models
```bash
# GPT-4 (most capable)
python3 experiments/evaluate.py --model gpt-4 --count 5

# GPT-3.5 Turbo (faster, cheaper)
python3 experiments/evaluate.py --model gpt-3.5-turbo --count 20

# GPT-4 Turbo
python3 experiments/evaluate.py --model gpt-4-turbo --count 5
```

## Implementation Comparison

### New Approach (Current) ✅
Uses the official OpenAI library:

```python
from openai import OpenAI

client = OpenAI(api_key=your_key)
response = client.chat.completions.create(
    model="gpt-4o-mini-2024-07-18",
    messages=[{"role": "user", "content": "Solve this puzzle..."}]
)
```

**Benefits:**
- ✅ Official OpenAI library (recommended approach)
- ✅ Automatic retry handling and error management
- ✅ Built-in rate limiting and timeout handling
- ✅ Better compatibility with future API changes
- ✅ Cleaner, more maintainable code
- ✅ Automatic token counting and usage tracking

### Old Approach (Replaced) ❌
Used raw HTTP requests:

```python
import requests

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": model, "messages": messages}
)
```

**Issues:**
- ❌ Manual error handling and retries
- ❌ No built-in rate limiting
- ❌ More complex token counting
- ❌ Higher maintenance overhead

## Installation

Make sure you have the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install "openai>=1.0.0"
```

## Environment Setup

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your-actual-api-key-here
```

## Testing Without API Key

Use the mock client for development and testing:
```bash
python3 experiments/evaluate.py --mock --count 5
```

This allows you to test the entire pipeline without making real API calls or using credits. 