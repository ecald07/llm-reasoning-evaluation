#!/usr/bin/env python3
"""
LLM Client for Reasoning Evaluation

Handles API calls to language models (GPT-3.5-turbo and compatible models).
Includes token counting, error handling, and response parsing.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI


class LLMClient:
    """Client for interacting with large language models via OpenAI API."""
    
    def __init__(self, 
                 model: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize LLM client using the official OpenAI library.
        
        Args:
            model: Model name (e.g., "gpt-3.5-turbo", "gpt-4", "gpt-4o-mini-2024-07-18")
            api_key: API key (if None, reads from OPENAI_API_KEY env var)
            base_url: API base URL
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=60.0
        )
        self.retry_delay = retry_delay
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Note: This is a rough approximation. For exact counts, use tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token for English text
        return max(1, len(text) // 4)
    
    def create_chat_completion(self, 
                             messages: List[Dict[str, str]], 
                             temperature: float = 0.7,
                             max_tokens: Optional[int] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Create a chat completion using the OpenAI library.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters
            
        Returns:
            API response dict (converted from OpenAI response object)
            
        Raises:
            Exception: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Convert OpenAI response object to dict for compatibility
            return {
                "choices": [
                    {
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "id": response.id
            }
            
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}")
    
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: Optional[int] = None) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response_text, token_usage_dict)
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract response text
        response_text = response["choices"][0]["message"]["content"]
        
        # Extract token usage
        usage = response.get("usage", {})
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
        
        return response_text, token_usage
    
    def validate_api_key(self) -> bool:
        """
        Test if the API key is valid by making a simple API call.
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            self.generate_response("Hello", max_tokens=5)
            return True
        except Exception:
            return False


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls."""
    
    def __init__(self):
        # Initialize without requiring API key
        self.model = "mock-gpt-3.5-turbo"
        self.api_key = "mock-key"
        self.base_url = "mock://api"
        self.max_retries = 1
        self.retry_delay = 0.1
    
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: Optional[int] = None) -> Tuple[str, Dict[str, int]]:
        """
        Generate a mock response for testing.
        """
        # Mock responses for different puzzle types
        if "Tower of Hanoi" in prompt or "hanoi" in prompt.lower():
            if "compact" in prompt.lower() or "function" in prompt.lower():
                response = """Here's a recursive function to solve Tower of Hanoi:

```python
def solve_hanoi(n, source, dest, aux):
    if n == 1:
        yield (source, dest)
    else:
        yield from solve_hanoi(n-1, source, aux, dest)
        yield (source, dest)
        yield from solve_hanoi(n-1, aux, dest, source)

# Generate solution
moves = list(solve_hanoi(3, 'peg_0', 'peg_2', 'peg_1'))
print(moves)
```"""
            else:
                response = """To solve this Tower of Hanoi puzzle, I need to move all disks from the source peg to the destination peg.

moves = [("peg_0", "peg_2"), ("peg_0", "peg_1"), ("peg_2", "peg_1"), ("peg_0", "peg_2"), ("peg_1", "peg_0"), ("peg_1", "peg_2"), ("peg_0", "peg_2")]"""
        
        elif "river crossing" in prompt.lower() or "farmer" in prompt.lower():
            if "compact" in prompt.lower() or "function" in prompt.lower():
                response = """Here's a function to solve the river crossing:

```python
def solve_river_crossing():
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

solution = solve_river_crossing()
print(solution)
```"""
            else:
                response = """To solve the river crossing puzzle:

moves = [("right", ["farmer", "goat"]), ("left", ["farmer"]), ("right", ["farmer", "wolf"]), ("left", ["farmer", "goat"]), ("right", ["farmer", "cabbage"]), ("left", ["farmer"]), ("right", ["farmer", "goat"])]"""
        
        else:
            response = "I need more information to solve this puzzle."
        
        # Mock token usage
        prompt_tokens = self.count_tokens(prompt + (system_prompt or ""))
        completion_tokens = self.count_tokens(response)
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
        
        return response, token_usage
    
    def validate_api_key(self) -> bool:
        """Mock validation always returns True."""
        return True


def create_llm_client(model: str = "gpt-3.5-turbo", 
                     api_key: Optional[str] = None,
                     mock: bool = False) -> LLMClient:
    """
    Factory function to create appropriate LLM client.
    
    Args:
        model: Model name
        api_key: API key (optional)
        mock: If True, return MockLLMClient for testing
        
    Returns:
        LLMClient instance
    """
    if mock:
        return MockLLMClient()
    else:
        return LLMClient(model=model, api_key=api_key)


if __name__ == "__main__":
    # Test the client
    print("Testing LLM Client...")
    
    # Test with mock client
    mock_client = create_llm_client(mock=True)
    response, usage = mock_client.generate_response("Solve Tower of Hanoi with 3 disks")
    print(f"Mock response: {response[:100]}...")
    print(f"Token usage: {usage}")
    
    # Test with real client (if API key available)
    try:
        real_client = create_llm_client()
        if real_client.validate_api_key():
            print("Real API client is working!")
        else:
            print("Real API client validation failed (check API key)")
    except ValueError as e:
        print(f"Real API client not available: {e}") 