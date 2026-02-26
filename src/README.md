<div align="center">
<h1>
  <div class="image-wrapper" style="display: inline-block;">
    <picture>
      <source media="(prefers-color-scheme: dark)" alt="logo" height="150" srcset="https://raw.githubusercontent.com/trycua/cua/main/img/logo_white.png" style="display: block; margin: auto;">
      <source media="(prefers-color-scheme: light)" alt="logo" height="150" srcset="https://raw.githubusercontent.com/trycua/cua/main/img/logo_black.png" style="display: block; margin: auto;">
      <img alt="Shows my svg">
    </picture>
  </div>

  [![Python](https://img.shields.io/badge/Python-333333?logo=python&logoColor=white&labelColor=333333)](#)
  [![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=F0F0F0)](#)
  [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.com/invite/mVnXXpdE85)
  [![PyPI](https://img.shields.io/pypi/v/cua-computer?color=333333)](https://pypi.org/project/cua-computer/)
</h1>
</div>

**cua-agent** is a general Computer-Use framework with liteLLM integration for running agentic workflows on macOS, Windows, and Linux sandboxes. It provides a unified interface for computer-use agents across multiple LLM providers with advanced callback system for extensibility.

## Features

- **Safe Computer-Use/Tool-Use**: Using Computer SDK for sandboxed desktops
- **Multi-Agent Support**: Anthropic Claude, OpenAI computer-use-preview, UI-TARS, Omniparser + any LLM
- **Multi-API Support**: Take advantage of liteLLM supporting 100+ LLMs / model APIs, including local models (`huggingface-local/`, `ollama_chat/`, `mlx/`)
- **Cross-Platform**: Works on Windows, macOS, and Linux with cloud and local computer instances
- **Extensible Callbacks**: Built-in support for image retention, cache control, PII anonymization, budget limits, and trajectory tracking

## Install

```bash
pip install "cua-agent[all]"

# or install specific providers
pip install "cua-agent[openai]"        # OpenAI computer-use-preview support
pip install "cua-agent[anthropic]"     # Anthropic Claude support
pip install "cua-agent[omni]"          # Omniparser + any LLM support
pip install "cua-agent[uitars]"        # UI-TARS
pip install "cua-agent[uitars-mlx]"    # UI-TARS + MLX support
pip install "cua-agent[uitars-hf]"     # UI-TARS + Huggingface support
pip install "cua-agent[glm45v-hf]"     # GLM-4.5V + Huggingface support
pip install "cua-agent[ui]"            # Gradio UI support
```

## Quick Start

```python
import asyncio
import os
from agent import ComputerAgent
from computer import Computer

async def main():
    # Set up computer instance
    async with Computer(
        os_type="linux",
        provider_type="cloud",
        name=os.getenv("CUA_CONTAINER_NAME"),
        api_key=os.getenv("CUA_API_KEY")
    ) as computer:
        
        # Create agent
        agent = ComputerAgent(
            model="anthropic/claude-3-5-sonnet-20241022",
            tools=[computer],
            only_n_most_recent_images=3,
            trajectory_dir="trajectories",
            max_trajectory_budget=5.0  # $5 budget limit
        )
        
        # Run agent
        messages = [{"role": "user", "content": "Take a screenshot and tell me what you see"}]
        
        async for result in agent.run(messages):
            for item in result["output"]:
                if item["type"] == "message":
                    print(item["content"][0]["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Supported Models

### Anthropic Claude (Computer Use API)
```python
model="anthropic/claude-3-5-sonnet-20241022"
model="anthropic/claude-3-7-sonnet-20250219"
model="anthropic/claude-opus-4-20250514"
model="anthropic/claude-sonnet-4-20250514"
```

### OpenAI Computer Use Preview
```python
model="openai/computer-use-preview"
```

### UI-TARS (Local or Huggingface Inference)
```python
model="huggingface-local/ByteDance-Seed/UI-TARS-1.5-7B"
model="ollama_chat/0000/ui-tars-1.5-7b"
```

### Omniparser + Any LLM
```python
model="omniparser+ollama_chat/mistral-small3.2"
model="omniparser+vertex_ai/gemini-pro"
model="omniparser+anthropic/claude-3-5-sonnet-20241022"
model="omniparser+openai/gpt-4o"
```

## Custom Tools

Define custom tools using decorated functions:

```python
from computer.helpers import sandboxed

@sandboxed()
def read_file(location: str) -> str:
    """Read contents of a file
    
    Parameters
    ----------
    location : str
        Path to the file to read
        
    Returns
    -------
    str
        Contents of the file or error message
    """
    try:
        with open(location, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def calculate(a: int, b: int) -> int:
    """Calculate the sum of two integers"""
    return a + b

# Use with agent
agent = ComputerAgent(
    model="anthropic/claude-3-5-sonnet-20241022",
    tools=[computer, read_file, calculate]
)
```

## Callbacks System

agent provides a comprehensive callback system for extending functionality:

### Built-in Callbacks

```python
from agent.callbacks import (
    ImageRetentionCallback,
    TrajectorySaverCallback, 
    BudgetManagerCallback,
    LoggingCallback
)

agent = ComputerAgent(
    model="anthropic/claude-3-5-sonnet-20241022",
    tools=[computer],
    callbacks=[
        ImageRetentionCallback(only_n_most_recent_images=3),
        TrajectorySaverCallback(trajectory_dir="trajectories"),
        BudgetManagerCallback(max_budget=10.0, raise_error=True),
        LoggingCallback(level=logging.INFO)
    ]
)
```

### Custom Callbacks

```python
from agent.callbacks.base import AsyncCallbackHandler

class CustomCallback(AsyncCallbackHandler):
    async def on_llm_start(self, messages):
        """Preprocess messages before LLM call"""
        # Add custom preprocessing logic
        return messages
    
    async def on_llm_end(self, messages):
        """Postprocess messages after LLM call"""
        # Add custom postprocessing logic
        return messages
    
    async def on_usage(self, usage):
        """Track usage information"""
        print(f"Tokens used: {usage.total_tokens}")
```

## Budget Management

Control costs with built-in budget management:

```python
# Simple budget limit
agent = ComputerAgent(
    model="anthropic/claude-3-5-sonnet-20241022",
    max_trajectory_budget=5.0  # $5 limit
)

# Advanced budget configuration
agent = ComputerAgent(
    model="anthropic/claude-3-5-sonnet-20241022",
    max_trajectory_budget={
        "max_budget": 10.0,
        "raise_error": True,  # Raise error when exceeded
        "reset_after_each_run": False  # Persistent across runs
    }
)
```

## Trajectory Management

Save and replay agent conversations:

```python
agent = ComputerAgent(
    model="anthropic/claude-3-5-sonnet-20241022",
    trajectory_dir="trajectories",  # Auto-save trajectories
    tools=[computer]
)

# Trajectories are saved with:
# - Complete conversation history
# - Usage statistics and costs
# - Timestamps and metadata
# - Screenshots and computer actions
```

## Configuration Options

### ComputerAgent Parameters

- `model`: Model identifier (required)
- `tools`: List of computer objects and decorated functions
- `callbacks`: List of callback handlers for extensibility
- `only_n_most_recent_images`: Limit recent images to prevent context overflow
- `verbosity`: Logging level (logging.INFO, logging.DEBUG, etc.)
- `trajectory_dir`: Directory to save conversation trajectories
- `max_retries`: Maximum API call retries (default: 3)
- `screenshot_delay`: Delay between actions and screenshots (default: 0.5s)
- `use_prompt_caching`: Enable prompt caching for supported models
- `max_trajectory_budget`: Budget limit configuration

### Environment Variables

```bash
# Computer instance (cloud)
export CUA_CONTAINER_NAME="your-container-name"
export CUA_API_KEY="your-cua-api-key"

# LLM API keys
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

## Advanced Usage

### Streaming Responses

```python
async for result in agent.run(messages, stream=True):
    # Process streaming chunks
    for item in result["output"]:
        if item["type"] == "message":
            print(item["content"][0]["text"], end="", flush=True)
        elif item["type"] == "computer_call":
            action = item["action"]
            print(f"\n[Action: {action['type']}]")
```

### Interactive Chat Loop

```python
history = []
while True:
    user_input = input("> ")
    if user_input.lower() in ['quit', 'exit']:
        break
        
    history.append({"role": "user", "content": user_input})
    
    async for result in agent.run(history):
        history += result["output"]
        
        # Display assistant responses
        for item in result["output"]:
            if item["type"] == "message":
                print(item["content"][0]["text"])
```

### Error Handling

```python
try:
    async for result in agent.run(messages):
        # Process results
        pass
except BudgetExceededException:
    print("Budget limit exceeded")
except Exception as e:
    print(f"Agent error: {e}")
```

## API Reference

### ComputerAgent.run()

```python
async def run(
    self,
    messages: Messages,
    stream: bool = False,
    **kwargs
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run the agent with the given messages.
    
    Args:
        messages: List of message dictionaries
        stream: Whether to stream the response
        **kwargs: Additional arguments
        
    Returns:
        AsyncGenerator that yields response chunks
    """
```

### Message Format

```python
messages = [
    {
        "role": "user",
        "content": "Take a screenshot and describe what you see"
    },
    {
        "role": "assistant", 
        "content": "I'll take a screenshot for you."
    }
]
```

### Response Format

```python
{
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I can see..."}]
        },
        {
            "type": "computer_call",
            "action": {"type": "screenshot"},
            "call_id": "call_123"
        },
        {
            "type": "computer_call_output",
            "call_id": "call_123",
            "output": {"image_url": "data:image/png;base64,..."}
        }
    ],
    "usage": {
        "prompt_tokens": 150,
        "completion_tokens": 75,
        "total_tokens": 225,
        "response_cost": 0.01,
    }
}
```

## License

MIT License - see LICENSE file for details.