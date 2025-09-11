import os
import asyncio
import logging

from agent.agent import ComputerAgent
from computer import Computer
from computer.helpers import sandboxed

os.environ["OPENAI_API_KEY"] = ''
def calculate(a: int, b: int) -> int:
    """Calculate the sum of two integers
    
    Parameters
    ----------
    a : int
        First integer
    b : int
        Second integer
        
    Returns
    -------
    int
        Sum of the two integers
    """
    return a + b


agent = ComputerAgent(
    model="openai/gpt-4o-mini",
    tools=[calculate],
    only_n_most_recent_images=3,
    verbosity=logging.INFO,
    trajectory_dir="trajectories",
    use_prompt_caching=True,
    max_trajectory_budget={ "max_budget": 1.0, "raise_error": True, "reset_after_each_run": False },
)

history = [{"role": "user", "content": "tính tổng của 1 và 2"}]

async def main():
  async for result in agent.run(history, stream=False):
    print(f"----result2: {result}")

if __name__=="__main__":
  asyncio.run(main())
