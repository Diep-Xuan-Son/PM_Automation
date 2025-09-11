"""HUD integration: Generic HuggingFace dataset evaluation runner (CUA proxy).

This module exposes two helpers to evaluate HUD-compatible datasets using
HUD's OperatorAgent, while proxying model calls through our ComputerAgent via
`FakeAsyncOpenAI` (see `agent/integrations/hud/agent.py`).

Exports:
- run_single_task(dataset_name, *, agent_type="cua-proxy", model=None, allowed_tools=None)
- run_full_dataset(dataset_name, *, agent_type="cua-proxy", model=None, allowed_tools=None, max_concurrent=30, max_steps=50)
"""
import time
from typing import Any, Optional

from PIL import Image
from datasets import load_dataset, Dataset
from hud.agents import OperatorAgent
from hud.datasets import Task, run_dataset
from hud.tools.computer.settings import computer_settings
from hud import trace

from agent.agent import ComputerAgent as BaseComputerAgent
from .proxy import FakeAsyncOpenAI


# ---------------------------------------------------------------------------
# Proxy OperatorAgent
# ---------------------------------------------------------------------------


class ProxyOperatorAgent(OperatorAgent):
    """OperatorAgent that proxies model calls through our ComputerAgent.

    Accepts the same config keys we pass via hud.run_dataset `agent_config`:
    - model: str | None
    - allowed_tools: list[str] | None
    Additional kwargs are forwarded to OperatorAgent (if any are supported).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        trajectory_dir: str | dict | None = None,
        # === ComputerAgent kwargs ===
        tools: list[Any] | None = None,
        custom_loop: Any | None = None,
        only_n_most_recent_images: int | None = None,
        callbacks: list[Any] | None = None,
        verbosity: int | None = None,
        max_retries: int | None = 3,
        screenshot_delay: float | int = 0.5,
        use_prompt_caching: bool | None = False,
        max_trajectory_budget: float | dict | None = None,
        telemetry_enabled: bool | None = True,
        **kwargs: Any,
    ) -> None:
        model = model or "computer-use-preview"
        allowed_tools = allowed_tools or ["openai_computer"]
        
        computer_shim = {
            'screenshot': lambda: Image.new('RGB', (computer_settings.OPENAI_COMPUTER_WIDTH, computer_settings.OPENAI_COMPUTER_HEIGHT)),
            'environment': 'linux',
            'dimensions': (computer_settings.OPENAI_COMPUTER_WIDTH, computer_settings.OPENAI_COMPUTER_HEIGHT)
        }
        # Build tools ensuring the computer_shim is included
        agent_tools: list[Any] = [computer_shim]
        if tools:
            agent_tools.extend(tools)

        computer_agent = BaseComputerAgent(
            model=model,
            tools=agent_tools,
            custom_loop=custom_loop,
            only_n_most_recent_images=only_n_most_recent_images,
            callbacks=callbacks,
            verbosity=verbosity,
            trajectory_dir=trajectory_dir,
            max_retries=max_retries,
            screenshot_delay=screenshot_delay,
            use_prompt_caching=use_prompt_caching,
            max_trajectory_budget=max_trajectory_budget,
            telemetry_enabled=telemetry_enabled,
        )
        model_client = FakeAsyncOpenAI(computer_agent)

        super().__init__( 
            model_client=model_client, # type: ignore[arg-type]
            model=model,
            allowed_tools=allowed_tools,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------


async def run_single_task(
    dataset: str | Dataset | list[dict[str, Any]],
    *,
    task_id: int = 0,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    # === ComputerAgent kwargs ===
    tools: list[Any] | None = None,
    custom_loop: Any | None = None,
    only_n_most_recent_images: int | None = None,
    callbacks: list[Any] | None = None,
    verbosity: int | None = None,
    trajectory_dir: str | dict | None = None,
    max_retries: int | None = 3,
    screenshot_delay: float | int = 0.5,
    use_prompt_caching: bool | None = False,
    max_trajectory_budget: float | dict | None = None,
    telemetry_enabled: bool | None = True,
) -> None:
    """Load one task from the dataset and execute it with Operator+CUA proxy."""

    # Load dataset and pick a sample
    if isinstance(dataset, str):
        dataset = load_dataset(dataset, split="train") # type: ignore[arg-type]
    elif isinstance(dataset, list):
        dataset = dataset
    else:
        dataset = dataset["train"]
    
    sample_task = dataset[task_id]  # type: ignore[index]
    task_prompt = sample_task.get("prompt", f"Task {sample_task.get('id', 0)}")  # type: ignore[attr-defined]

    with trace(name=task_prompt):
        task = Task(**sample_task)  # type: ignore[arg-type]

        agent = ProxyOperatorAgent(
            model=model,
            allowed_tools=allowed_tools,
            # === ComputerAgent kwargs passthrough ===
            tools=tools,
            custom_loop=custom_loop,
            only_n_most_recent_images=only_n_most_recent_images,
            callbacks=callbacks,
            verbosity=verbosity,
            trajectory_dir=trajectory_dir,
            max_retries=max_retries,
            screenshot_delay=screenshot_delay,
            use_prompt_caching=use_prompt_caching,
            max_trajectory_budget=max_trajectory_budget,
            telemetry_enabled=telemetry_enabled,
        )
        print(f"Running: {task_prompt}")
        result = await agent.run(task, max_steps=10)
        print(f"✅ Reward: {getattr(result, 'reward')}")


# ---------------------------------------------------------------------------
# Full-dataset runner
# ---------------------------------------------------------------------------


async def run_full_dataset(
    dataset: str | Dataset | list[dict[str, Any]],
    *,
    job_name: Optional[str] = None,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_concurrent: int = 30,
    max_steps: int = 50,
    split: str = "train",
    trajectory_dir: str | dict | None = None,
    # === ComputerAgent kwargs ===
    tools: list[Any] | None = None,
    custom_loop: Any | None = None,
    only_n_most_recent_images: int | None = 5,
    callbacks: list[Any] | None = None,
    verbosity: int | None = None,
    max_retries: int | None = 3,
    screenshot_delay: float | int = 0.5,
    use_prompt_caching: bool | None = False,
    max_trajectory_budget: float | dict | None = None,
    telemetry_enabled: bool | None = True,
) -> list[Any]:
    """Run evaluation across the entire dataset using hud.datasets.run_dataset."""

    # We pass OperatorAgent as the class and provide a config that injects our
    # FakeAsyncOpenAI per agent instantiation.

    if isinstance(dataset, str):
        dataset_name = dataset.split('/')[-1]
        job_name = job_name or f"Evaluation {dataset_name}"
        dataset = load_dataset(dataset, split=split) # type: ignore[arg-type]
    else:
        dataset_name = "custom"
        job_name = job_name or f"Evaluation {time.strftime('%H:%M %Y-%m-%d')}"

    # Execute evaluation
    return await run_dataset(
        name=job_name,
        dataset=dataset,
        agent_class=ProxyOperatorAgent,
        agent_config={
            "model": model,
            "allowed_tools": allowed_tools,
            "trajectory_dir": trajectory_dir,
            # === ComputerAgent kwargs passthrough ===
            "tools": tools,
            "custom_loop": custom_loop,
            "only_n_most_recent_images": only_n_most_recent_images,
            "callbacks": callbacks,
            "verbosity": verbosity,
            "max_retries": max_retries,
            "screenshot_delay": screenshot_delay,
            "use_prompt_caching": use_prompt_caching,
            "max_trajectory_budget": max_trajectory_budget,
            "telemetry_enabled": telemetry_enabled,
        },
        max_concurrent=max_concurrent,
        metadata={"dataset": dataset_name},
        max_steps=max_steps,
        auto_respond=True,
    )


__all__ = [
    "run_single_task",
    "run_full_dataset",
    "ProxyOperatorAgent",
]