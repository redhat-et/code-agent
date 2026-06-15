"""OpenRLHF AgentInstanceBase implementation for SWE-bench.

Connects OpenRLHF's multi-turn agent loop to the K8s Pod execution
environment.  Each instance manages one agent episode: reset() creates
a sandbox Pod, step() executes tool calls, and the final step runs
the SWE-bench eval to produce a binary reward.
"""

from __future__ import annotations

import json
import logging
import os

import torch
from openrlhf.utils.agent import AgentInstanceBase

try:
    from training.swe_bench.environment import SWEBenchEnvironment
    from training.swe_bench.system_prompt import build_system_prompt
    from training.swe_bench.tool_parser import parse_tool_call
except ImportError:
    from environment import SWEBenchEnvironment
    from system_prompt import build_system_prompt
    from tool_parser import parse_tool_call

logger = logging.getLogger(__name__)

_MAX_STEPS = int(os.environ.get("SWE_MAX_STEPS", "100"))
_MAX_OUTPUT_CHARS = int(os.environ.get("SWE_MAX_OUTPUT_CHARS", "8000"))
_IMAGE_REGISTRY = os.environ.get("SWE_IMAGE_REGISTRY", "")
_K8S_NAMESPACE = os.environ.get("SWE_K8S_NAMESPACE", "")
_SERVICE_ACCOUNT = os.environ.get("SWE_SERVICE_ACCOUNT", "swe-bench-training")


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... ({len(text) - limit} chars truncated) ...\n" + text[-half:]


def _format_tool_output(stdout: str, exit_code: int) -> str:
    truncated = _truncate(stdout)
    parts = [truncated]
    if exit_code != 0:
        parts.append(f"[exit code: {exit_code}]")
    return "\n".join(parts)


class SWEBenchAgentInstance(AgentInstanceBase):
    """One agent episode on a SWE-bench instance."""

    def __init__(self):
        self.env = SWEBenchEnvironment(
            image_registry=_IMAGE_REGISTRY,
            namespace=_K8S_NAMESPACE or None,
            service_account=_SERVICE_ACCOUNT,
        )
        self.step_count = 0
        self.max_steps = _MAX_STEPS
        self.instance_id = ""

    async def reset(self, states: dict, **kwargs) -> dict:
        """Create the sandbox Pod and return the system prompt.

        OpenRLHF passes:
          states["observation"] = problem_statement (from dataset ``input`` field)
          states["label"] = instance metadata JSON (from dataset ``label`` field)
        """
        problem_statement = states["observation"]
        label = states.get("label", "{}")
        instance = json.loads(label) if isinstance(label, str) else label

        self.instance_id = instance.get("instance_id", "unknown")
        self.step_count = 0

        await self.env.create(instance)

        prompt = build_system_prompt(problem_statement)
        return {"observation": prompt}

    async def step(self, state_dict: dict, **kwargs) -> dict:
        """Execute one tool call and return feedback + reward.

        OpenRLHF passes:
          state_dict["action_text"]      = model's latest generation
          state_dict["observation_text"] = full conversation so far
          state_dict["label"]            = ground truth label
        """
        action_text = state_dict.get("action_text", "")
        tool_call = parse_tool_call(action_text)

        self.step_count += 1
        force_submit = self.step_count >= self.max_steps

        if tool_call.type == "submit" or force_submit:
            return await self._handle_submit()

        return await self._handle_bash(tool_call.content)

    async def _handle_bash(self, command: str) -> dict:
        try:
            stdout, exit_code = await self.env.execute(command)
        except Exception as e:
            stdout = f"Error executing command: {e}"
            exit_code = 1

        feedback = _format_tool_output(stdout, exit_code)

        return {
            "rewards": torch.tensor(0.0),
            "scores": torch.tensor(0.0),
            "environment_feedback": feedback,
            "done": False,
            "extra_logs": {
                "step": self.step_count,
                "exit_code": exit_code,
            },
        }

    async def _handle_submit(self) -> dict:
        try:
            resolved, eval_output = await self.env.run_eval()
        except Exception as e:
            logger.error(f"[{self.instance_id}] Eval failed: {e}")
            resolved = False
        finally:
            await self.env.destroy()

        reward = 1.0 if resolved else 0.0
        logger.info(
            f"[{self.instance_id}] Episode done: "
            f"{'RESOLVED' if resolved else 'NOT RESOLVED'} "
            f"in {self.step_count} steps"
        )

        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(reward),
            "environment_feedback": "Patch submitted and evaluated.",
            "done": True,
            "extra_logs": {
                "resolved": reward,
                "steps": self.step_count,
            },
        }
