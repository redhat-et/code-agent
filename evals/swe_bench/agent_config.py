"""Agent configuration for SWE-bench evaluation.

Defines a YAML-driven agent config that specifies how to install and run
an agent inside a SWE-bench container image. This makes the agent swappable --
changing the YAML file switches from mini-swe-agent to OpenCode or any other agent.

Usage:
    config = load_agent_config("evals/swe_bench/agents/mini_swe_agent.yaml")
    rendered_cmd = render_template(config.agent_command, instance_id="django__django-11099", ...)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AgentConfig:
    """Configuration for an agent that runs inside SWE-bench containers.

    Attributes:
        name: Human-readable agent name.
        install_command: Shell command to install the agent (pip install, etc.).
        agent_command: Shell command template to run the agent.
            Supports placeholders like {instance_id}, {model_name}, etc.
        patch_extraction: How to extract the patch after the agent finishes.
            - "preds_json:<path>" -- read a preds.json file (mini-swe-agent native)
            - "git_diff" -- run git diff in /testbed
            - "file:<path>" -- read raw patch from a file
        needs_swebench_dataset: If True, the agent handles dataset loading
            itself (e.g., mini-extra swebench). If False, the problem
            statement is passed via env var / file.
        env: Extra environment variables (templates with placeholders).
        resources: K8s resource requests/limits.
        job_timeout: activeDeadlineSeconds for the K8s Job.
    """

    name: str
    install_command: str = ""
    agent_command: str = ""
    patch_extraction: str = "git_diff"
    needs_swebench_dataset: bool = False
    env: dict[str, str] = field(default_factory=dict)
    resources: dict = field(default_factory=lambda: {
        "requests": {"cpu": "2", "memory": "4Gi"},
        "limits": {"cpu": "2", "memory": "4Gi", "ephemeral-storage": "4Gi"},
    })
    job_timeout: int = 600


def load_agent_config(path: str | Path) -> AgentConfig:
    """Load an AgentConfig from a YAML file.

    If the path doesn't exist as given, tries to resolve it relative to
    the package's agents/ directory (e.g., "mini_swe_agent.yaml" or
    "evals/swe_bench/agents/mini_swe_agent.yaml" will both work).

    Args:
        path: Path to the agent config YAML.

    Returns:
        Populated AgentConfig instance.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        KeyError: If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        # Try resolving relative to the agents/ directory within this package
        agents_dir = Path(__file__).parent / "agents"
        candidates = [
            agents_dir / path.name,                          # just the filename
            agents_dir / path,                               # relative path as-is
            Path(__file__).parent.parent.parent / path,      # from package root
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Agent config not found: {path}\n"
                f"Searched: {[str(c) for c in candidates]}\n"
                f"Package dir: {Path(__file__).parent}"
            )

    with open(path) as f:
        data = yaml.safe_load(f)

    default_resources = {
        "requests": {"cpu": "2", "memory": "4Gi"},
        "limits": {"cpu": "2", "memory": "4Gi", "ephemeral-storage": "4Gi"},
    }

    return AgentConfig(
        name=data["name"],
        install_command=data.get("install_command", ""),
        agent_command=data.get("agent_command", ""),
        patch_extraction=data.get("patch_extraction", "git_diff"),
        needs_swebench_dataset=data.get("needs_swebench_dataset", False),
        env=data.get("env", {}),
        resources=data.get("resources", default_resources),
        job_timeout=data.get("job_timeout", 600),
    )


def render_template(template: str, **kwargs: str) -> str:
    """Render a template string by substituting {placeholder} values.

    Uses simple string replacement rather than f-strings or jinja2
    to avoid issues with shell syntax containing braces.

    Only replaces placeholders that are provided in kwargs.
    Unknown placeholders are left as-is.

    Args:
        template: Template string with {placeholder} syntax.
        **kwargs: Values to substitute.

    Returns:
        Rendered string.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result
