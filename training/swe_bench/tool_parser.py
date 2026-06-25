"""Parse model output for tool calls.

Defines a simple tag-based protocol for the agent:
  <bash>command here</bash>     -- execute a bash command
  <submit/>                     -- submit the current patch

If no tags are found, the entire output is treated as a bash command
(graceful fallback for early training when the model hasn't learned
the format yet).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_BASH_PATTERN = re.compile(r"<bash>(.*?)</bash>", re.DOTALL)
_SUBMIT_PATTERN = re.compile(r"<submit\s*/>")


@dataclass
class ToolCall:
    type: Literal["bash", "submit"]
    content: str = ""


def parse_tool_call(text: str) -> ToolCall:
    """Extract the first tool call from model-generated text.

    When both ``<bash>`` and ``<submit/>`` are present, whichever
    appears first wins.  Falls back to treating the full text as a
    bash command if no tags are found.
    """
    bash_match = _BASH_PATTERN.search(text)
    submit_match = _SUBMIT_PATTERN.search(text)

    if bash_match and submit_match:
        if bash_match.start() < submit_match.start():
            return ToolCall(type="bash", content=bash_match.group(1).strip())
        return ToolCall(type="submit")

    if submit_match:
        return ToolCall(type="submit")

    if bash_match:
        return ToolCall(type="bash", content=bash_match.group(1).strip())

    stripped = text.strip()
    if stripped:
        return ToolCall(type="bash", content=stripped)

    return ToolCall(type="submit")
