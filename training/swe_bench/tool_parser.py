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

    Returns a ToolCall with type="submit" if a <submit/> tag is found,
    type="bash" with the command content if a <bash>...</bash> block is
    found, or type="bash" with the full text as a fallback.
    """
    if _SUBMIT_PATTERN.search(text):
        return ToolCall(type="submit")

    match = _BASH_PATTERN.search(text)
    if match:
        return ToolCall(type="bash", content=match.group(1).strip())

    stripped = text.strip()
    if stripped:
        return ToolCall(type="bash", content=stripped)

    return ToolCall(type="submit")
