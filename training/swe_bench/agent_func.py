"""OpenRLHF agent entry point for SWE-bench RL training.

Loaded by OpenRLHF via ``--agent_func_path agent_func.py``.
The framework looks for a class named ``AgentExecutor`` that inherits
from ``AgentExecutorBase``.
"""

import logging
import os

from openrlhf.utils.agent import MultiTurnAgentExecutor

try:
    from training.swe_bench.agent_instance import InfrastructureError, SWEBenchAgentInstance
except ImportError:
    from agent_instance import InfrastructureError, SWEBenchAgentInstance

logger = logging.getLogger(__name__)

_MAX_ROLLOUT_RETRIES = int(os.environ.get("SWE_MAX_ROLLOUT_RETRIES", "2"))


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(SWEBenchAgentInstance)

    async def execute(self, *args, **kwargs):
        last_error = None
        for attempt in range(_MAX_ROLLOUT_RETRIES + 1):
            try:
                return await super().execute(*args, **kwargs)
            except InfrastructureError as e:
                last_error = e
                if attempt < _MAX_ROLLOUT_RETRIES:
                    logger.warning(
                        f"Infrastructure error on attempt {attempt + 1}/{_MAX_ROLLOUT_RETRIES + 1}, "
                        f"retrying with fresh rollout: {e}"
                    )

        raise InfrastructureError(
            f"Rollout failed after {_MAX_ROLLOUT_RETRIES + 1} attempts"
        ) from last_error
