"""OpenRLHF agent entry point for SWE-bench RL training.

Loaded by OpenRLHF via ``--agent_func_path agent_func.py``.
The framework looks for a class named ``AgentExecutor`` that inherits
from ``AgentExecutorBase``.
"""

from openrlhf.utils.agent import MultiTurnAgentExecutor

try:
    from training.swe_bench.agent_instance import SWEBenchAgentInstance
except ImportError:
    from agent_instance import SWEBenchAgentInstance


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(SWEBenchAgentInstance)
