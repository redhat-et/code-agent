"""SWE-bench evaluation pipeline.

Two-phase evaluation using Ray for orchestration and native K8s Jobs
for test execution.

Phase 1 (run_patch_generation.py):
    Generate patches via vLLM, run AST verification.

Phase 2 (run_test_execution.py):
    Run tests in pre-built SWE-bench container images via K8s Jobs,
    grade results.
"""
