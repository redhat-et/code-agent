# SWE-bench Evaluation Pipeline

Evaluate models on [SWE-bench](https://swe-bench.github.io/) using Ray for orchestration and native Kubernetes Jobs for test execution.

## Overview

The pipeline has two steps:

1. **Build prompts** (one-time) — Clone repos and construct prompts using swebench's official pipeline. Upload to S3/MinIO.
2. **Evaluate** — A single unified Ray job handles patch generation and test execution. Two strategies are supported, and both work in single-shot or multi-turn mode:
   - `naive`: vLLM inference from pre-built prompts, followed by K8s test execution
   - `agent`: agentic loop (mini-swe-agent or custom) running inside K8s Jobs

**Multi-turn** (`MAX_TURNS > 1`) is orthogonal to strategy. With either strategy the model receives feedback from configurable intermediate verifiers after each attempt and can revise its patch before the final K8s test evaluation.

## Prerequisites

- OpenShift / Kubernetes cluster with KubeRay operator
- MinIO deployed with a `swe-bench` bucket (`oc apply -f infra/deploy/minio.yaml`)
- vLLM model server deployed (`oc apply -f inference/deploy/vllm-server-deployment.yaml`)
- RBAC for K8s Job creation from Ray worker pods (`oc apply -f evals/swe_bench/deploy/rbac.yaml`)

## Step 0: Build Prompted Dataset (one-time, naive strategy only)

Builds prompts using swebench's style-3 format with oracle file selection. Only needs to run once per dataset.

```bash
oc apply -f evals/swe_bench/deploy/job-build-prompts.yaml
oc logs -f job/build-swe-bench-prompts
```

Outputs to: `s3://swe-bench/swe-bench-verified/prompts/style-3-oracle.jsonl`

## Step 1: Run Evaluation

Deploy the Ray cluster, apply RBAC, and run:

```bash
oc apply -f evals/swe_bench/deploy/rbac.yaml
oc apply -f evals/swe_bench/deploy/raycluster-patch-gen.yaml
oc port-forward svc/swe-bench-patch-gen-head-svc 8265:8265

# Naive strategy, single-shot (default)
MODEL_NAME="Qwen/Qwen3-1.7B" RUN_ID=my-run bash run_swe_bench_eval.sh

# Naive strategy, multi-turn (3 attempts, AST check between turns)
MAX_TURNS=3 INTERMEDIATE_VERIFIERS="ast_check" RUN_ID=my-run bash run_swe_bench_eval.sh

# Agent strategy, single-shot
STRATEGY=agent RUN_ID=my-run bash run_swe_bench_eval.sh

# Agent strategy, multi-turn
STRATEGY=agent MAX_TURNS=3 INTERMEDIATE_VERIFIERS="ast_check" RUN_ID=my-run bash run_swe_bench_eval.sh
```

Results are uploaded to `s3://swe-bench/runs/{RUN_ID}/results.json`.

## Quick Test

Run on a small subset to validate the setup:

```bash
INSTANCE_LIMIT=16 RUN_ID=test-16 bash run_swe_bench_eval.sh
```

## Configuration

### `run_swe_bench_eval.sh` — common variables

| Variable | Default | Description |
|---|---|---|
| `STRATEGY` | `naive` | `naive` \| `agent` |
| `VLLM_URL` | `http://vllm-server:8000/v1` | vLLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen3-1.7B` | Model name in vLLM |
| `DATASET` | `SWE-bench/SWE-bench_Verified` | HuggingFace dataset |
| `NUM_WORKERS` | `2` | Ray workers |
| `INSTANCE_LIMIT` | `0` (all) | Limit instances for testing |
| `RUN_ID` | `eval-run` | Unique run identifier |
| `MAX_CONCURRENT_JOBS` | `4` | K8s Jobs per Ray worker |
| `JOB_TIMEOUT` | `1800` | Per-instance K8s Job timeout (seconds) |

### Multi-turn variables (apply to both strategies)

| Variable | Default | Description |
|---|---|---|
| `MAX_TURNS` | `1` | Max generation attempts (1 = single-shot) |
| `INTERMEDIATE_VERIFIERS` | _(none)_ | Space-separated verifiers to run between turns (e.g. `ast_check`) |
| `AGGREGATOR` | `mean` | Score aggregation: `mean` \| `min` \| `weighted_sum` |

### Naive strategy variables

| Variable | Default | Description |
|---|---|---|
| `PROMPTS` | `s3://swe-bench/verified/prompts/style-3-oracle.jsonl` | Prompted dataset |
| `MAX_TOKENS` | `16000` | Max tokens per generation call |
| `TEMPERATURE` | `0.15` | Sampling temperature |

### Agent strategy variables

| Variable | Default | Description |
|---|---|---|
| `AGENT_CONFIG` | `evals/swe_bench/agents/mini_swe_agent.yaml` | Agent config YAML |
| `STEP_LIMIT` | `150` | Max agent steps per instance |
| `COST_LIMIT` | `3.0` | Max cost in dollars per instance |

## Architecture

```
Step 0: Build Prompts (one-time K8s Job, naive strategy only)
  clone repos → read source files → build style-3 prompts → upload to S3

Evaluation (single Ray cluster):

  naive, single-shot (MAX_TURNS=1):
    download prompts → distribute to workers → call vLLM →
    create K8s test Jobs → collect pod logs → grade → upload results

  naive, multi-turn (MAX_TURNS>1):
    for each turn (up to MAX_TURNS):
      call vLLM → extract patch →
      run intermediate verifiers:
        static (e.g. ast_check) → inline in worker process
        dynamic (e.g. swe_test) → K8s Job
      if all pass → stop early
      otherwise → format feedback → append to conversation
    run final K8s test Job → grade → upload results

  agent, single-shot (MAX_TURNS=1):
    distribute to workers → each worker creates K8s Jobs running
    the agent loop → collect results → grade → upload results

  agent, multi-turn (MAX_TURNS>1):
    for each turn (up to MAX_TURNS):
      run agent K8s Job (no in-container eval) → extract patch →
      run intermediate verifiers → format feedback →
      inject prior patches + feedback as context for next turn
    run final K8s test Job → grade → upload results
```

No nested containers — SWE-bench test images run as native K8s Jobs.

## Multi-turn Design Notes

- **Final patch**: always the last generated attempt (not the best-scoring one). Per-turn scores are recorded in `results.json` for analysis.
- **Early exit**: if all intermediate verifiers pass their `pass_threshold`, the loop stops and proceeds directly to final evaluation.
- **Static vs. dynamic verifiers**: static verifiers (e.g. `ast_check`) work from the patch diff alone and run inline with no K8s overhead. Dynamic verifiers (e.g. `swe_test`) require a K8s Job. Configuring only static verifiers as intermediate checks keeps turn latency low.
- **`MAX_TURNS=1`** is single-shot behavior — no intermediate evaluation, one generation followed by the final K8s test.
