# SWE-bench Evaluation Pipeline

Evaluate models on [SWE-bench](https://swe-bench.github.io/) using Ray for orchestration and native Kubernetes Jobs for test execution.

## Overview

The pipeline has three steps:

1. **Build prompts** (one-time) -- Clone repos and construct prompts using swebench's official pipeline. Upload to S3/MinIO.
2. **Phase 1: Patch generation** -- Ray workers call vLLM to generate patches from the pre-built prompts.
3. **Phase 2: Test execution** -- Ray workers create K8s Jobs using SWE-bench's pre-built container images, collect test output, and grade results.

## Prerequisites

- OpenShift / Kubernetes cluster with KubeRay operator
- MinIO deployed with a `swe-bench` bucket (`oc apply -f infra/deploy/minio.yaml`)
- vLLM model server deployed (`oc apply -f inference/deploy/vllm-server-deployment.yaml`)

## Step 0: Build Prompted Dataset (one-time)

Builds prompts using swebench's style-3 format with oracle file selection. Only needs to run once per dataset.

```bash
oc apply -f evals/swe_bench/deploy/job-build-prompts.yaml
oc logs -f job/build-swe-bench-prompts
```

The job uploads the prompted dataset to `s3://swe-bench/swe-bench-verified/prompts/style-3-oracle.jsonl`.

## Step 1: Generate Patches (Phase 1)

Deploy the Ray cluster and run inference:

```bash
oc apply -f evals/swe_bench/deploy/raycluster-patch-gen.yaml
oc port-forward svc/swe-bench-patch-gen-head-svc 8265:8265


MODEL_NAME="<MODEL_NAME>" RUN_ID=<RUN_ID> bash run_swe_bench_phase1.sh 
```

Predictions are uploaded to `s3://swe-bench/runs/{RUN_ID}/predictions.jsonl`.

## Step 2: Run Tests (Phase 2)

Deploy the RBAC, Ray cluster, and run test execution:

```bash
oc apply -f evals/swe_bench/deploy/rbac.yaml
oc apply -f evals/swe_bench/deploy/raycluster-test-exec.yaml
oc port-forward svc/swe-bench-test-exec-head-svc 8265:8265

RUN_ID=<RUN_ID> bash run_swe_bench_phase2.sh 
```

Results are uploaded to `s3://swe-bench/runs/{RUN_ID}/results.json`.

## Quick Test

Run on a small subset to validate the setup:

```bash
# Phase 1: generate 16 patches
INSTANCE_LIMIT=16 RUN_ID=test-16 bash run_swe_bench_phase1.sh

# Phase 2: run tests on those 16
INSTANCE_LIMIT=16 RUN_ID=test-16 bash run_swe_bench_phase2.sh
```

## Verify Harness with Gold Patches

Use the ground-truth patches from the dataset to confirm the eval harness works correctly (skips Phase 1):

```bash
PREDICTIONS=gold INSTANCE_LIMIT=16 RUN_ID=gold-test bash run_swe_bench_phase2.sh
```

If gold patches resolve, the harness is working. If they don't, there's a setup issue.

## Configuration

### Phase 1 (`run_swe_bench_phase1.sh`)

| Variable | Default | Description |
|---|---|---|
| `VLLM_URL` | `http://vllm-server:8000/v1` | vLLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen3-1.7B` | Model name in vLLM |
| `DATASET` | `SWE-bench/SWE-bench_Verified` | HuggingFace dataset |
| `PROMPTS` | `s3://swe-bench/swe-bench-verified/prompts/style-3-oracle.jsonl` | Prompted dataset |
| `NUM_WORKERS` | `2` | Ray workers |
| `MAX_TOKENS` | `4096` | Max tokens for generation |
| `TEMPERATURE` | `0.0` | Sampling temperature |
| `INSTANCE_LIMIT` | `0` (all) | Limit instances for testing |
| `RUN_ID` | `eval-run` | Unique run identifier |

### Phase 2 (`run_swe_bench_phase2.sh`)

| Variable | Default | Description |
|---|---|---|
| `PREDICTIONS` | `s3://.../{RUN_ID}/predictions.jsonl` | Predictions from Phase 1, or `gold` |
| `NUM_WORKERS` | `4` | Ray workers |
| `MAX_CONCURRENT_JOBS` | `4` | K8s Jobs per worker |
| `TIMEOUT` | `1800` | Per-instance timeout (seconds) |
| `INSTANCE_LIMIT` | `0` (all) | Limit instances for testing |
| `RUN_ID` | `eval-run` | Must match Phase 1 |

## Architecture

```
Step 0: Build Prompts (one-time K8s Job)
  clone repos -> read source files -> build style-3 prompts -> upload to S3

Phase 1: Patch Generation (Ray cluster)
  download prompts from S3 -> distribute to Ray workers -> call vLLM -> upload predictions to S3

Phase 2: Test Execution (Ray cluster)
  download predictions from S3 -> distribute to Ray workers ->
  each worker creates K8s Jobs using SWE-bench pre-built images ->
  collect pod logs -> grade with swebench -> upload results to S3
```

No nested containers -- SWE-bench test images run as native K8s Jobs.
