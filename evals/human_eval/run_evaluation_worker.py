from evals.human_eval.evaluation import evaluate_functional_correctness
import argparse
from pathlib import Path
import json
import logging
import os 
import sys


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Execute tests and evaluate results")
    parser.add_argument("--working-dir", type=str, default="/data", help="Artifacts (dataset and results) base working directory")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Ensure model name is the same for both inference and evaluation")
    parser.add_argument("--dataset-file", type=str, default="HumanEval.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    results_dir = Path(args.working_dir + "/generated/results")
    samples_dir = Path(args.working_dir + "/generated/samples")
    results_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"working dir: {args.working_dir}")
    logger.info(f"samples dir: {samples_dir}")
    logger.info(f"results dir: {results_dir}")
    logger.info(f"using model: {args.model}")
    logger.info(f"using problem-file: {args.dataset_file}")

    name = args.model.split("/")
    if len(name) > 1:
        sample_name = name[1].lower().replace(".","-")
    else:
        sample_name = name[0].lower().replace(".","-")

    sample_file = samples_dir / f"{sample_name}.json"
    if not sample_file.exists():
        logger.error(
            f"sample file {sample_file} not found, have you executed 'run_inference_worker' ?"
    )
    sys.exit(1)

    results = evaluate_functional_correctness(args.working_dir,sample_name, args.dataset_file, args.workers, args.timeout)
    logger.info(f"executed {results} samples")

if __name__ == "__main__":
    main()



