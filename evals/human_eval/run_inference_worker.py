from evals.human_eval.inference_worker import InferenceWorker

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class CallableClass:
    def __call__(self, value):
        start_marker = "```python"
        end_marker = "```"
        start = value.index(start_marker) + len(start_marker)
        try:
            end = value.index(end_marker, start)
        except ValueError:
            return value[start:]
        return value[start:end]


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Execute LLM inference and extract result")
    parser.add_argument("--working-dir", type=str, default="/data", help="Artifacts (dataset and results) base working directory")
    parser.add_argument("--endpoint-url", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", type=str, default="/data/HumanEval.json")
    parser.add_argument("--api-key", type=str, default="not-needed")
    args = parser.parse_args()

    logger.info(f"using dataset: {args.dataset}")
    system_message = """You are a python code assistant, be concise with your code creation, don't add any preamble, usage, test code or reasoning messages,
    just generate the code, ensure the code generation starts with ```python and ends with ```"""

    results_dir = Path(args.working_dir + "/results")
    samples_dir = Path(args.working_dir + "/samples")
    results_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        instances = json.load(f)

    format_output = CallableClass()

    impl = InferenceWorker([args.endpoint_url + "/v1"], args.model, system_message=system_message, api_key=args.api_key)
    results = impl.generate_batch(instances, None, format_output, instance_id_key="task_id", prompt_key="prompt")

    name = args.model.split("/")
    if len(name) > 1:
        sample_file = str(samples_dir) + "/" + name[1].lower().replace(".", "-") + ".json"
    else:
        sample_file = str(samples_dir) + "/" + name[0].lower().replace(".", "-") + ".json"

    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(results, f)
    
    logger.info(f"wrote sample file to {sample_file}")


if __name__ == '__main__':
    main()
