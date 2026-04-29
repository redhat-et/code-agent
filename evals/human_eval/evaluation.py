from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
from evals.common.grader import BaseAggregateReport
from evals.human_eval.execution import check_correctness
import json
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def evaluate_functional_correctness(
    working_dir: str,
    sample_name: str,
    dataset_file: str,
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes results to directory
    """
   
    sample_file = working_dir + "/samples/" + sample_name + ".json"
    with open(sample_file) as fs:
        sample_data = json.load(fs)

    problem_file = working_dir + "/" + dataset_file
    with open(problem_file) as fp:
        problem_data = json.load(fp)

    holder = {}
    for x in problem_data: 
        task_id = x["task_id"]
        holder[task_id] = x
    
    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        n_samples = 0
        results = []

        logger.info("reading samples...")
        for sample in sample_data:
            task_id = sample["instance_id"]
            completion = sample["prediction"]
            args = (holder[task_id], completion, timeout)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            n_samples += 1


        report = BaseAggregateReport()
        passed = 0
        failed = 0
        passed_ids = []
        failed_ids = []
        logger.info("running test suites...")
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result['passed']:
                passed += 1
                passed_ids.append(result['task_id'])
            else: 
                failed += 1
                passed_ids.append(result['task_id'])
            
            logger.debug(f"result {result['task_id']} {result['passed']}")
        
        report.passed = passed
        report.failed = failed
        report.total = n_samples
        report.passed_ids = passed_ids
        report.failed_ids = failed_ids

        report.finalize()

        # Write final report
        full_report = {
            "summary": report.to_dict(),
            "instance_results": results,
        }
        results_file = working_dir + "/results/results.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(full_report,f)
        logger.info(f"wrote full report to {results_file}")

    return n_samples


