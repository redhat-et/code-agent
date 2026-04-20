"""Generate a deduplicated list of SWE-bench container images to mirror.

Loads the SWE-bench dataset, calls make_test_spec() for each instance
to derive the exact DockerHub image reference, deduplicates, and writes
a text file with one image per line.  This list is consumed by the
mirror K8s Job (job-mirror-images.yaml) which uses skopeo to copy
images into the cluster's internal registry.

Usage (local):
    python -m evals.swe_bench.mirror_images \
        --dataset princeton-nlp/SWE-bench_Verified \
        --output image-list.txt

    # Or with a limit for testing:
    python -m evals.swe_bench.mirror_images \
        --dataset princeton-nlp/SWE-bench_Lite \
        --limit 50 \
        --output image-list.txt
"""

from __future__ import annotations

import argparse
import logging
import sys

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_unique_images(
    dataset: str,
    split: str = "test",
    namespace: str = "swebench",
    limit: int = 0,
) -> list[str]:
    """Return a sorted, deduplicated list of image references.

    Args:
        dataset: HuggingFace dataset name (e.g. princeton-nlp/SWE-bench_Verified).
        split: Dataset split.
        namespace: DockerHub namespace used by swebench (default "swebench").
        limit: Max instances to process (0 = all).

    Returns:
        Sorted list of unique image references.
    """
    instances = load_swebench_dataset(dataset, split=split)
    if limit > 0:
        instances = instances[:limit]

    logger.info(f"Processing {len(instances)} instances from {dataset}")

    images: set[str] = set()
    failed_ids: list[str] = []
    for inst in instances:
        instance_id = inst.get("instance_id", "?")
        try:
            test_spec = make_test_spec(inst, namespace=namespace)
            images.add(test_spec.instance_image_key)
        except Exception as e:
            logger.warning(f"Failed to build TestSpec for {instance_id}: {e}")
            failed_ids.append(instance_id)

    unique = sorted(images)
    logger.info(
        f"{len(instances)} instances -> {len(unique)} unique images "
        f"({len(failed_ids)} errors)"
    )

    if failed_ids:
        logger.warning(
            f"WARNING: {len(failed_ids)} instances failed TestSpec generation: "
            f"{', '.join(failed_ids)}"
        )
        logger.warning(
            "Images for these instances will NOT be mirrored. "
            "Re-run after fixing the underlying issue."
        )

    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Generate deduplicated SWE-bench image list for mirroring"
    )
    parser.add_argument(
        "--dataset", type=str, default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--namespace", type=str, default="swebench",
        help="DockerHub namespace for pre-built images",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max instances to process (0 = all)",
    )
    parser.add_argument(
        "--output", type=str, default="image-list.txt",
        help="Output file path (one image per line)",
    )
    args = parser.parse_args()

    images = get_unique_images(
        dataset=args.dataset,
        split=args.split,
        namespace=args.namespace,
        limit=args.limit,
    )

    if not images:
        logger.error("No images found")
        sys.exit(1)

    with open(args.output, "w") as f:
        for img in images:
            f.write(img + "\n")

    logger.info(f"Wrote {len(images)} images to {args.output}")

    # Print summary of repos
    repos: dict[str, int] = {}
    for img in images:
        # image format: swebench/sweb.eval.x86_64.<repo>_1776_<name>-<id>:latest
        # extract the repo portion for summary
        parts = img.split("/", 1)
        if len(parts) == 2:
            repo_tag = parts[1]
        else:
            repo_tag = parts[0]
        # Group by everything before the instance-specific suffix
        repo_key = repo_tag.rsplit(":", 1)[0]  # strip tag
        # The repo is the part after sweb.eval.x86_64.
        prefix = "sweb.eval.x86_64."
        if prefix in repo_key:
            repo_key = repo_key[repo_key.index(prefix) + len(prefix):]
            # Repo name is up to the first hyphen-followed-by-digits pattern
            # but for summary, just use the full key
        repos[repo_key] = repos.get(repo_key, 0) + 1

    logger.info("Images per repo:")
    for repo, count in sorted(repos.items()):
        logger.info(f"  {repo}: {count}")


if __name__ == "__main__":
    main()
