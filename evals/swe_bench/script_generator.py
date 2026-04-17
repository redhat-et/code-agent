"""Generate the inline shell command embedded in each K8s Job spec.

The command writes the model patch via heredoc, applies it with
swebench's 3-method fallback, then writes and runs the eval script.

Adapted from https://github.com/MichaelClifford/swe-bench-on-kfp
"""

from swebench.harness.constants import DOCKER_PATCH, DOCKER_WORKDIR

# The three patch-apply strategies swebench uses, tried in order.
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

PATCH_HEREDOC = "EOF_MODEL_PATCH_7829361"
EVAL_HEREDOC = "EOF_EVAL_SCRIPT_4819253"


def job_eval_command(model_patch: str, eval_script: str) -> list[str]:
    """Return the shell command for a K8s Job to evaluate an instance.

    The patch and eval script are embedded directly in the command as
    heredocs, so no PVC or volume mount is needed to pass data in.
    Test output is captured via pod logs.

    Args:
        model_patch: The model's prediction patch (diff text).
        eval_script: The eval script content from TestSpec.eval_script.

    Returns:
        Command as a list of strings for a K8s Job container spec:
        ["/bin/bash", "-c", "<inline script>"].
    """
    apply_lines = []
    for i, cmd in enumerate(GIT_APPLY_CMDS):
        keyword = "if" if i == 0 else "elif"
        apply_lines.append(f'{keyword} {cmd} "{DOCKER_PATCH}"; then')
        apply_lines.append('    echo ">>>>> Applied Patch"')

    apply_lines.append("else")
    apply_lines.append('    echo ">>>>> Patch Apply Failed"')
    apply_lines.append("    exit 1")
    apply_lines.append("fi")

    script = "\n".join([
        f"export HOME=/tmp",
        f"git config --global --add safe.directory {DOCKER_WORKDIR}",
        f"cd {DOCKER_WORKDIR}",
        "",
        f"cat > {DOCKER_PATCH} << '{PATCH_HEREDOC}'",
        model_patch or "",
        PATCH_HEREDOC,
        "",
        "\n".join(apply_lines),
        "",
        f"cat > /tmp/eval.sh << '{EVAL_HEREDOC}'",
        eval_script,
        EVAL_HEREDOC,
        "",
        "/bin/bash /tmp/eval.sh",
    ])

    return ["/bin/bash", "-c", script]
