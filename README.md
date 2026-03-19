### Project Description

The generic problem being attempted is:
- Use an LLM to generate code.
- Run the generated code in live execution environments to compute:
  - Multiple reward signals from verifiers - linters, type-checkers, AST-based syntax checking, unit tests, performance tests.
  - Get additional context - error messages, traces, perf counters.
- Use RL to update model.

### Example Applications

- Generate Helion GPU kernels where tracing is done via nsys + ncu and reward is latency or throughput.
- CVE fixes
- Bug fixes
- More verifiers can be added e.g. given arxiv paper, generate implementation code and verification is benchmark performance vs what is quoted in paper (soft-constraint because of noise).


### Notes on development process

* Generate code with Claude (in the browser and not Claude Code)
* Sketch architecture (classes, interfaces) manually
* Compare manual design vs Claude's design and understand differences.
* Refactor and modify Claude's code

For simple scripts, check code carefully before committing. We will not be generating huge amounts of code that get committed without review.

We are making an explicit choice (for this project) to not let Claude Code edit the code.

### RL Loop

There are two obvious choices. We will work in the context of SWE-bench for now i.e. the task is to generate patches to fix bugs.

Single-shot:

```python
for i in range(num_episodes):
    for k in range(batch_size):
        generate patch for task k
        run verification hierarchy
        retrieve reward signals and additional context (traces)
    update model (REINFORCE, PPO, GRPO etc.)
```

In the formulation above, the model generates the patch and the reward signal is used to update the model. The additional context isn't used. This makes it harded to do credit-assignment across the token-level trajectory.

One can think of the single-shot as RL with sparse trajectory-level rewards. This is also called outcome-based reward modeling (ORM).

Multi-turn:

```python
for i in range(num_episodes):
    for k in range(batch_size):
        for b in range(budget_steps):
            generate patch
            run verification hierarchy
            retrieve reward signals and additional context
            append additional context to existing context
    update model (REINFORCE, PPO, GRPO etc.)
```

In this formulation, there's an additional innermost loop that runs for budget_steps iterations. In each step, a generated patch runs through the verification hiearchy and additional context (traces, perf counters, linter errors) is appended as additional context for the next iteration. The list of budget_step rewards and gradients of log probabilities are used to update the model before moving to the next episode.

The multi-turn setup is equivalent to RL with dense per-timestep rewards. This setup is closer to process-based reward modeling (PRM).
