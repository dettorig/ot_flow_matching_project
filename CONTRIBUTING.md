# Contributing

This is a 2-person academic project (ENSAE M2, Optimal Transport course).
The notes below describe the workflow we agreed on.

## Branches

- `main` is protected. No direct pushes, no force-push.
- Feature branches: `feat/<initials>-<short-topic>`, e.g.
  `feat/mt-rk4-solver`, `feat/gd-vp-schedule`.
- One PR per feature, reviewed by the other contributor before merge.

## Commit messages

Conventional Commits style:

```
<type>(<scope>): <short imperative summary>
```

Allowed types:

| Type       | Use for                                                |
| ---------- | ------------------------------------------------------ |
| `feat`     | New user-facing capability (solver, metric, dataset…)  |
| `refactor` | Code reorganisation with no behavior change            |
| `fix`      | Bug fix                                                |
| `test`     | Adding or improving tests                              |
| `docs`     | README, notebook markdown, docstrings                  |
| `chore`    | Tooling, configs, dependencies                         |
| `exp`      | Experiment notebook / run / sweep                      |
| `style`    | Formatting only (ruff/black, no logic)                 |

Examples:

- `refactor: extract samplers to otfm/datasets.py`
- `feat(solvers): add RK4 integrator`
- `exp: schedules x couplings full sweep`
- `docs: write methodology section in baseline notebook`

Keep summary line ≤ 70 characters. Body optional, wrapped at 72.

## Code style

- Python ≥ 3.10.
- Formatter: `black`. Linter: `ruff` (config in `pyproject.toml`).
- All public functions in `otfm/` get a one-line docstring stating
  shapes and dtypes of inputs/outputs.
- No JAX/NumPy dtype mixing in critical paths.

## Notebooks

- Each new experiment lives in its own notebook under `experiments/`.
- We use `nbstripout` to keep diffs readable: outputs are stripped on
  commit. Run `nbstripout --install` once after cloning.
- Long-running experiments dump CSVs to `results/` so figures can be
  regenerated without re-training.

## Tests

- `pytest -q` must pass on `main` at all times.
- Add a test for any non-trivial pure function in `otfm/`.
- Marginal checks for couplings, gradient sanity for losses, order of
  convergence for solvers.

## Pull requests

- Description states: what changed, why, and key results (figures or
  numbers if applicable).
- Reviewer checks: tests pass, no notebook outputs in diff, no `print`
  debug left over, message follows convention.
- Squash-merge by default to keep `main` history flat and atomic.

## Reproducibility

Seeds are centralised in `configs/base.yaml`. Any new experiment must
log the seed it used.
