# otfm — OT couplings for Flow Matching

A controlled, reproducible study of how the choice of coupling
$\pi \in \Pi(\nu_0, \nu_1)$ affects training and rollout in
**Flow Matching** on 2D toy data (moons → 8-Gaussians).

ENSAE Paris — M2 Optimal Transport course project. The deliverable is
the annotated notebook(s) under `experiments/`.

---

## What is in this repo

```
otfm/                package: datasets, couplings, model, training,
                     solvers, runtime, metrics, plotting, sweeps
tests/               pytest sanity checks (marginals, grad, solver order)
configs/base.yaml    shared hyper-parameters for every experiment
experiments/         experiment notebooks (one per topic) — the deliverable
scripts/             optional non-interactive runners
firstrun.ipynb       legacy demo notebook (kept for reference)
```

## Couplings compared

1. `independent` — $\pi = \nu_0 \otimes \nu_1$.
2. `hungarian_exact_ot` — exact discrete OT (`linear_sum_assignment`).
3. `sinkhorn_sampled` — pairs sampled from a Sinkhorn plan
   ($\varepsilon$-regularised entropic OT).
4. `sinkhorn_barycentric` — deterministic pairs via the barycentric
   projection of the same Sinkhorn plan.
5. **Perturbed OT** — pairs from $(1-\alpha)\pi^\star + \alpha\pi_{\mathrm{ind}}$.

## Quickstart

```bash
git clone https://github.com/dettorig/ot_flow_matching_project.git
cd ot_flow_matching_project

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,notebook]"
```

## Running the experiments

Open the notebooks in `experiments/` top-to-bottom:

- `baseline_4_couplings.ipynb` — the four canonical couplings under
  identical training.
- `nfe_solver_comparison.ipynb` — Euler / Heun / RK4 at matched NFE.
- `multiseed_baseline.ipynb` — mean ± std across seeds.

For faster iteration, edit `configs/base.yaml` (smaller `n`,
fewer `train_steps`).

## Hardware

JAX is installed in CPU mode by default. For GPU/Metal, install the
matching JAX wheel manually following the
[official instructions](https://docs.jax.dev/en/latest/installation.html).

## Tests

```bash
pytest -q             # 22 tests, ~15 s on CPU
ruff check otfm tests # lint
```

CI runs both on every push and PR (`.github/workflows/test.yml`).

## Authors

- Giovanni Dettori
- Mael Tremouille

## License

MIT — see [LICENSE](LICENSE).
