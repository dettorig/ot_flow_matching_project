# otfm — OT couplings for Flow Matching

A controlled, reproducible study of how the choice of coupling
$\pi \in \Pi(\nu_0, \nu_1)$ affects training and rollout in
**Flow Matching** on 2D toy data (moons → 8-Gaussians).

ENSAE Paris — M2 Optimal Transport course project.

---

## What is in this repo

```
otfm/                package: datasets, couplings, model, training,
                     solvers, runtime, metrics, plotting, sweeps
tests/               pytest sanity checks (marginals, grad, solver order)
configs/base.yaml    shared hyper-parameters for every experiment
experiments/         experiment notebooks (one per topic)
scripts/             non-interactive runners that dump CSVs
results/             versioned CSVs the report figures are derived from
report/              LaTeX source of the report (+ tables + biblio)
slides/              Beamer slides
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

# Optional: keep notebook diffs free of outputs
nbstripout --install
```

## Reproducing the report

```bash
# 1. Sanity tests (a few seconds)
pytest -q

# 2. Run the full benchmark (≈ 5 min on CPU with default config)
python scripts/run_full_benchmark.py \
    --config configs/base.yaml \
    --output results/full_v1.csv

# 3. Generate the report figures
jupyter nbconvert --to notebook --execute experiments/final_figures.ipynb \
    --output /tmp/final_figures.run.ipynb

# 4. Build the PDF
cd report && latexmk -pdf main.tex
```

For faster iteration during development, override the config from the
command line:

```bash
python scripts/run_full_benchmark.py --n 256 --train-steps 500 --seeds 0 1
```

## Hardware

JAX is installed in CPU mode by default. For GPU/Metal, install the
matching JAX wheel manually following the
[official instructions](https://docs.jax.dev/en/latest/installation.html).

## Project layout details

- All hyper-parameters live in [`configs/base.yaml`](configs/base.yaml).
  Notebooks read it at the top of the file; do not duplicate values.
- Results are CSVs under `results/` and are versioned. Figures in the
  report are regenerated from these CSVs, never from one-off runs.
- `experiments/_make_notebooks.py` regenerates the experiment notebooks
  from a single Python source of truth, so notebook structure stays in
  git diffs.

## Tests

```bash
pytest -q             # 22 tests, ~45 s on CPU
ruff check otfm tests # lint
```

CI runs both on every push and PR (`.github/workflows/test.yml`).

## Authors

- Giovanni Dettori
- Mael Tremouille

## License

MIT — see [LICENSE](LICENSE).
