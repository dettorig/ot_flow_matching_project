# OT Flow Matching Coupling Study

A controlled study of how the choice of coupling
$\pi \in \Pi(\nu_0, \nu_1)$ between source and target affects
**Flow Matching** training and rollout on 2D toy data
(moons → 8-Gaussians).

ENSAE Paris — M2 Optimal Transport course project.

## Couplings compared

1. **independent** — $\pi = \nu_0 \otimes \nu_1$, random pairs.
2. **hungarian_exact_ot** — exact discrete OT via the Hungarian algorithm.
3. **sinkhorn_sampled** ($\varepsilon$) — pairs sampled from a Sinkhorn plan.
4. **sinkhorn_barycentric** ($\varepsilon$) — deterministic pairs via the
   barycentric projection of the same Sinkhorn plan.
5. **perturbed-OT** ($\alpha$) — $\pi_\alpha = (1-\alpha)\pi^\star + \alpha\pi_{\mathrm{ind}}$,
   linear interpolation between exact OT and the independent plan.

The **deliverable is the annotated notebook** `main.ipynb`. Read it
top to bottom; the helper code lives in the small `otfm/` package
(`datasets`, `couplings`, `model`, `training`, `runtime`, `solvers`,
`metrics`, `plotting`, `sweeps`, `diagnostics`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
jupyter notebook main.ipynb
```

JAX in `requirements.txt` is the CPU build. For GPU/Metal install the
matching wheel manually
([JAX install instructions](https://docs.jax.dev/en/latest/installation.html)).

## Authors

- Giovanni Dettori
- Mael Tremouille

MIT — see [LICENSE](LICENSE).
