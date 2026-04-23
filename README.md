# OT Flow Matching Coupling Study

This repository contains a Jupyter notebook for controlled Flow Matching experiments on 2D toy data, focused on how **coupling choice** affects training and rollout behavior.

Notebook:
- `firstrun.ipynb`

## What is implemented

The notebook compares four couplings under the same model, optimizer, training budget, and rollout solver:

1. `independent`
2. `hungarian_exact_ot`
3. `sinkhorn_eps_<epsilon>` (sampled from Sinkhorn plan)
4. `sinkhorn_barycentric` (barycentric target from same Sinkhorn plan)

It also computes key metrics per coupling:
- Path Energy (PE)
- Endpoint transport error (`W2^2`, Hungarian estimate)
- Normalized Path Energy (NPE)
- Low-NFE endpoint quality (Euler steps: 4, 8, 16, 32, 64)
- Training difficulty proxies:
  - loss variance
  - target velocity variance

## Environment setup

## 1) Create environment

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

## 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) JAX note (CPU/GPU)

The `jax` package in `requirements.txt` is the CPU default in most setups.

If you need a specific backend (CUDA/ROCm/TPU), follow official JAX install instructions and install the matching wheel for your platform.

## Run

Launch Jupyter:

```bash
jupyter notebook
```

Open `firstrun.ipynb` and run cells top-to-bottom.

## Expected runtime

For `n=1000`, `train_steps=2000`, 4 couplings on CPU:
- typically several minutes to tens of minutes depending on hardware.

For faster iteration:
- reduce `n` (e.g. 256 or 512),
- reduce `train_steps` (e.g. 800–1200),
- use smaller hidden widths.

## Reproducibility notes

- Keep the same dataset samples and random seed policy across couplings.
- Keep identical model initialization, optimizer, and training budget for fair comparison.
- Use the same rollout solver and step counts when comparing low-NFE quality.

