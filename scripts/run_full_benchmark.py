"""Non-interactive benchmark runner.

Runs the four canonical couplings across multiple seeds and writes a tidy
CSV to ``results/full_v1.csv``. Useful when you want to separate compute
from plotting.

Usage::

    python scripts/run_full_benchmark.py --config configs/base.yaml \\
        --output results/full_v1.csv

The script is intentionally light on dependencies: it only uses ``otfm``
plus pyyaml / pandas / numpy.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import yaml

from otfm.sweeps import SweepConfig, run_baseline_sweep


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("configs/base.yaml"),
        help="YAML config file (default: configs/base.yaml).",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/full_v1.csv"),
        help="Where to write the resulting CSV.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override the number of samples per distribution.",
    )
    p.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Override the training budget (Adam steps).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override the seed list.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    sc = SweepConfig(
        n=args.n if args.n is not None else cfg["multiseed"]["n"],
        train_steps=(
            args.train_steps if args.train_steps is not None else cfg["multiseed"]["train_steps"]
        ),
        widths=cfg["model"]["widths"],
        seeds=args.seeds if args.seeds is not None else cfg["multiseed"]["seeds"],
    )

    print(f"[run_full_benchmark] config={args.config} seeds={sc.seeds} n={sc.n}")
    t0 = time.time()
    df = run_baseline_sweep(sc)
    elapsed = time.time() - t0
    print(f"[run_full_benchmark] sweep done in {elapsed:.1f}s, {len(df)} rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[run_full_benchmark] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
