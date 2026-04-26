# Contributing

This is a 2-person academic project (ENSAE M2, OT course). Light
conventions only.

## Branches

- `main` is shared. No force-push.
- Feature branches: `feat/<initials>-<topic>` (e.g. `feat/mt-rk4`,
  `feat/gd-vp-schedule`).

## Commit messages

`<type>: <short imperative summary>` with `<type>` in
`feat | refactor | fix | docs | chore | exp`.

Examples:
- `feat: add RK4 solver`
- `exp: schedules x couplings sweep`
- `docs: write methodology section in baseline notebook`

Keep the summary line under ~70 characters.

## Notebook hygiene

- The deliverable is `main.ipynb`. Helpers live in the `otfm/` package;
  the notebook imports from it rather than redefining functions inline.
- Add markdown cells liberally so the notebook reads as a self-contained
  report.
- Do not commit large output cells if avoidable; clear stale outputs
  (`Kernel → Restart & Clear Output`) before committing whenever the
  diff would otherwise blow up.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

CI runs ruff + pytest on every push (`.github/workflows/test.yml`).

## Reproducibility

- Seeds are explicit at the top of each experiment block. Don't change
  them silently.
- `requirements.txt` is pinned to the exact versions the project was
  developed against.
