# Repository guidance

This library provides graph-regularized CP tensor decomposition and
a conjugate gradient solver for Sylvester equations. Users supply
their own Laplacians.

## Development

- Install the project with `uv sync`.
- Run tests with `uv run python -m unittest discover -s tests -v`.
- Verify packaging changes with `uv build`.
- The public import name is `grcp`; implementation is in `src/grcp.py`.

## Changes

- Keep changes focused; avoid unrelated rewrites.
- Add regression tests for bug fixes and numerical changes.
- Update the README when public behavior or usage changes.
- Discuss changes to the public API or mathematical objective before
  implementing them, unless the task already authorizes them.
- Summarize changes, validation results, and remaining limitations.

## Numerical behavior

- Preserve the objective documented in the README.
- Keep CP weights at one during ALS; do not normalize factors during
  optimization.
- Do not introduce ridge or other penalties unless explicitly requested.
- Preserve caller-owned tensors, Laplacians, and initial guesses.
