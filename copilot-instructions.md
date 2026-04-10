# Copilot Instructions for kite_fem

This workspace contains a Python package for finite element simulation of kites.

## Key points

- The package root is `src/kite_fem`.
- The project uses `pyproject.toml` for metadata, dependencies, and Pixi configuration.
- Python package code is under `src/kite_fem/*.py`.
- Examples are in `examples/` and validation scripts are under `examples/validation/`.

## Workflow guidance

- Prefer editing Python source files under `src/kite_fem/`.
- Use the active Python environment configured for the workspace.
- Check `pyproject.toml` before modifying dependencies or tool settings.
- Prefer `pixi` for dependency and environment management instead of `pip` when working in this repo.
- Avoid making assumptions about the environment; verify package availability with `python -m pip list` or `python -c` imports.

## Common tasks

- Fix runtime or import issues by inspecting `src/kite_fem/__init__.py`, `FEMStructure.py`, `BeamElement.py`, `SpringElement.py`, and `Functions.py`.
- Use `pytest` for tests if new code is added.
- Keep changes minimal and consistent with the existing package structure.

## Notes

- If the user asks for environment-related help, verify if `pixi` or direct `pip` is the intended tool.
- For `ipython` or shell issues, prefer diagnosing the active interpreter and cache permissions.
