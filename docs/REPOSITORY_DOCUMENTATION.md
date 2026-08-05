# kite_fem Repository Documentation

## Purpose

`kite_fem` is a Python finite-element modeling package for structural simulations of leading-edge inflatable kites. The model combines:

- Axial spring elements for canopy panels, bridles, lines, and optional non-compressive members.
- Pulley elements represented as two coupled spring segments sharing one total rest length.
- Timoshenko beam elements from `pyfe3d`, with inflatable-beam stiffness properties updated from empirical relations.
- A nonlinear static equilibrium solver that iterates on free degrees of freedom until internal and external forces balance.

The central equilibrium equation solved by the package is:

```text
f_internal(u) = f_external
residual(u) = f_external - f_internal(u)
J(u) delta_u = residual(u)
u_next = u + delta_u
```

where `J(u)` is the tangent of the internal force vector with respect to nodal displacement. The implementation uses `pyfe3d` material stiffness and now adds the missing geometric tangent terms for springs and pulleys.

## Repository Layout

```text
kite_fem/
  __init__.py
  .codex
  .gitignore
  citation.cff
  LICENSE.md
  pyproject.toml
  README.md
  README_dev.md
  data/TUDELFT_V3_KITE/
  docs/images/
  examples/
  examples/validation/
  src/kite_fem/
  tests/
```

The importable package is in `src/kite_fem`. Example scripts are executable demonstrations, validation scripts, or plotting utilities. Data files store TU Delft V3 kite model inputs, force vectors, masses, and saved result states.

## Core Package Files

### `src/kite_fem/FEMStructure.py`

Defines the main `FEM_structure` class. This is the package orchestrator: it stores nodal states, builds elements, assembles global matrices and force vectors, applies boundary conditions, and runs the nonlinear solver.

Constructor inputs:

- `initial_conditions`: list of `[position, velocity, mass, fixed]` for every node. Position and velocity are 3-vectors. Mass and velocity are currently stored for compatibility but not used in the static solve.
- `spring_matrix`: optional list of `[n1, n2, k, c, l0, springtype]`.
- `pulley_matrix`: optional list of `[n1, n2, n3, k, c, l0]`. Each row becomes two spring elements: `n1-n2` and `n2-n3`.
- `beam_matrix`: optional list of `[n1, n2, d, p, l0]`.

Important attributes:

- `num_nodes`: number of nodes.
- `N`: total degrees of freedom, `DOF * num_nodes`; `pyfe3d.DOF` is six.
- `coords_init`: flattened translational initial coordinates, length `3 * num_nodes`.
- `coords_current`: flattened translational current coordinates.
- `coords_rotations_init`: flattened six-DOF initial state per node.
- `coords_rotations_current`: flattened six-DOF current state per node.
- `fixed`: Boolean mask of fixed DOFs.
- `bc`: Boolean mask where `True` means a DOF is free and included in the solve.
- `fe`: global external force vector.
- `fi`: global internal force vector.
- `KC0`: assembled global tangent/stiffness matrix.
- `Kbc`: `KC0` reduced to free DOFs.
- `spring_elements`, `beam_elements`: element object lists.
- `pulley_ids`: nodes that act as pulley center nodes.
- Solver histories: `iteration_history`, `relax_history`, `residual_norm_history`, `crisfield_history`.

Key methods:

- `update_internal_forces()`: recomputes `fi` from springs, pulleys, and beams at the current state.
- `update_stiffness_matrix()`: assembles the tangent matrix. It updates spring active/inactive states, asks `pyfe3d` for material stiffness, adds geometric tangent terms for springs and pulleys, adds optional identity stiffness, and extracts `Kbc`.
- `solve(...)`: nonlinear static solve with residual or Crisfield convergence checks, relaxation, step clipping, stiffness update frequency, and `spsolve` with `lsqr` fallback.
- `solve(..., tangent_method="assembled")`: default fast solve. Uses assembled pyfe3d stiffness plus the explicit spring/pulley geometric tangent.
- `solve(..., tangent_method="finite_difference")`: slower full Newton tangent mode. Numerically differentiates `f_internal - f_external` on free DOFs, so it captures beam property derivatives and displacement-dependent external force derivatives.
- `reset()`: restores current coordinates to the initial coordinates.
- `modify_get_spring_rest_length(...)`: updates selected spring rest lengths and returns all current spring rest lengths.

Solver notes:

- The solver residual is `fe - fi`.
- The linear step solves `Kbc * delta = residual[bc]`.
- `Kbc` must approximate `d(fi)/d(u)`, not only the secant/material stiffness.
- The spring tangent now includes transverse geometric stiffness `T / l * (I - n n^T)`.
- Pulley tangents now include both segment geometric terms and cross-coupling from each segment length into the other segment tension.
- In `assembled` mode, beam tangent behavior still relies on `pyfe3d` beam stiffness with properties updated at the current deformation.
- In `finite_difference` mode, the tangent is computed from the full force residual and therefore includes nonlinear beam-property effects present in `update_internal_forces()`.
- `fe` may be a fixed vector or a callable. If it is callable, it is evaluated as `fe(self)` each iteration. This enables displacement-dependent external forces.
- For displacement-dependent external forces, `finite_difference` mode includes the external force derivative in the tangent through `d(fi - fe)/du`.

### `src/kite_fem/SpringElement.py`

Wraps `pyfe3d.Spring` and provides force and stiffness state logic for line-like elements.

Element setup:

- `__init__(n1, n2, init_k_KC0)`: creates the pyfe3d spring, stores node IDs, global DOF offsets, and sparse matrix offset.
- `set_spring_properties(l0, k, springtype, i_other_pulley=0)`: sets rest length, stiffness, spring type, and optional paired pulley element index.

Supported spring types:

- `default`: linear axial spring that can carry tension and compression.
- `noncompressive`: carries force only when current length is at least rest length.
- `pulley`: carries force only when the combined length of its two pulley segments is at least the shared rest length.

Key methods:

- `unit_vector(coords)`: returns element unit vector and current length using flattened translational coordinates.
- `update_current_stiffness_state(coords, l_other_pulley=0.0)`: sets `spring.kxe` to either `k` or zero depending on active/compressed state; returns unit vector, length, and active flag.
- `update_KC0(...)`: updates the element rotation matrix and writes pyfe3d stiffness entries into global sparse arrays.
- `spring_internal_forces(...)`: returns a six-DOF nodal force contribution, with rotational components set to zero.

### `src/kite_fem/BeamElement.py`

Wraps `pyfe3d.BeamC` and applies empirical inflatable beam properties.

Element setup:

- `__init__(n1, n2, init_k_KC0)`: creates a pyfe3d beam, stores node IDs, global DOF offsets, and sparse offset.
- `set_inflatable_beam_properties(d, p, L)`: computes radius, area, moments of inertia, torsional constant, pressure, length, and initializes material properties.

Inflatable beam model:

- `update_inflatable_beam_properties()` derives torsional and bending stiffness from empirical relations attributed in comments to Breukels (2011).
- `get_beam_rotation()` and `get_beam_deflection()` read the pyfe3d beam probe displacement vector and scale values to a 1 m reference beam.
- `get_beam_deflection()` also sets `collapsed` based on an empirical collapse threshold.

Key methods:

- `unit_vector(coords)`: returns beam axis direction and length.
- `update_rotation_matrix(coords)`: updates pyfe3d beam orientation.
- `update_KC0(...)`: writes pyfe3d beam stiffness entries.
- `beam_internal_forces(displacement, coords, fi)`: updates orientation, beam probe displacement, inflatable properties, and internal force vector.

### `src/kite_fem/Functions.py`

Utility functions for modifying structures and extracting engineering diagnostics.

Functions:

- `relaxbridles(kite, canopy_nodes, origin)`: temporarily fixes canopy nodes, applies downward load at origin nodes, solves twice, and returns a new `FEM_structure` with relaxed bridle geometry.
- `fix_nodes(kite, indices)`: sets all six DOFs of selected nodes to fixed.
- `set_pressure(kite, pressure)`: assigns pressure to every beam element.
- `check_element_strain(structure, print_results=False)`: calculates strain percentages for spring and beam elements and optionally prints a formatted report.
- `adapt_stiffnesses(structure, max_stiffness=50000)`: increases stiffness/area for elements exceeding strain thresholds, capped by `max_stiffness`.
- `extract_cross_sections(kite, canopy_sections)`: projects canopy section node coordinates into local section planes and normalizes chord length.
- `set_new_origin(kite, node)`: shifts current coordinates so the selected node becomes the origin.

Known implementation notes:

- `adapt_stiffnesses()` uses `if strain > 1` before `elif strain > 2`, so the `> 2` branch is currently unreachable.
- `extract_cross_sections()` uses `le_node` from an earlier loop when creating `origin`; this may be unintended if each section needs its own origin.

### `src/kite_fem/Plotting.py`

Matplotlib plotting helpers for structures, forces, convergence, strain, collapsed beams, and cross sections.

Functions:

- `plot_structure(...)`: plots nodes and elements in 3D or projected 2D, with optional external, internal, residual, and displacement vectors.
- `plot_structure_with_strain(...)`: colors elements by strain using a diverging colormap and optional colorbar.
- `plot_convergence(structure, convergence_criteria="residual")`: plots residual or Crisfield history with relaxation factor on a secondary axis.
- `plot_structure_with_collapsed_beams(...)`: plots beams with `collapsed=True` in a separate color.
- `plot_cross_sections(kite, canopy_sections)`: plots normalized 2D projected canopy sections.

Important plotting conventions:

- Spring types are colored separately by default.
- Pulley nodes are highlighted using `structure.pulley_ids`.
- `plot_displacements=True` computes one linearized displacement vector from the current residual.
- 2D plotting supports `xy`, `xz`, and `yz` planes.

### `src/kite_fem/saveload.py`

Serialization helpers for `FEM_structure` objects using compressed NumPy `.npz` files.

Functions:

- `save_fem_structure(fem_structure, filepath)`: saves current coordinates, six-DOF state, force vectors, connectivity matrices, solver histories, spring properties, beam properties, collapsed flags, and reconstructed initial conditions.
- `load_fem_structure(filepath)`: reconstructs a `FEM_structure`, restores deformed state, force vectors, element properties, collapsed flags, and solver history.

Format notes:

- The saved file is binary `.npz`, compact but not human-readable.
- `allow_pickle=True` is used on load.
- Initial conditions are stored as rows `[pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, fixed_as_int]`.

### `src/kite_fem/__init__.py`

Empty package initializer. It marks `kite_fem` as an importable package but exports no public symbols directly.

## Tests

### `tests/test_spring_tangent.py`

Regression tests for the nonlinear solver tangent.

Tests:

- `test_stretched_spring_tangent_matches_finite_difference()`: verifies the assembled tangent for a stretched default spring matches finite differences of `fi`.
- `test_pulley_tangent_matches_finite_difference()`: verifies the assembled tangent for a pulley pair includes segment coupling and matches finite differences.

Helper:

- `finite_difference_internal_force_tangent(structure, eps=1e-6)`: perturbs each free DOF and approximates `d(fi)/d(u)`.

## Example Files

### `examples/FEM_noncompressive.py`

Demonstrates default and non-compressive springs in a four-node system. It applies positive and negative y-direction forces to show how the non-compressive spring becomes slack or tensioned. Produces initial and final plots.

### `examples/FEM_pulley.py`

Demonstrates a three-pulley topology with seven nodes. It solves under an angled load at the bottom node, plots the deformed geometry, plots convergence, and checks whether pulley angles are equal as expected for ideal frictionless pulleys.

### `examples/FEM_kite.py`

Hard-coded simplified TU Delft V3 kite model. It defines many nodes, spring elements, pulley elements, and a full force vector, solves the model, and plots initial/deformed structure and strain visualization.

### `examples/FEM_detailed_kite.py`

Loads `data/TUDELFT_V3_KITE/detailed_kite_initial.npz` and `fe_6d.csv`, solves the detailed kite model, and produces multiple diagnostic plots: structure, strain, collapsed beams, and convergence.

### `examples/FEM_canopy_section.py`

Creates a discretized canopy section with spring canopy members, tether springs, and beam edges. It applies a distributed load, solves the canopy, plots convergence, and reports whether any beam collapsed.

### `examples/FEM_beam_verification.py`

Contains analytic/empirical inflatable beam force-deflection and torsion curves for comparison. It also sketches a setup for using `FEM_structure` beam elements in verification, but the solve/plot comparison section is not completed.

### `examples/FEM_saddle.py`

Creates a saddle-form spring mesh adapted from the Particle System Simulator example, solves it, and plots initial/deformed shapes.

### `examples/FEM_tent_sample.py`

Creates a small tent-like structure with beams and an optional spring, scales geometry to a target beam length, applies loads, solves, and plots original versus deformed geometry. Includes local helper functions for subdivision, beam lengths, geometry scaling, and deformed plotting.

## Validation Files

### `examples/validation/Hanging_test.py`

Runs a batch of hanging-kite validation load cases in parallel. It loads `hanging_test_initial.npz`, applies gravity and point/tip loads, adjusts beam pressure, iteratively solves and adapts stiffnesses, saves each load case result, and records timing data.

Main functions:

- `create_kite()`: fresh loaded kite factory.
- `loading(N, m_arr, tip_load, point_load)`: builds the external force vector.
- `solve_single_case(args)`: solves one pressure/load combination and saves the result.
- `get_load_cases()`: returns ten pressure/load combinations.

### `examples/validation/Comparedata.py`

Compares saved validation results against measured validation data.

Important behavior:

- Loads `struc_geometry_hanging_test.yaml` using external `kitesim` utilities.
- Extracts validation lengths from solved models.
- Writes `model_results.csv`.
- Computes shape correlation and mean absolute deviation for each load case.

This script depends on `kitesim`, which is not declared in `pyproject.toml`.

### `examples/validation/Plot_case.py`

Plots saved validation cases. It can create a composite figure for multiple load cases or detailed plots for one case, including initial/deformed overlays, forces, strain, collapsed beams, convergence, and optional cross sections.

### `examples/validation/validation_data.csv`

Experimental validation reference data. Rows correspond to load cases and columns are measured geometric quantities used by `Comparedata.py`.

### `examples/validation/model_results.csv`

Model-produced validation metrics generated by `Comparedata.py`. It includes load case ID, geometric quantities, solver tolerance, and max strain.

### `examples/validation/results/*.npz`

Saved validation states:

- `initial.npz`: initial validation model.
- `load_case_1.npz` through `load_case_10.npz`: solved FEM structures for the ten validation load cases.

### `examples/validation/results/timing.csv`

Timing data written by `Hanging_test.py`, containing load case ID and elapsed solve time.

## Data Files

### `data/TUDELFT_V3_KITE/detailed_kite_initial.npz`

Saved detailed kite initial model used by `examples/FEM_detailed_kite.py`.

### `data/TUDELFT_V3_KITE/hanging_test_initial.npz`

Saved initial model for hanging-test validation.

### `data/TUDELFT_V3_KITE/fe_6d.csv`

External force vector with six entries per node. Used by the detailed kite example.

### `data/TUDELFT_V3_KITE/mass_hanging_test.csv`

Node mass data used to build gravity loads in the hanging-test validation.

### `data/TUDELFT_V3_KITE/struc_geometry_all_in_surfplan.yaml`

YAML structural geometry data for the full TU Delft V3 kite model.

### `data/TUDELFT_V3_KITE/struc_geometry_hanging_test.yaml`

YAML structural geometry data for the hanging-test model. Used by validation comparison tooling through `kitesim`.

## Documentation and Images

### `README.md`

Project overview, installation instructions for Linux, Ubuntu/Debian, and Windows, dependency notes, contribution guide, citation note, license note, and waiver/copyright text.

### `README_dev.md`

Developer workflow notes based on a Git branch model: create GitHub issues, branch from `develop`, implement, open PRs, merge, prune, and delete branches.

### `docs/images/kitemodel.svg`

SVG image of the TU Delft V3 kite finite-element model, referenced by `README.md`.

### `docs/images/hangingkite.svg`

SVG image of the V3 kite under gravity/shortened bridle lines, referenced by `README.md`.

### `docs/REPOSITORY_DOCUMENTATION.md`

This file. It documents repository structure, files, APIs, data, examples, validation artifacts, and solver behavior.

## Project Metadata Files

### `pyproject.toml`

Build and package configuration.

Important settings:

- Build backend: `setuptools.build_meta`.
- Package name: `kite_fem`.
- Version: `0.1.0`.
- Source package discovery: `src`.
- Runtime dependencies: `matplotlib`, `numpy`, `scipy`, and `pyfe3d` installed from GitHub main branch.
- Dev dependencies: `pytest`, `black`, `flake8`, `pytest-cov`.
- Pytest option: `--cov=kite_fem`.

### `LICENSE.md`

MIT license.

### `citation.cff`

Citation metadata placeholder. Currently contains only `cff-version: 1.2.0`.

### `.gitignore`

Standard Python ignore rules for bytecode, virtual environments, build artifacts, test/coverage outputs, IDE files, and common local artifacts.

### `.codex`

Empty local marker/config file.

### `__init__.py`

Empty root-level initializer. It is not part of the packaged `src/kite_fem` package but may have been added to mark the repository root as importable in some local workflows.

The solver:

- Synchronizes active/inactive stiffness state for default, non-compressive, and pulley springs before stiffness assembly.
- Adds spring geometric stiffness to the global tangent.
- Adds pulley segment geometric stiffness.
- Adds pulley cross-coupling terms because each pulley segment tension depends on both segment lengths.
- Adds `tangent_method="finite_difference"` for a full residual tangent on free DOFs.
- Allows `fe` to be a callable `fe(structure)`, enabling displacement-dependent external force models.
- In finite-difference tangent mode, forms the numerical tangent of `f_internal - f_external`, so the solver step respects `J delta = f_external - f_internal` even when external forces depend on displacement.
- Leaves the existing `spsolve`/`lsqr`, relaxation, step limiting, and convergence history behavior intact.

The new tests compare the assembled free-DOF tangent against finite differences of internal force for both a stretched spring and a pulley. They also verify that `finite_difference` mode includes the derivative of a displacement-dependent external force.
