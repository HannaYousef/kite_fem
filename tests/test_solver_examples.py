import numpy as np
import pytest

from kite_fem.Functions import (
    adapt_stiffnesses,
    load_resultant_and_moment,
    trim_structure_for_dead_load,
)
from kite_fem.FEMStructure import FEM_structure
from kite_fem.saveload import load_fem_structure


def assert_converged(structure, fe, residual_limit, **solve_kwargs):
    converged, _ = structure.solve(fe=fe, print_info=False, **solve_kwargs)
    assert converged
    assert structure.residual_norm_history[-1] < residual_limit


def angle_between(v1, v2):
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-30)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def test_load_resultant_and_moment_about_reference_node():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[2.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "default"]],
    )
    fe = np.zeros(structure.N)
    fe[1 * 6 + 2] = 3.0
    fe[1 * 6 + 4] = 2.0

    resultant, moment = load_resultant_and_moment(structure, fe)

    np.testing.assert_allclose(resultant, [0.0, 0.0, 3.0])
    np.testing.assert_allclose(moment, [0.0, -4.0, 0.0])


def test_adapt_stiffnesses_only_increases_effective_stiffness():
    spring_structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.1, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "default"]],
    )
    adapt_stiffnesses(spring_structure, max_stiffness=500.0)
    assert spring_structure.spring_elements[0].k == pytest.approx(500.0)
    assert spring_structure.spring_elements[0].spring.kxe == pytest.approx(500.0)

    beam_structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.1, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        beam_matrix=[[0, 1, 0.2, 0.3, 1.0]],
    )
    beam = beam_structure.beam_elements[0]
    area_before = beam.A
    axial_stiffness = beam.E * beam.A / beam.L

    adapt_stiffnesses(beam_structure, max_stiffness=0.5 * axial_stiffness)
    assert beam.A == pytest.approx(area_before)
    assert beam.prop.A == pytest.approx(area_before)

    adapt_stiffnesses(beam_structure, max_stiffness=2.0 * axial_stiffness)
    assert beam.A == pytest.approx(2.0 * area_before)
    assert beam.prop.A == pytest.approx(beam.A)

    with pytest.raises(ValueError, match="max_stiffness"):
        adapt_stiffnesses(beam_structure, max_stiffness=-1.0)


def test_noncompressive_example_converges_in_both_load_directions():
    initial_conditions = [
        [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
        [[1.0, 0.0, 0.0], [0, 0, 0], 1, False],
        [[2.0, 0.0, 0.0], [0, 0, 0], 1, True],
        [[1.0, 10.0, 0.0], [0, 0, 0], 1, True],
    ]
    spring_matrix = [
        [0, 1, 1.0, 1.0, 0.0, "default"],
        [1, 2, 1.0, 1.0, 0.0, "default"],
        [1, 3, 1.0, 1.0, 10.0, "noncompressive"],
    ]
    fe = np.zeros(len(initial_conditions) * 6)
    fe[7] = 10.0

    structure = FEM_structure(initial_conditions, spring_matrix)
    assert_converged(
        structure,
        fe,
        residual_limit=0.11,
        tolerance=1e-2,
        max_iterations=300,
        step_limit=0.2,
        relax_init=0.5,
        relax_update=0.95,
        k_update=1,
    )
    assert_converged(
        structure,
        -fe,
        residual_limit=0.11,
        tolerance=1e-2,
        max_iterations=300,
        step_limit=0.2,
        relax_init=0.5,
        relax_update=0.95,
        k_update=1,
    )


def test_pulley_example_converges_and_balances_angles():
    initial_conditions = [
        [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
        [[1.0, 0.0, 0.0], [0, 0, 0], 1, True],
        [[2.0, 0.0, 0.0], [0, 0, 0], 1, True],
        [[3.0, 0.0, 0.0], [0, 0, 0], 1, True],
        [[1.0, -1.0, 0.0], [0, 0, 0], 1, False],
        [[2.0, -1.0, 0.0], [0, 0, 0], 1, False],
        [[1.5, -2.0, 0.0], [0, 0, 0], 1, False],
    ]
    pulley_matrix = [
        [0, 4, 1, 1000.0, 0.0, 3.0],
        [2, 5, 3, 1000.0, 0.0, 3.0],
        [4, 6, 5, 1000.0, 0.0, 3.0],
    ]
    structure = FEM_structure(initial_conditions, pulley_matrix=pulley_matrix)
    fe = np.zeros(structure.N)
    fe[(structure.num_nodes - 1) * 6 + 1] = -100.0
    fe[(structure.num_nodes - 1) * 6] = 50.0

    assert_converged(
        structure,
        fe,
        residual_limit=1e-3,
        tolerance=1e-3,
        convergence_criteria="residual",
        max_iterations=5000,
        step_limit=0.3,
        relax_init=0.5,
        relax_update=0.95,
        k_update=1,
        I_stiffness=15,
    )

    x = structure.coords_current.reshape(structure.num_nodes, 3)
    force_vectors = fe.reshape(structure.num_nodes, 6)[:, :3]
    checks = [
        (4, 0, 1, x[6] - x[4]),
        (5, 2, 3, x[6] - x[5]),
        (6, 4, 5, force_vectors[6]),
    ]
    for node, left, right, reference in checks:
        angle_left = angle_between(x[node] - x[left], reference)
        angle_right = angle_between(x[node] - x[right], reference)
        assert abs(angle_left - angle_right) < 1.0


def test_tent_beam_spring_example_converges():
    p_beam = 0.5
    d_beam = 0.1
    E_spring = 5.5e9
    d_spring = 1.0
    target_beam_element_length = 0.1

    x = np.array([-1, 1, 1, -1, 0, 0], dtype=float)
    y = np.array([-1, -1, 1, 1, -0.1, 0.1], dtype=float)
    z = np.array([0, 0, 0, 0, 1, 1], dtype=float)
    beam_connectivity = np.array([[0, 4], [1, 4], [2, 5], [3, 5]])
    spring_connectivity = np.array([[4, 5]])

    lengths = []
    for n1, n2 in beam_connectivity:
        lengths.append(np.linalg.norm([x[n2] - x[n1], y[n2] - y[n1], z[n2] - z[n1]]))
    scale = target_beam_element_length / np.mean(lengths)
    x, y, z = x * scale, y * scale, z * scale

    initial_conditions = []
    for i in range(len(x)):
        initial_conditions.append([[x[i], y[i], z[i]], [0, 0, 0], 1, z[i] == 0])

    beam_matrix = []
    for n1, n2 in beam_connectivity:
        length = np.linalg.norm([x[n2] - x[n1], y[n2] - y[n1], z[n2] - z[n1]])
        beam_matrix.append([n1, n2, d_beam, p_beam, length])

    spring_matrix = []
    for n1, n2 in spring_connectivity:
        length = np.linalg.norm([x[n2] - x[n1], y[n2] - y[n1], z[n2] - z[n1]])
        area = np.pi * d_spring**2 / 4.0
        spring_matrix.append([n1, n2, E_spring * area / length, 0.0, length, "default"])

    structure = FEM_structure(
        initial_conditions,
        beam_matrix=beam_matrix,
        spring_matrix=spring_matrix,
    )
    fe = np.zeros(len(x) * 6)
    for i in np.where(z == max(z))[0]:
        fe[i * 6 + 1] = 2.5

    assert_converged(
        structure,
        fe,
        residual_limit=0.05,
        max_iterations=500,
        tolerance=0.01,
        step_limit=0.005,
        relax_init=0.25,
        relax_update=0.95,
        k_update=1,
        I_stiffness=0,
    )


def test_detailed_kite_data_model_builds_finite_stiffness_matrix():
    structure = load_fem_structure(
        "data/TUDELFT_V3_KITE/detailed_kite_initial.npz"
    )

    structure.update_internal_forces()
    structure.update_stiffness_matrix()

    assert structure.Kbc.shape == (np.count_nonzero(structure.bc),) * 2
    assert np.isfinite(structure.fi).all()
    assert np.isfinite(structure.Kbc.data).all()


def test_detailed_kite_dead_load_trim_preserves_lengths_and_removes_tow_moment():
    structure = load_fem_structure(
        "data/TUDELFT_V3_KITE/detailed_kite_initial.npz"
    )
    fe = np.loadtxt("data/TUDELFT_V3_KITE/fe_6d.csv")
    coords_before = structure.coords_init.reshape(-1, 3)
    spring_lengths_before = np.array(
        [
            np.linalg.norm(
                coords_before[element.spring.n2] - coords_before[element.spring.n1]
            )
            for element in structure.spring_elements
        ]
    )

    structure, rotation = trim_structure_for_dead_load(structure, fe)
    coords_after = structure.coords_init.reshape(-1, 3)
    spring_lengths_after = np.array(
        [
            np.linalg.norm(
                coords_after[element.spring.n2] - coords_after[element.spring.n1]
            )
            for element in structure.spring_elements
        ]
    )
    _, moment = load_resultant_and_moment(structure, fe)

    np.testing.assert_allclose(spring_lengths_after, spring_lengths_before, atol=1e-12)
    np.testing.assert_allclose(coords_after[0], coords_before[0], atol=1e-12)
    assert np.degrees(rotation.magnitude()) == pytest.approx(28.7255, abs=1e-4)
    assert np.linalg.norm(moment) < 1e-8

    initial_residual = np.linalg.norm(fe[structure.bc])
    converged, _ = structure.solve(
        fe=fe,
        max_iterations=60,
        tolerance=0.02,
        convergence_criteria="residual",
        step_limit=0.04,
        relax_init=0.35,
        relax_min=0.05,
        relax_update=0.98,
        I_stiffness=0,
        pseudo_dt=0.4,
        k_reg_min=10.0,
        beam_tangent_method="assembled",
        print_info=False,
    )

    assert converged
    assert structure.regularized_residual_norm_history[-1] < 0.02
    assert structure.residual_norm_history[-1] < initial_residual
    assert np.isfinite(structure.coords_current).all()
