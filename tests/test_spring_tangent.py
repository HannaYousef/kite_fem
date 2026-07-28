import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kite_fem.FEMStructure import FEM_structure


def finite_difference_internal_force_tangent(structure, eps=1e-6):
    structure.update_internal_forces()
    force_base = structure.fi.copy()
    coords_base = structure.coords_rotations_current.copy()
    free = np.where(structure.bc)[0]
    tangent = np.zeros((len(free), len(free)))

    for col, dof in enumerate(free):
        structure.coords_rotations_current = coords_base.copy()
        structure.coords_rotations_current[dof] += eps
        structure.coords_current = structure.coords_rotations_current[
            structure._FEM_structure__coordmask
        ]
        structure.update_internal_forces()
        tangent[:, col] = (structure.fi[free] - force_base[free]) / eps

    structure.coords_rotations_current = coords_base
    structure.coords_current = structure.coords_rotations_current[
        structure._FEM_structure__coordmask
    ]
    return tangent


def test_stretched_spring_tangent_matches_finite_difference():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "default"]],
    )

    structure.update_internal_forces()
    structure.update_stiffness_matrix()

    expected = finite_difference_internal_force_tangent(structure)
    np.testing.assert_allclose(structure.Kbc.toarray(), expected, atol=1e-3)


def test_pulley_tangent_matches_finite_difference():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.0, 0.0, 0.0], [0, 0, 0], 1, False],
            [[1.0, 1.0, 0.0], [0, 0, 0], 1, True],
        ],
        pulley_matrix=[[0, 1, 2, 100.0, 0.0, 1.0]],
    )

    structure.update_internal_forces()
    structure.update_stiffness_matrix()

    expected = finite_difference_internal_force_tangent(structure)
    np.testing.assert_allclose(structure.Kbc.toarray(), expected, atol=1e-3)


def test_beam_finite_difference_tangent_matches_implemented_force_derivative():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        beam_matrix=[[0, 1, 0.1, 0.5, 1.0]],
    )
    structure.coords_rotations_current = structure.coords_rotations_init.copy()
    structure.coords_rotations_current[7] += 0.04
    structure.coords_rotations_current[8] += 0.02
    structure.coords_rotations_current[10] += 0.05
    structure.coords_current = structure.coords_rotations_current[
        structure._FEM_structure__coordmask
    ]

    structure.update_internal_forces()
    structure.update_stiffness_matrix(beam_tangent_method="finite_difference")

    expected = finite_difference_internal_force_tangent(structure)
    np.testing.assert_allclose(
        structure.Kbc.toarray(),
        expected,
        rtol=5e-4,
        atol=1e-2,
    )


def test_beam_force_is_objective_under_rigid_translation_and_rotation():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, False],
            [[1.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        beam_matrix=[[0, 1, 0.2, 0.3, 1.0]],
    )
    rotation_axis = np.array([1.0, 2.0, -1.0])
    rotation_axis /= np.linalg.norm(rotation_axis)
    rigid_rotation = Rotation.from_rotvec(np.radians(35.0) * rotation_axis)
    translation = np.array([2.0, -1.0, 0.5])
    current = structure.coords_rotations_init.copy()
    current[0:3] = translation
    current[6:9] = translation + rigid_rotation.apply([1.0, 0.0, 0.0])
    current[3:6] = rigid_rotation.as_rotvec()
    current[9:12] = rigid_rotation.as_rotvec()
    structure.coords_rotations_current = current
    structure.coords_current = current[structure._FEM_structure__coordmask]

    structure.update_internal_forces()

    np.testing.assert_allclose(structure.fi, 0.0, atol=1e-10)
    assert structure.beam_elements[0].get_beam_deflection() == pytest.approx(0.0)
    assert not structure.beam_elements[0].collapsed


def test_finite_difference_tangent_includes_external_force_derivative():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[2.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "default"]],
    )
    alpha = 7.0

    def follower_force(current_structure):
        fe = np.zeros(current_structure.N)
        node_1_y = current_structure.coords_current[4]
        fe[7] = alpha * node_1_y
        return fe

    structure.solve(
        fe=follower_force,
        max_iterations=0,
        tangent_method="finite_difference",
        I_stiffness=0,
        print_info=False,
    )

    expected = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 50.0 - alpha, 0.0],
            [0.0, 0.0, 50.0],
        ]
    )
    np.testing.assert_allclose(structure.Kbc.toarray(), expected, atol=1e-3)


def test_slack_noncompressive_spring_has_zero_force_and_zero_tangent():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[0.5, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "noncompressive"]],
    )

    structure.update_internal_forces()
    structure.update_stiffness_matrix()

    np.testing.assert_allclose(structure.fi, 0.0)
    np.testing.assert_allclose(structure.Kbc.toarray(), 0.0)


def test_unloaded_equilibrium_converges_with_zero_crisfield_reference():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "default"]],
    )

    converged, _ = structure.solve(
        max_iterations=1,
        I_stiffness=0,
        print_info=False,
    )

    assert converged
    assert structure.crisfield_history == [0.0]
    assert structure.residual_norm_history == [0.0]


def test_zero_length_spring_raises_clear_error():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 0.0, "default"]],
    )

    with pytest.raises(ValueError, match="Zero-length spring"):
        structure.update_internal_forces()


def test_solver_validates_external_force_shape_and_options():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[2.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "default"]],
    )

    with pytest.raises(ValueError, match="External force vector"):
        structure.solve(fe=np.zeros(structure.N - 1), print_info=False)

    with pytest.raises(ValueError, match="convergence_criteria"):
        structure.solve(convergence_criteria="unknown", print_info=False)

    with pytest.raises(ValueError, match="step_control"):
        structure.solve(step_control="unknown", print_info=False)

    with pytest.raises(ValueError, match="fd_epsilon"):
        structure.solve(
            tangent_method="finite_difference",
            fd_epsilon=0.0,
            print_info=False,
        )

    with pytest.raises(ValueError, match="beam_tangent_method"):
        structure.solve(beam_tangent_method="unknown", print_info=False)

    with pytest.raises(ValueError, match="line_search_reduction"):
        structure.solve(line_search_reduction=1.0, print_info=False)

    with pytest.raises(ValueError, match="pseudo_dt"):
        structure.solve(pseudo_dt=-1.0, print_info=False)

    with pytest.raises(ValueError, match="k_reg_min"):
        structure.solve(k_reg_min=-1.0, print_info=False)


def test_pseudo_transient_term_is_not_counted_as_physical_internal_force():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ]
    )
    fe = np.zeros(structure.N)
    fe[6] = 10.0

    converged, _ = structure.solve(
        fe=fe,
        max_iterations=5,
        tolerance=1e-10,
        convergence_criteria="residual",
        step_limit=1.0,
        relax_init=1.0,
        I_stiffness=0,
        pseudo_dt=1.0,
        k_reg_min=100.0,
        print_info=False,
    )

    assert converged
    np.testing.assert_allclose(structure.coords_rotations_current[6], 1.1)
    np.testing.assert_allclose(structure.fi[6], 0.0)
    np.testing.assert_allclose(structure.residual_norm_history[-1], 10.0)
    np.testing.assert_allclose(
        structure.regularized_residual_norm_history[-1],
        0.0,
        atol=1e-12,
    )


def test_pseudo_transient_continuation_can_be_removed_for_physical_equilibrium():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "default"]],
    )
    fe = np.zeros(structure.N)
    fe[6] = 10.0

    structure.solve(
        fe=fe,
        max_iterations=5,
        tolerance=1e-10,
        convergence_criteria="residual",
        step_limit=1.0,
        relax_init=1.0,
        I_stiffness=0,
        pseudo_dt=1.0,
        k_reg_min=100.0,
        print_info=False,
    )
    assert structure.residual_norm_history[-1] > 1.0

    converged, _ = structure.solve(
        fe=fe,
        max_iterations=5,
        tolerance=1e-10,
        convergence_criteria="residual",
        step_limit=1.0,
        relax_init=1.0,
        I_stiffness=0,
        pseudo_dt=None,
        print_info=False,
    )

    assert converged
    np.testing.assert_allclose(structure.coords_rotations_current[6], 1.1)
    np.testing.assert_allclose(structure.fi[6], 10.0)
    np.testing.assert_allclose(structure.residual_norm_history[-1], 0.0, atol=1e-12)


def test_nearly_taut_noncompressive_spring_is_active_but_force_is_clamped():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.0 - 1e-12, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "noncompressive"]],
    )

    structure.update_internal_forces()
    structure.update_stiffness_matrix()

    np.testing.assert_allclose(structure.fi, 0.0)
    assert structure.Kbc.toarray()[0, 0] > 0.0


def test_tension_smoothing_regularizes_slightly_slack_spring():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[0.99, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "noncompressive"]],
    )

    structure.solve(
        max_iterations=0,
        I_stiffness=0,
        tension_smoothing=0.01,
        print_info=False,
    )

    assert structure.fi[6] > 0.0
    assert 0.0 < structure.Kbc.toarray()[0, 0] < 100.0


def test_repeated_solve_starts_from_current_displacement():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "default"]],
    )
    fe = np.zeros(structure.N)
    fe[6] = 1.0

    structure.solve(
        fe=fe,
        max_iterations=20,
        tolerance=1e-8,
        convergence_criteria="residual",
        step_limit=1.0,
        I_stiffness=0,
        print_info=False,
    )
    first_x = structure.coords_current[3]

    structure.solve(
        fe=2.0 * fe,
        max_iterations=20,
        tolerance=1e-8,
        convergence_criteria="residual",
        step_limit=1.0,
        I_stiffness=0,
        print_info=False,
    )
    second_x = structure.coords_current[3]

    assert second_x > first_x
    np.testing.assert_allclose(second_x, 1.02, atol=1e-8)


def test_global_step_control_preserves_direction():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[2.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "default"]],
    )
    fe = np.zeros(structure.N)
    fe[6] = 200.0
    fe[7] = 50.0

    structure.solve(
        fe=fe,
        max_iterations=1,
        convergence_criteria="residual",
        step_limit=0.01,
        step_control="global",
        relax_init=1.0,
        I_stiffness=0,
        print_info=False,
    )

    displacement = structure.coords_rotations_current - structure.coords_rotations_init
    free_step = displacement[structure.bc]
    assert np.isclose(np.max(np.abs(free_step)), 0.01)
    np.testing.assert_allclose(free_step[1] / free_step[0], 1.0, atol=1e-8)
