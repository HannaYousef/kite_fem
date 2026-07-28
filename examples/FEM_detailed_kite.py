from pathlib import Path

from kite_fem.Plotting import (
    plot_structure,
    plot_structure_with_strain,
    plot_convergence,
    plot_structure_with_collapsed_beams,
)
from kite_fem.Functions import (
    adapt_stiffnesses,
    load_resultant_and_moment,
    trim_structure_for_dead_load,
)
from kite_fem.saveload import load_fem_structure
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_DIR = Path(__file__).resolve().parents[1]
source_kite = load_fem_structure(
    PROJECT_DIR / "data" / "TUDELFT_V3_KITE" / "detailed_kite_initial.npz"
)
fe = np.loadtxt(PROJECT_DIR / "data" / "TUDELFT_V3_KITE" / "fe_6d.csv")
TOW_NODE = 0
CONTINUATION_STEPS = 8

print("Frozen global dead-load approximation; use ASKITE for coupled aerodynamics.")
print("DOF", np.sum(source_kite.bc))
print("Beam", np.size(source_kite.beam_elements))
print("Springs", np.size(source_kite.spring_elements))
resultant, moment = load_resultant_and_moment(
    source_kite, fe, reference_node=TOW_NODE
)
print("Resultant load", resultant)
print(
    f"Untrimmed moment about tow node {TOW_NODE}: "
    f"{moment} (norm {np.linalg.norm(moment):.6g} N m)"
)

kite, trim_rotation = trim_structure_for_dead_load(
    source_kite,
    fe,
    reference_node=TOW_NODE,
)
_, moment = load_resultant_and_moment(kite, fe, reference_node=TOW_NODE)
print(f"Rigid trim rotation: {np.degrees(trim_rotation.magnitude()):.6g} deg")
print(f"Moment after rigid trim: {moment} (norm {np.linalg.norm(moment):.6g} N m)")

ax1, fig1 = plot_structure(
    kite,
    plot_nodes=False,
    fe=fe,
    plot_external_forces=True,
    linewidth=[1, 0.75, 1, 3.5],
    plot_node_numbers=False,
)
ax2, fig2 = plot_structure(
    kite,
    plot_nodes=False,
    plot_displacements=False,
    solver="spsolve",
    e_colors=["black", "black", "black", "black"],
    linewidth=[1, 0.75, 1, 3.5],
    plot_2d=True,
    plot_2d_plane="yz",
)
ax3, fig3 = plot_structure_with_strain(kite)

# Pseudo-transient solves move through slack-line state changes. Their proximal
# forces are temporary; only the final solve below is accepted as equilibrium.
for continuation_step in range(CONTINUATION_STEPS):
    continuation_converged, _ = kite.solve(
        fe=fe,
        max_iterations=80,
        tolerance=0.02,
        convergence_criteria="residual",
        step_limit=0.03,
        relax_init=0.3,
        relax_min=0.03,
        relax_update=0.98,
        k_update=1,
        I_stiffness=0,
        pseudo_dt=0.4,
        k_reg_min=10.0,
        beam_tangent_method="assembled",
        restore_best=True,
        print_info=False,
    )
    if not continuation_converged:
        raise RuntimeError(
            f"Continuation step {continuation_step + 1} did not converge. "
            f"Best solve residual: {kite.best_regularized_residual_norm:.6g} N."
        )
    max_element_strain = adapt_stiffnesses(kite, max_stiffness=50000.0)
    print(
        f"Continuation {continuation_step + 1}/{CONTINUATION_STEPS}: "
        f"physical residual {kite.residual_norm_history[-1]:.6g} N, "
        f"max element strain {max_element_strain:.6g}%"
    )

converged, _ = kite.solve(
    fe=fe,
    max_iterations=400,
    tolerance=0.01,
    convergence_criteria="residual",
    step_limit=0.005,
    relax_init=0.2,
    relax_min=0.01,
    relax_update=0.99,
    k_update=1,
    I_stiffness=0,
    pseudo_dt=None,
    beam_tangent_method="assembled",
    restore_best=True,
)
if not converged:
    raise RuntimeError(
        "The unregularized detailed-kite equilibrium did not converge. "
        f"Best residual: {kite.best_residual_norm:.6g} N."
    )

kite.update_internal_forces()
physical_residual = np.linalg.norm((fe - kite.fi)[kite.bc])
_, final_moment = load_resultant_and_moment(kite, fe, reference_node=TOW_NODE)
final_moment_norm = np.linalg.norm(final_moment)
print(f"Final physical residual: {physical_residual:.6g} N")
print(
    f"Final moment about tow node {TOW_NODE}: "
    f"{final_moment} (norm {final_moment_norm:.6g} N m)"
)
if physical_residual >= 0.01 or final_moment_norm >= 0.5:
    raise RuntimeError(
        "The detailed-kite state does not satisfy the physical force/moment checks."
    )

coords_initial = kite.coords_init.reshape(-1, 3)
coords_final = kite.coords_current.reshape(-1, 3)
max_reference_displacement = np.max(
    np.linalg.norm(coords_final - coords_initial, axis=1)
)
reference = coords_initial[TOW_NODE]
incremental_rotation, _ = Rotation.align_vectors(
    coords_final - coords_final[TOW_NODE],
    coords_initial - reference,
)
rigid_fit = incremental_rotation.apply(coords_initial - reference) + coords_final[
    TOW_NODE
]
max_nonrigid_displacement = np.max(
    np.linalg.norm(coords_final - rigid_fit, axis=1)
)

line_data = []
slack_line_elements = 0
for spring_id, spring in enumerate(kite.spring_elements):
    _, length = spring.unit_vector(kite.coords_current)
    if spring.springtype == "pulley":
        other = kite.spring_elements[spring.i_other_pulley]
        length += other.unit_vector(kite.coords_current)[1]
    strain = (length - spring.l0) / spring.l0
    if spring.springtype in ("noncompressive", "pulley"):
        line_data.append((strain, spring_id, length, spring))
        slack_line_elements += strain < -1e-9

beam_strains = []
for beam in kite.beam_elements:
    _, length = beam.unit_vector(kite.coords_current)
    beam_strains.append((length - beam.L) / beam.L)

line_data.sort(key=lambda item: item[0])
min_line_strain, _, _, _ = line_data[0]
max_line_strain, max_line_id, max_line_length, max_line = line_data[-1]
max_line_extension = max_line_length - max_line.l0
max_line_force = max_line.k * max_line_extension
max_beam_strain = max(abs(strain) for strain in beam_strains)
collapsed_beams = sum(beam.collapsed for beam in kite.beam_elements)

print(f"Maximum displacement from trimmed reference: {max_reference_displacement:.6g} m")
print(f"Incremental rigid rotation: {np.degrees(incremental_rotation.magnitude()):.6g} deg")
print(f"Maximum best-fit non-rigid displacement: {max_nonrigid_displacement:.6g} m")
print(
    "Tension-only element strain range: "
    f"{100 * min_line_strain:.6g}% to {100 * max_line_strain:.6g}% "
    f"({slack_line_elements} slack line elements)"
)
print(
    f"Most strained active element {max_line_id}: "
    f"extension {1000 * max_line_extension:.6g} mm, "
    f"force {max_line_force:.6g} N"
)
print(
    "Beam axial strain range: "
    f"{100 * min(beam_strains):.6g}% to {100 * max(beam_strains):.6g}%"
)
print(
    "Collapsed beams: ",
    collapsed_beams,
    "/",
    len(kite.beam_elements),
    sep="",
)
if max_line_strain >= 0.05 or max_beam_strain >= 0.005 or collapsed_beams:
    raise RuntimeError(
        "The detailed-kite state converged but failed the structural plausibility checks."
    )

ax4, fig4 = plot_structure(
    kite,
    fe=fe,
    fe_magnitude=1.5,
    plot_residual_forces=False,
    plot_external_forces=True,
    plot_nodes=False,
    plot_displacements=False,
    solver="spsolve",
    linewidth=[1, 0.75, 1, 3.5],
)
ax4.set_title("Deformed equilibrium shape")
ax2, fig2 = plot_structure(
    kite,
    plot_nodes=False,
    plot_external_forces=True,
    plot_displacements=False,
    solver="spsolve",
    e_colors=["red", "red", "red", "red"],
    linewidth=[1, 0.75, 1, 3.5],
    plot_2d=True,
    plot_2d_plane="yz",
    ax=ax2,
    fig=fig2,
)
ax2.set_title("Trimmed reference (black) and deformed equilibrium (red)")
ax5, fig5 = plot_structure_with_strain(kite)
ax6, fig6 = plot_structure_with_collapsed_beams(kite, plot_nodes=False)
ax7, fig7 = plot_convergence(kite, "residual")

if plt.get_backend().lower() == "agg":
    plt.close("all")
else:
    plt.show()
