from pathlib import Path

from kite_fem.FEMStructure import FEM_structure
from kite_fem.Plotting import (
    plot_structure,
    plot_structure_with_strain,
    plot_convergence,
    plot_structure_with_collapsed_beams,
    plot_cross_sections
)
from kite_fem.Functions import relaxbridles,fix_nodes,adapt_stiffnesses
from kite_fem.saveload import save_fem_structure,load_fem_structure
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
kite = load_fem_structure(PROJECT_DIR / "data" / "TUDELFT_V3_KITE" / "detailed_kite_initial.npz")
fe = np.loadtxt(PROJECT_DIR / "data" / "TUDELFT_V3_KITE" / "fe_6d.csv")

print("DOF",np.sum(kite.bc))
print("Beam",np.size(kite.beam_elements))
print("Springs",np.size(kite.spring_elements))

ax1,fig1 = plot_structure(kite,plot_nodes=False,fe=fe,plot_external_forces=True,linewidth = [1,0.75,1,3.5],plot_node_numbers=False)
ax2,fig2 = plot_structure(kite, plot_nodes=False,plot_displacements=False,solver="spsolve",e_colors = ['black', 'black', 'black', 'black'],linewidth = [1,0.75,1,3.5],plot_2d=True,plot_2d_plane="yz")
ax3,fig3 = plot_structure_with_strain(kite)
kite.solve(fe=fe, max_iterations=1000, tolerance=0.01, step_limit=.005, relax_init=.25, relax_min=0.00, relax_update=0.9998, k_update=1,I_stiffness=15)

ax4,fig4 = plot_structure(kite,fe=fe,fe_magnitude=1.5, plot_residual_forces=False,plot_external_forces=True,plot_nodes=False,plot_displacements=False,solver="spsolve",linewidth = [1,0.75,1,3.5])
ax2,fig2 = plot_structure(kite,plot_nodes=False,plot_external_forces=True,plot_displacements=False,solver="spsolve",e_colors = ['red', 'red', 'red', 'red'], linewidth = [1,0.75,1,3.5],plot_2d=True,plot_2d_plane="yz",ax=ax2,fig=fig2)
ax5,fig5 = plot_structure_with_strain(kite)
ax6,fig6 = plot_structure_with_collapsed_beams(kite,plot_nodes=False)
ax7,fig7 = plot_convergence(kite,"crisfield")

plt.show()