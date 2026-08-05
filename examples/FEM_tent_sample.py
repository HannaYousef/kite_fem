import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D
import numpy as np
from kite_fem.FEMStructure import FEM_structure
from kite_fem.Plotting import (
    plot_structure,
    plot_convergence,
    plot_structure_with_strain,
)


def plot_deformed_structure(x, y, z, tent, N_beam, N_spring, f, u_magnitude):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Original nodes
    ax.scatter(x, y, z, marker="o", label="Original")

    # Deformed nodes (assuming coords_current = [x0,y0,z0,x1,y1,z1,...])
    x_def = tent.coords_current[0::3]
    y_def = tent.coords_current[1::3]
    z_def = tent.coords_current[2::3]

    # apply scaling factor on u
    x_def = u_magnitude * (x_def - x) + x
    y_def = u_magnitude * (y_def - y) + y
    z_def = u_magnitude * (z_def - z) + z

    ax.scatter(x_def, y_def, z_def, marker="x", label="Deformed")

    # --- Plot beam connectivity (lines) ---
    for elem in N_beam:
        i, j = elem

        # Original configuration
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], "b-", alpha=0.5)

        # Deformed configuration
        ax.plot([x_def[i], x_def[j]], [y_def[i], y_def[j]], [z_def[i], z_def[j]], "r-")

    # --- Plot spring connectivity (lines) ---
    counter = 1
    for elem in N_spring:
        i, j = elem

        # Original configuration
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], "k--", alpha=0.5)

        # Deformed configuration
        ax.plot([x_def[i], x_def[j]], [y_def[i], y_def[j]], [z_def[i], z_def[j]], "m--")

        # strain
        l0 = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)
        l = np.sqrt(
            (x_def[i] - x_def[j]) ** 2
            + (y_def[i] - y_def[j]) ** 2
            + (z_def[i] - z_def[j]) ** 2
        )
        strain = (l - l0) / l0
        print("Strain in spring {} is {}".format(counter, strain))
        counter += 1

    # --- Load quiver from global force vector ---
    f_nodes = f.reshape(-1, 6)

    Fx = f_nodes[:, 0]
    Fy = f_nodes[:, 1]
    Fz = f_nodes[:, 2]

    # Only plot where there is a force
    mask = (Fx != 0) | (Fy != 0) | (Fz != 0)

    x_load = x[mask]
    y_load = y[mask]
    z_load = z[mask]

    u = Fx[mask]
    v = Fy[mask]
    w = Fz[mask]

    # Normalize for visualization (optional but recommended)
    mag = np.sqrt(u**2 + v**2 + w**2)
    mag[mag == 0] = 1  # avoid division by zero

    u_norm = u / mag
    v_norm = v / mag
    w_norm = w / mag

    scale = 0.5  # adjust visually

    ax.quiver(x_load, y_load, z_load, u_norm, v_norm, w_norm, length=scale, color="g")

    # Formatting
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # equal scaling

    ax.legend()
    plt.show()


def subdivide_beams(x, y, z, connectivity, n):
    new_nodes = []
    new_conn = []

    # Start with existing nodes
    nodes = np.vstack((x, y, z)).T.tolist()

    for elem in connectivity:
        i, j = elem
        A = np.array(nodes[i])
        B = np.array(nodes[j])

        prev_index = i

        for k in range(1, n):
            t = k / n
            new_point = (1 - t) * A + t * B
            nodes.append(new_point.tolist())
            new_index = len(nodes) - 1

            new_conn.append([prev_index, new_index])
            prev_index = new_index

        new_conn.append([prev_index, j])

    nodes = np.array(nodes)
    new_conn = np.array(new_conn)

    return nodes[:, 0], nodes[:, 1], nodes[:, 2], new_conn


def beam_lengths(x, y, z, connectivity):
    lengths = []
    for i, j in connectivity:
        dx = x[j] - x[i]
        dy = y[j] - y[i]
        dz = z[j] - z[i]
        lengths.append(np.sqrt(dx**2 + dy**2 + dz**2))
    return np.asarray(lengths)


def scale_geometry_to_target_beam_length(x, y, z, connectivity, target_length):
    lengths = beam_lengths(x, y, z, connectivity)
    mean_length = np.mean(lengths)
    scale = target_length / mean_length
    return x * scale, y * scale, z * scale, scale


# USER INPUT
p_beam = 0.5  # bar
d_beam = 0.1  # m
E_spring = 5.5e9  # GPa
d_spring = 1  # m
n_iter_max = 500
u_magnitude = 1  # scaling factor for u plot

f_vec = [500, 5]  # define load for both geometries here in [N]
target_beam_element_length = 0.1  # m (validated range in paper: 0.5-3 m)
n_lines = [1, 1]  # one FE beam per geometric beam

geometry = 2  # 1 without spring, 2 with spring

# geometry 1 coordinates and connectivity without spring
x1 = np.array([-1, 1, 1, -1, 0])
y1 = np.array([-1, -1, 1, 1, 0])
z1 = np.array([0, 0, 0, 0, 1])
N1_beam = np.array([[0, 4], [1, 4], [2, 4], [3, 4]])

# geometry 2 coordinates and connectivity with spring
x2 = np.array([-1, 1, 1, -1, 0, 0])
y2 = np.array([-1, -1, 1, 1, -0.1, 0.1])
z2 = np.array([0, 0, 0, 0, 1, 1])
N2_beam = np.array([[0, 4], [1, 4], [2, 5], [3, 5]])
N2_spring = np.array([[4, 5]])


if geometry == 1:
    print("Running geometry without spring")
    x_scaled, y_scaled, z_scaled, scale = scale_geometry_to_target_beam_length(
        x1, y1, z1, N1_beam, target_beam_element_length
    )
    x, y, z, N_beam = subdivide_beams(x_scaled, y_scaled, z_scaled, N1_beam, n_lines[0])
    N_spring = []
    f = f_vec[0]
elif geometry == 2:
    print("Running geometry with spring")
    x_scaled, y_scaled, z_scaled, scale = scale_geometry_to_target_beam_length(
        x2, y2, z2, N2_beam, target_beam_element_length
    )
    x, y, z, N_beam = subdivide_beams(x_scaled, y_scaled, z_scaled, N2_beam, n_lines[1])
    N_spring = N2_spring
    f = f_vec[1] / 2  # apply half the load on two nodes

beam_lengths_current = beam_lengths(x, y, z, N_beam)
print(
    "Beam length stats [m]: min={:.3f}, mean={:.3f}, max={:.3f}, scale={:.3f}".format(
        np.min(beam_lengths_current),
        np.mean(beam_lengths_current),
        np.max(beam_lengths_current),
        scale,
    )
)

# coordinates
initial_conditions = []
for i in range(len(x)):
    pos = [x[i], y[i], z[i]]
    vel = [0, 0, 0]
    mass = 1
    fixed = False
    if z[i] == 0:
        fixed = True
    initial_conditions.append([pos, vel, mass, fixed])

# beam elements
beam_matrix = []
for i in range(len(N_beam)):
    n1 = N_beam[i][0]
    n2 = N_beam[i][1]
    dx = x[n2] - x[n1]
    dy = y[n2] - y[n1]
    dz = z[n2] - z[n1]
    l0 = np.sqrt(dx**2 + dy**2 + dz**2)
    beam_matrix.append([n1, n2, d_beam, p_beam, l0])

# spring elements
spring_matrix = []
A = np.pi * d_spring**2 / 4
c = None
spring_type = "default"
for i in range(len(N_spring)):
    n1 = N_spring[i][0]
    n2 = N_spring[i][1]
    dx = x[n2] - x[n1]
    dy = y[n2] - y[n1]
    dz = z[n2] - z[n1]
    l0 = np.sqrt(dx**2 + dy**2 + dz**2)
    k = E_spring * A / l0
    print("Spring stiffness k = {}".format(k))
    spring_matrix.append([n1, n2, k, c, l0, spring_type])

# Create FEM structure and show initial configuration
tent = FEM_structure(
    initial_conditions, beam_matrix=beam_matrix, spring_matrix=spring_matrix
)

# setup external forces and solve
fe = np.zeros(len(x) * 6)
index = np.where(z == max(z))[0]
for i in index:
    fe[i * 6 + 1] = f  # +1 = y-direction DOF

tent.solve(
    fe=fe,
    max_iterations=n_iter_max,
    tolerance=0.01,
    step_limit=0.005,
    relax_init=0.25,
    relax_update=0.95,
    k_update=1,
    I_stiffness=0,
)

plot_deformed_structure(x, y, z, tent, N_beam, N_spring, fe, u_magnitude)
