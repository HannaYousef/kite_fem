from kite_fem.FEMStructure import FEM_structure
import numpy as np
from pyfe3d import DOF
from scipy.spatial.transform import Rotation



def _rebuild_structure_with_positions(kite, positions, velocities=None):
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (kite.num_nodes, 3):
        raise ValueError(
            f"positions must have shape ({kite.num_nodes}, 3), got {positions.shape}."
        )

    if velocities is None:
        velocities = [condition[1] for condition in kite.initial_conditions]
    velocities = np.asarray(velocities, dtype=float)
    if velocities.shape != (kite.num_nodes, 3):
        raise ValueError(
            f"velocities must have shape ({kite.num_nodes}, 3), got {velocities.shape}."
        )

    initial_conditions = []
    for node_id, (_, _, mass, fixed) in enumerate(kite.initial_conditions):
        initial_conditions.append(
            [positions[node_id].copy(), velocities[node_id].copy(), mass, fixed]
        )

    rebuilt = FEM_structure(
        initial_conditions,
        kite.spring_matrix,
        kite.pulley_matrix,
        kite.beam_matrix,
    )

    for source, target in zip(kite.spring_elements, rebuilt.spring_elements):
        target.k = source.k
        target.l0 = source.l0
        target.spring.kxe = source.k
        target.tension_smoothing = source.tension_smoothing

    beam_attributes = ("r", "A", "I", "J", "k", "p", "L", "E", "G", "collapsed")
    beam_prop_attributes = ("A", "Ay", "Az", "Iyy", "Izz", "J", "E", "G")
    for source, target in zip(kite.beam_elements, rebuilt.beam_elements):
        for attribute in beam_attributes:
            setattr(target, attribute, getattr(source, attribute))
        for attribute in beam_prop_attributes:
            setattr(target.prop, attribute, getattr(source.prop, attribute))
        target.beam.length = source.beam.length

    return rebuilt


def relaxbridles(
    kite: FEM_structure,
    canopy_nodes,
    origin,
    pull_down_force_z=-100.0,
    settle_force_z=-1.0,
):
    kite = fix_nodes(kite,canopy_nodes)
    fe = np.zeros(kite.N)

    for id in origin:
        kite.bc[6 * id+2] = True
        kite.fixed[6 * id+2] = False
        fe[6*id+2] = pull_down_force_z
    
    kite.solve(fe,max_iterations=300,tolerance=0.01,print_info=False)
    for id in origin:
        fe[6*id+2] = settle_force_z
    kite.solve(fe,max_iterations=300,tolerance=0.01,print_info=False)

    newcoords = np.reshape(kite.coords_current, (-1, 3))
    newcoords = newcoords - newcoords[origin[0]]
    return _rebuild_structure_with_positions(kite, newcoords)

def fix_nodes(kite: FEM_structure,indices):
    for id in indices:
        kite.bc[6 * id : 6 * id + 6] = False
        kite.fixed[6 * id : 6 * id + 6] = True
    return kite

def set_pressure(kite: FEM_structure,pressure):
    for beam in kite.beam_elements:
        beam.p = pressure
    return kite

def load_resultant_and_moment(structure: FEM_structure, fe, reference_node=0):
    fe = np.asarray(fe, dtype=float)
    if fe.shape != (structure.N,):
        raise ValueError(
            f"External force vector must have shape ({structure.N},), got {fe.shape}."
        )

    coords = structure.coords_current.reshape(structure.num_nodes, 3)
    nodal_loads = fe.reshape(structure.num_nodes, DOF)
    forces = nodal_loads[:, :3]
    nodal_moments = nodal_loads[:, 3:]
    reference = coords[reference_node]
    resultant = forces.sum(axis=0)
    moment = (
        np.cross(coords - reference, forces).sum(axis=0)
        + nodal_moments.sum(axis=0)
    )
    return resultant, moment


def trim_structure_for_dead_load(structure: FEM_structure, fe, reference_node=0):
    """Rotate an undeformed structure to its stable rigid-body dead-load trim.

    The force vectors stay fixed in the global frame. The returned structure is
    rebuilt at the orientation that minimizes the dead-load potential, so the
    force-position moment about ``reference_node`` is zero before flexible
    deformation. Element lengths, rest lengths, masses, and material properties
    are preserved.

    This operation is not valid for aerodynamic or other follower loads whose
    vectors must be recomputed when the structure rotates.
    """
    fe = np.asarray(fe, dtype=float)
    if fe.shape != (structure.N,):
        raise ValueError(
            f"External force vector must have shape ({structure.N},), got {fe.shape}."
        )
    if not 0 <= reference_node < structure.num_nodes:
        raise ValueError("reference_node is outside the structure node range.")
    if not np.allclose(
        structure.coords_rotations_current,
        structure.coords_rotations_init,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Dead-load trim requires an undeformed structure.")

    nodal_loads = fe.reshape(structure.num_nodes, DOF)
    forces = nodal_loads[:, :3]
    if not np.allclose(nodal_loads[:, 3:], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("Dead-load trim does not support explicit nodal moments.")

    coords = structure.coords_init.reshape(structure.num_nodes, 3)
    reference = coords[reference_node]
    relative_coords = coords - reference
    active = (np.linalg.norm(forces, axis=1) > 0.0) & (
        np.linalg.norm(relative_coords, axis=1) > 0.0
    )
    if not np.any(active):
        raise ValueError("Dead-load trim requires a force away from the reference node.")

    rotation, _ = Rotation.align_vectors(
        forces[active],
        relative_coords[active],
    )
    positions_rotated = rotation.apply(relative_coords) + reference
    velocities = np.asarray(
        [condition[1] for condition in structure.initial_conditions],
        dtype=float,
    )
    velocities_rotated = rotation.apply(velocities)
    trimmed = _rebuild_structure_with_positions(
        structure,
        positions_rotated,
        velocities_rotated,
    )
    return trimmed, rotation

def check_element_strain(structure, print_results=False):
    """
    Check current element lengths against their rest lengths (l0) with percentage strain.
    
    Parameters:
    -----------
    structure : FEM_structure
        The FEM structure object
    print_results : bool, default=True
        Whether to print the strain results to console
    return_data : bool, default=False
        Whether to return the strain data as a dictionary
    
    Returns:
    --------
    strain_data : dict (optional, if return_data=True)
        Dictionary containing strain information for springs and beams
    """
    
    strain_data = {
        'spring_strains': [],
        'beam_strains': [],
        'spring_info': [],
        'beam_info': []
    }
    
    if print_results:
        print("\n" + "="*60)
        print("ELEMENT STRAIN ANALYSIS")
        print("="*60)
    
    # Check spring elements
    if structure.spring_elements:
        if print_results:
            print(f"\nSPRING ELEMENTS ({len(structure.spring_elements)} total):")
            print("-" * 60)
            print(f"{'ID':<4} {'Type':<15} {'Current [m]':<12} {'Rest [m]':<10} {'Strain [%]':<12}")
            print("-" * 60)
        
        for i, spring_element in enumerate(structure.spring_elements):
            n1 = spring_element.spring.n1
            n2 = spring_element.spring.n2
            
            # Calculate current length
            x1 = structure.coords_current[n1 * DOF // 2]
            y1 = structure.coords_current[n1 * DOF // 2 + 1]
            z1 = structure.coords_current[n1 * DOF // 2 + 2]
            x2 = structure.coords_current[n2 * DOF // 2]
            y2 = structure.coords_current[n2 * DOF // 2 + 1]
            z2 = structure.coords_current[n2 * DOF // 2 + 2]
            
            current_length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            
            # Handle pulley elements differently
            if spring_element.springtype == "pulley":
                # For pulley elements, add the length of the matching element
                other_element = structure.spring_elements[spring_element.i_other_pulley]
                other_length = other_element.unit_vector(structure.coords_current)[1]
                total_current_length = current_length + other_length
                rest_length = spring_element.l0  # This is the total rest length for both elements
                strain_percent = (total_current_length - rest_length) / rest_length * 100
                
                # Store information about combined length for pulley elements
                spring_info_entry = {
                    'id': i,
                    'type': spring_element.springtype.capitalize(),
                    'current_length': total_current_length,
                    'individual_length': current_length,
                    'other_element_length': other_length,
                    'rest_length': rest_length,
                    'strain_percent': strain_percent,
                    'nodes': (n1, n2),
                    'other_element_id': spring_element.i_other_pulley
                }
            else:
                # For regular springs, use individual length
                rest_length = spring_element.l0
                strain_percent = (current_length - rest_length) / rest_length * 100
                
                spring_info_entry = {
                    'id': i,
                    'type': spring_element.springtype.capitalize(),
                    'current_length': current_length,
                    'rest_length': rest_length,
                    'strain_percent': strain_percent,
                    'nodes': (n1, n2)
                }
            
            strain_data['spring_strains'].append(strain_percent)
            strain_data['spring_info'].append(spring_info_entry)
            
            if print_results:
                if spring_element.springtype == "pulley":
                    print(f"{i:<4} {spring_element.springtype.capitalize():<15} {total_current_length:<12.6f} {rest_length:<10.6f} {strain_percent:<12.2f}")
                else:
                    print(f"{i:<4} {spring_element.springtype.capitalize():<15} {current_length:<12.6f} {rest_length:<10.6f} {strain_percent:<12.2f}")
    
    # Check beam elements
    if structure.beam_elements:
        if print_results:
            print(f"\nBEAM ELEMENTS ({len(structure.beam_elements)} total):")
            print("-" * 60)
            print(f"{'ID':<4} {'Type':<15} {'Current [m]':<12} {'Rest [m]':<10} {'Strain [%]':<12}")
            print("-" * 60)
        
        for i, beam_element in enumerate(structure.beam_elements):
            n1 = beam_element.beam.n1
            n2 = beam_element.beam.n2
            
            # Calculate current length
            x1 = structure.coords_current[n1 * DOF // 2]
            y1 = structure.coords_current[n1 * DOF // 2 + 1]
            z1 = structure.coords_current[n1 * DOF // 2 + 2]
            x2 = structure.coords_current[n2 * DOF // 2]
            y2 = structure.coords_current[n2 * DOF // 2 + 1]
            z2 = structure.coords_current[n2 * DOF // 2 + 2]
            
            current_length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            rest_length = beam_element.L  # Beam rest length
            strain_percent = (current_length - rest_length) / rest_length * 100
            
            strain_data['beam_strains'].append(strain_percent)
            strain_data['beam_info'].append({
                'id': i,
                'type': 'Beam',
                'current_length': current_length,
                'rest_length': rest_length,
                'strain_percent': strain_percent,
                'nodes': (n1, n2)
            })
            
            if print_results:
                print(f"{i:<4} {'Beam':<15} {current_length:<12.6f} {rest_length:<10.6f} {strain_percent:<12.2f}")
    
    # Summary statistics
    all_strains = strain_data['spring_strains'] + strain_data['beam_strains']
    
    if all_strains and print_results:
        print("\n" + "="*60)
        print("STRAIN SUMMARY:")
        print("-" * 60)
        print(f"Total elements: {len(all_strains)}")
        print(f"Maximum strain: {max(all_strains):.2f}%")
        print(f"Minimum strain: {min(all_strains):.2f}%")
        print(f"Average strain: {np.mean(all_strains):.2f}%")
        print(f"Standard deviation: {np.std(all_strains):.2f}%")
        
        # Count elements by strain level
        compression = [s for s in all_strains if s < -1]
        tension = [s for s in all_strains if s > 1]
        neutral = [s for s in all_strains if -1 <= s <= 1]
        
        print(f"\nStrain distribution:")
        print(f"  High compression (< -1%): {len(compression)} elements")
        print(f"  Neutral (-1% to 1%):      {len(neutral)} elements")
        print(f"  High tension (> 1%):      {len(tension)} elements")
        
        if compression:
            print(f"  Max compression: {min(compression):.2f}%")
        if tension:
            print(f"  Max tension: {max(tension):.2f}%")
    
    return strain_data

def adapt_stiffnesses(structure: FEM_structure, max_stiffness=50000):
    """Increase overstrained element stiffness without ever weakening an element."""
    if not np.isscalar(max_stiffness):
        raise ValueError("max_stiffness must be a finite positive scalar.")
    try:
        max_stiffness = float(max_stiffness)
    except (TypeError, ValueError) as error:
        raise ValueError("max_stiffness must be a finite positive scalar.") from error
    if not np.isfinite(max_stiffness) or max_stiffness <= 0:
        raise ValueError("max_stiffness must be a finite positive scalar.")

    max_strain = 0.0
    coords = structure.coords_current
    for spring_element in structure.spring_elements:
        if spring_element.springtype == "pulley":
            length = spring_element.unit_vector(coords)[1]
            other_element = structure.spring_elements[spring_element.i_other_pulley]
            other_length = other_element.unit_vector(coords)[1]
            length += other_length
            l0 = spring_element.l0
        else:
            length = spring_element.unit_vector(coords)[1]
            l0 = spring_element.l0
        strain = (length - l0) / l0 * 100
        if strain > 2:
            stiffness_factor = strain
        elif strain > 1:
            stiffness_factor = 2.0
        else:
            stiffness_factor = 1.0
        spring_element.k = min(
            spring_element.k * stiffness_factor,
            max_stiffness,
        )
        spring_element.spring.kxe = spring_element.k
        max_strain = max(max_strain, strain)

    for beam_element in structure.beam_elements:
        length = beam_element.unit_vector(coords)[1]
        l0 = beam_element.L
        strain = abs((length - l0) / l0 * 100)
        E = beam_element.E
        if strain > 2:
            stiffness_factor = strain
        elif strain > 1:
            stiffness_factor = 2.0
        else:
            stiffness_factor = 1.0

        current_axial_stiffness = E * beam_element.A / l0
        target_axial_stiffness = min(
            current_axial_stiffness * stiffness_factor,
            max_stiffness,
        )
        if target_axial_stiffness > current_axial_stiffness:
            beam_element.A = target_axial_stiffness * l0 / E
            beam_element.prop.A = beam_element.A
        max_strain = max(max_strain, strain)
    return max_strain
    
def extract_cross_sections(kite,canopy_sections):
    canopy_sections = np.asarray(canopy_sections)
    le_nodes = canopy_sections[:,0]
    min_le = le_nodes[0]
    max_le = le_nodes[-1]
    coords = kite.coords_current.reshape(-1,3)
    x_vectors = [] 
    y_vectors = []
    z_vectors = []

    for le_node in le_nodes:
        te_node = le_node+1
        right_node = le_node-2
        left_node = le_node+2
        x_vector = coords[te_node]-coords[le_node]
        x_vector = x_vector/np.linalg.norm(x_vector)
        if right_node < min_le :

            left_vector = coords[left_node] - coords[le_node]
            right_vector = -left_vector
        elif left_node > max_le:
            right_vector = coords[right_node] - coords[le_node]
            left_vector = -right_vector

        else:
            right_vector = coords[right_node] - coords[le_node]
            left_vector = coords[left_node] - coords[le_node]

        right_coords = coords[le_node] + right_vector
        left_coords = coords[le_node] + left_vector
        z_vector = right_coords-left_coords
        z_vector = z_vector/np.linalg.norm(z_vector)

        y_vector = -np.cross(z_vector, x_vector)
        y_vector = y_vector / np.linalg.norm(y_vector)
        x_vectors.append(x_vector)
        y_vectors.append(y_vector)

    projected_coords_list = []
    for section,x_vector,y_vector in zip(canopy_sections,x_vectors,y_vectors):
        # Project all nodes onto the xy plane
        origin = coords[le_node]
        
        projected_coords = []
        for node_id in section:
            node_pos = coords[node_id]
            relative_pos = node_pos - origin
            
            # Project onto x and y directions
            x_proj = np.dot(relative_pos, x_vector)
            y_proj = np.dot(relative_pos, y_vector)
            projected_coords.append([x_proj, y_proj])
        
        projected_coords = np.asarray(projected_coords)
        origin_displacement = projected_coords[0]
        projected_coords -= origin_displacement
        x_scale = 1.0 / projected_coords[-1][0]
        projected_coords *= x_scale

        projected_coords_list.append(projected_coords)
    return projected_coords_list

def set_new_origin(kite,node):
    coords = kite.coords_current.reshape(-1,3)
    origin = coords[node]
    coords -= origin
    kite.coords_current = coords.flatten()
    
