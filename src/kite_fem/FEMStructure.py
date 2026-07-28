from kite_fem.SpringElement import SpringElement
from kite_fem.BeamElement import BeamElement
from pyfe3d import DOF, INT, DOUBLE, SpringData, BeamCData
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import lsqr, spsolve
import numpy as np
import time
import warnings


class FEM_structure:
    def __init__(self, 
                initial_conditions, #Describes each node's coordinates,velocity, mass and boundary condition 
                #Example initial_conditions = [[pos1, vel1, mass1, fixed1],[pos2, vel2, mass2, fixed]2, ... ] pos and vel are 3d vectors
                spring_matrix=None, #Describes spring connections between two nodes, each entry is one element
                #Example spring_matrix = [[n1, n2, k, c, l0, springtype],[n1, n2, k, c, l0, springtype], ... ]
                pulley_matrix=None, #Describes pulley connections between three nodes, each entry turns into two spring elements
                #Example pulley_matrix = [[n1, n2, n3, k, c, l0],[n1, n2, n3, k, c, l0], ... ]
                beam_matrix=None, #Describes bean connections between two nodes, each entry is one element
                #Example beam_matrix = [[n1, n2, d, p, l0],[n1, n2, d, p, l0], ... ]
                 ):

        #Store arrays
        self.initial_conditions = initial_conditions
        self.spring_matrix = spring_matrix
        self.pulley_matrix = pulley_matrix
        self.beam_matrix = beam_matrix

        #Determining number of nodes,DOF's and elements      
        self.num_nodes = len(initial_conditions)
        self.N = DOF * self.num_nodes
        num_spring_elements = 0
        num_beam_elements = 0
        if spring_matrix is not None:
            num_spring_elements += len(spring_matrix)
        if pulley_matrix is not None:
            num_spring_elements += 2*len(pulley_matrix)
        if beam_matrix is not None:
            num_beam_elements += len(beam_matrix)

        #allocating spare arrays for stiffness matrix
        self.__springdata = SpringData()
        self.__beamdata = BeamCData()
        array_size = (self.__springdata.KC0_SPARSE_SIZE * num_spring_elements + self.__beamdata.KC0_SPARSE_SIZE * num_beam_elements)
        self.__KC0r = np.zeros(array_size, dtype=INT)
        self.__KC0c = np.zeros(array_size, dtype=INT)
        self.__KC0v = np.zeros(array_size, dtype=DOUBLE)
        self.__init_KC0 = 0

        #Setting up the initial conditions and elements
        self.__setup_initial_conditions(initial_conditions)
        self.spring_elements = []
        self.beam_elements = []
        self.pulley_ids = []
        if spring_matrix is not None:
            self.__setup_spring_elements(spring_matrix)
        if pulley_matrix is not None:
            self.__setup_pulley_elements(pulley_matrix)
        if beam_matrix is not None:
            self.__setup_beam_elements(beam_matrix)
        #Overwriting boundary conditions from elements with fixed nodes from initial_conditions
        self.bc = np.where(self.fixed == True, False, self.bc)

        #mask to extract coords array from coords_rotations array
        self.__coordmask = np.zeros(self.N, dtype=bool)
        self.__coordmask[0::DOF] = self.__coordmask[1::DOF] = self.__coordmask[2::DOF] = True
        #Allocating force arrays
        self.fe = np.zeros(self.N, dtype=DOUBLE)
        self.fi = np.zeros(self.N, dtype=DOUBLE)
        #Identity matrix for stiffness improvement
        self.__identity_matrix = identity(self.N, format="csc")
        self.__I_stiffness = 0
        self.__fd_epsilon = 1e-6

    def __setup_initial_conditions(self, initial_conditions):
        #sets up initial positions, velocities, masses and fixed nodes. Velocities and masses are not used, but were included to match PSS inputs (https://github.com/awegroup/Particle_System_Simulator)
        self.fixed = np.zeros(self.N, dtype=bool)
        self.coords_init = np.zeros((self.num_nodes, 3), dtype=np.float64)
        self.coords_rotations_init = np.zeros((self.num_nodes, 6), dtype=np.float64)
        self.mass_diag = np.zeros(self.N, dtype=np.float64)
        #assigning all initial conditions, and setting fixed DOF's
        for id, (pos, vel, mass, fixed) in enumerate(initial_conditions):
            self.coords_init[id] = pos
            self.coords_rotations_init[id] = np.concatenate([pos, [0, 0, 0]])
            self.mass_diag[DOF * id : DOF * id + 3] = mass
            if fixed == True:
                self.fixed[DOF * id : DOF * id + 6] = True
        #Initalising coords (translational DOF's) and coords_rotation (translational+rotational DOF's) flat arrays
        self.coords_init = self.coords_init.flatten()
        self.coords_current = self.coords_init.flatten()
        self.coords_rotations_init = self.coords_rotations_init.flatten()
        self.coords_rotations_current = self.coords_rotations_init.flatten()
        #allocating displacement array for reinitialisation
        self.displacement_reinit = np.zeros(self.N, dtype=DOUBLE)
        #Initialising boundary conditions array (True = free DOF, False = fixed DOF)
        self.bc = np.ones(self.N, dtype=bool)

    def __setup_spring_elements(self, connectivity_matrix):
        for n1, n2, k, c, l0, springtype in connectivity_matrix:
            #initialise spring element and assign properties
            spring_element = SpringElement(n1, n2, self.__init_KC0)
            spring_element.set_spring_properties(l0, k, springtype)
            self.spring_elements.append(spring_element)
            #update index for sparse stiffness matrix (required for pyfe3d)
            self.__init_KC0 += self.__springdata.KC0_SPARSE_SIZE
            #fixes rotational DOF's for the nodes connected by the spring
            for id in [n1, n2]:
                self.bc[DOF * id+3 : DOF * id + 6] = False

    def __setup_pulley_elements(self, connectivity_matrix):
        for n1, n2, n3, k, c, l0 in connectivity_matrix:
            #initialise pulley as two spring elements
            i_other_pulley = len(self.spring_elements) + 1
            spring_element = SpringElement(n1, n2, self.__init_KC0)
            #first spring gets the index of the second spring to later access its length
            spring_element.set_spring_properties(l0, k, "pulley", i_other_pulley)
            self.spring_elements.append(spring_element)
            i_other_pulley -= 1
            self.__init_KC0 += self.__springdata.KC0_SPARSE_SIZE
            spring_element = SpringElement(n2, n3, self.__init_KC0)
            #second spring gets the index of the second spring to later access its length
            spring_element.set_spring_properties(l0, k, "pulley", i_other_pulley)
            self.spring_elements.append(spring_element)
            #update index for sparse stiffness matrix (required for pyfe3d)
            self.__init_KC0 += self.__springdata.KC0_SPARSE_SIZE
            #fixes rotational DOF's for the nodes connected by the springs
            self.pulley_ids.append(n2)
            for id in [n1, n2, n3]:
                self.bc[DOF * id+3 : DOF * id + 6] = False
                
    def __setup_beam_elements(self, connectivity_matrix): 
        for n1, n2, d, p, l0 in connectivity_matrix:
            #initialise beam element and assign properties
            beam_element = BeamElement(n1, n2, self.__init_KC0)
            beam_element.set_inflatable_beam_properties(d,p,l0)
            beam_element.set_reference_geometry(self.coords_init)
            self.beam_elements.append(beam_element)
            #update index for sparse stiffness matrix (required for pyfe3d)
            self.__init_KC0 += self.__beamdata.KC0_SPARSE_SIZE
            #frees all DOF's for the nodes connected by the beam (overwrites fixed DOF's from spring elements)
            for id in [n1, n2]:
                self.bc[DOF * id+3 : DOF * id + 6] = True
        
    def update_stiffness_matrix(self, beam_tangent_method="finite_difference"):
        # In assembled mode, springs/pulleys use analytical tangent terms.
        # Beams can either use pyfe3d's constitutive KC0 approximation or a
        # finite-difference derivative of the implemented beam force law.
        if beam_tangent_method not in ("assembled", "finite_difference"):
            raise ValueError("beam_tangent_method must be 'assembled' or 'finite_difference'.")

        self.__KC0v *= 0
        #Update stiffness matrix due to spring elements
        for i, spring_element in enumerate(self.spring_elements):
            if spring_element.springtype == "pulley":
                other_element = self.spring_elements[spring_element.i_other_pulley]
                l_other_pulley = other_element.unit_vector(self.coords_current)[1]
                spring_element.update_current_stiffness_state(self.coords_current, l_other_pulley)
            else:
                spring_element.update_current_stiffness_state(self.coords_current)
            self.__KC0r, self.__KC0c, self.__KC0v = spring_element.update_KC0(self.__KC0r, self.__KC0c, self.__KC0v, self.coords_current)
        #Update stiffness matrix due to beam elements
        if beam_tangent_method == "assembled":
            for beam_element in self.beam_elements:
                self.__KC0r, self.__KC0c, self.__KC0v = beam_element.update_KC0(self.__KC0r, self.__KC0c, self.__KC0v, self.coords_current)
        # if np.count_nonzero(np.isnan(self.__KC0v)) > 0:
        #     raise ValueError("NaN detected in stiffness matrix")
        
        # Assemble global stiffness matrix        
        self.KC0 = coo_matrix((self.__KC0v, (self.__KC0r, self.__KC0c)), shape=(self.N, self.N)).tocsc()
        self.KC0 += self.__spring_geometric_tangent()
        if beam_tangent_method == "finite_difference":
            self.KC0 += self.__beam_finite_difference_tangent()
        # Add identity matrix to improve convergence, this adds stiffness in each DOF
        self.KC0 += self.__identity_matrix * self.__I_stiffness
        # Extract matrix for free DOF's
        self.Kbc= self.KC0[self.bc, :][:, self.bc]

    def __spring_geometric_tangent(self):
        # Add the displacement-dependent part of d(f_internal)/d(u) for springs.
        rows = []
        cols = []
        vals = []
        visited_pulleys = set()

        def add_block(row_node, col_node, block):
            row_base = row_node * DOF
            col_base = col_node * DOF
            for i in range(3):
                for j in range(3):
                    value = block[i, j]
                    if value != 0:
                        rows.append(row_base + i)
                        cols.append(col_base + j)
                        vals.append(value)

        def add_segment_force_tangent(force_element, variable_element, block):
            force_nodes = (force_element.spring.n1, force_element.spring.n2)
            variable_nodes = (variable_element.spring.n1, variable_element.spring.n2)
            force_signs = (-1.0, 1.0)
            variable_signs = (-1.0, 1.0)
            for row_node, row_sign in zip(force_nodes, force_signs):
                for col_node, col_sign in zip(variable_nodes, variable_signs):
                    add_block(row_node, col_node, row_sign * col_sign * block)

        eye = np.eye(3)
        for i, spring_element in enumerate(self.spring_elements):
            if spring_element.springtype == "pulley":
                if i in visited_pulleys:
                    continue

                other_index = spring_element.i_other_pulley
                other_element = self.spring_elements[other_index]
                visited_pulleys.update((i, other_index))

                n1, l1, active_1 = spring_element.update_current_stiffness_state(
                    self.coords_current,
                    other_element.unit_vector(self.coords_current)[1],
                )
                n2, l2, active_2 = other_element.update_current_stiffness_state(
                    self.coords_current,
                    l1,
                )
                if not (active_1 and active_2):
                    continue

                tension = spring_element.k * spring_element.current_extension
                tangent_1 = tension / l1 * (eye - np.outer(n1, n1))
                tangent_2 = tension / l2 * (eye - np.outer(n2, n2))
                add_segment_force_tangent(spring_element, spring_element, tangent_1)
                add_segment_force_tangent(other_element, other_element, tangent_2)
                add_segment_force_tangent(spring_element, other_element, spring_element.current_tangent_stiffness * np.outer(n1, n2))
                add_segment_force_tangent(other_element, spring_element, other_element.current_tangent_stiffness * np.outer(n2, n1))
                continue

            unit_vector, length, active = spring_element.update_current_stiffness_state(self.coords_current)
            if not active:
                continue

            tension = spring_element.k * spring_element.current_extension
            tangent = tension / length * (eye - np.outer(unit_vector, unit_vector))
            add_segment_force_tangent(spring_element, spring_element, tangent)

        if not vals:
            return coo_matrix((self.N, self.N), dtype=DOUBLE).tocsc()
        return coo_matrix((vals, (rows, cols)), shape=(self.N, self.N), dtype=DOUBLE).tocsc()

    def __beam_element_force(self, beam_element):
        fi = np.zeros(self.N, dtype=DOUBLE)
        displacement = self.coords_rotations_current - self.coords_rotations_init
        return beam_element.beam_internal_forces(displacement, self.coords_current, fi)

    def __beam_finite_difference_tangent(self):
        # Differentiate each implemented beam force contribution, including the
        # current orientation and the empirical inflatable-beam E/G updates.
        if not self.beam_elements:
            return coo_matrix((self.N, self.N), dtype=DOUBLE).tocsc()

        rows = []
        cols = []
        vals = []
        coords_rotations_base = self.coords_rotations_current.copy()
        coords_base = self.coords_current.copy()

        for beam_element in self.beam_elements:
            element_dofs = np.concatenate(
                [
                    np.arange(beam_element.beam.c1, beam_element.beam.c1 + DOF),
                    np.arange(beam_element.beam.c2, beam_element.beam.c2 + DOF),
                ]
            )
            base_force = self.__beam_element_force(beam_element)

            for dof in element_dofs:
                if not self.bc[dof]:
                    continue

                perturbation = self.__fd_epsilon * max(1.0, abs(coords_rotations_base[dof]))
                self.coords_rotations_current = coords_rotations_base.copy()
                self.coords_rotations_current[dof] += perturbation
                self.coords_current = self.coords_rotations_current[self.__coordmask]

                perturbed_force = self.__beam_element_force(beam_element)
                derivative = (
                    perturbed_force[element_dofs] - base_force[element_dofs]
                ) / perturbation

                for row, value in zip(element_dofs, derivative):
                    if value != 0:
                        rows.append(row)
                        cols.append(dof)
                        vals.append(value)

            self.coords_rotations_current = coords_rotations_base
            self.coords_current = coords_base

        self.coords_rotations_current = coords_rotations_base
        self.coords_current = coords_base
        for beam_element in self.beam_elements:
            self.__beam_element_force(beam_element)

        if not vals:
            return coo_matrix((self.N, self.N), dtype=DOUBLE).tocsc()
        return coo_matrix((vals, (rows, cols)), shape=(self.N, self.N), dtype=DOUBLE).tocsc()

    
    def update_internal_forces(self):
        self.fi *= 0
        #Add spring and pulley internal forces
        for spring_element in self.spring_elements:
            #retrieve internal forces from spring element
            if spring_element.springtype == "pulley":
                #add length of matching spring for pulley systems
                other_element = self.spring_elements[spring_element.i_other_pulley]
                l_other_pulley = other_element.unit_vector(self.coords_current)[1]
                fi_element = spring_element.spring_internal_forces(self.coords_current, l_other_pulley)
            else:
                fi_element = spring_element.spring_internal_forces(self.coords_current)
            #allocation of spring forces to nodes
            self.fi[spring_element.spring.n1 * DOF : (spring_element.spring.n1 + 1) * DOF] -= fi_element
            self.fi[spring_element.spring.n2 * DOF : (spring_element.spring.n2 + 1) * DOF] += fi_element

        #Add beam internal forces
        displacement = self.coords_rotations_current - self.coords_rotations_init
        for beam_element in self.beam_elements:
            self.fi = beam_element.beam_internal_forces(displacement,self.coords_current,self.fi)

    def __external_forces(self, fe):
        if fe is None:
            return np.zeros(self.N, dtype=DOUBLE)
        if callable(fe):
            external_forces = np.asarray(fe(self), dtype=DOUBLE)
        else:
            external_forces = np.asarray(fe, dtype=DOUBLE)
        if external_forces.shape != (self.N,):
            raise ValueError(
                f"External force vector must have shape ({self.N},), got {external_forces.shape}."
            )
        return external_forces

    def __finite_difference_tangent(self, fe, base_fi, base_fe):
        tangent = np.zeros((np.count_nonzero(self.bc), np.count_nonzero(self.bc)), dtype=DOUBLE)
        coords_rotations_base = self.coords_rotations_current.copy()
        coords_base = self.coords_current.copy()
        fi_base = self.fi.copy()
        fe_base = self.fe.copy()
        free_dofs = np.where(self.bc)[0]
        equilibrium_base = base_fi - base_fe

        for col, dof in enumerate(free_dofs):
            perturbation = self.__fd_epsilon * max(1.0, abs(coords_rotations_base[dof]))
            self.coords_rotations_current = coords_rotations_base.copy()
            self.coords_rotations_current[dof] += perturbation
            self.coords_current = self.coords_rotations_current[self.__coordmask]
            self.update_internal_forces()
            fe_perturbed = self.__external_forces(fe)
            equilibrium_perturbed = self.fi - fe_perturbed
            tangent[:, col] = (
                equilibrium_perturbed[free_dofs] - equilibrium_base[free_dofs]
            ) / perturbation

        self.coords_rotations_current = coords_rotations_base
        self.coords_current = coords_base
        self.fi = fi_base
        self.fe = fe_base
        tangent_matrix = coo_matrix(tangent).tocsc()
        if self.__I_stiffness != 0:
            tangent_matrix += identity(len(free_dofs), format="csc") * self.__I_stiffness
        return tangent_matrix

    def solve(
        self,
        fe=None,                    #external force vector for each DOF (length is self.N), if None then zero vector is used
        max_iterations=100,         #maximum number of iterations
        tolerance=1e-2,             #convergence tolerance in based on norm of residual forces (residual = fe - fi) [N]
        convergence_criteria = "crisfield", #crisfield or residual
        step_limit=0.2,             #maximum displacement or rotation step for each DOF per iteration (important for convergence)
        step_control="global",      #global preserves the correction direction; component clips each DOF independently
        relax_init=0.5,             #initial relaxation factor to scale displacement updats
        relax_update=0.95,          #relaxation factor update if not converging
        relax_min=0.0,              #Minimum value of the relax factor
        k_update=1,                 #frequency of stiffness matrix updates k_update=1 means updating every iteration.     
        I_stiffness=25,             #identity matrix stiffness addition to improve convergence
        pseudo_dt=None,             #pseudo-transient timestep; use only for continuation before an unregularized solve
        k_reg_min=0.0,              #minimum proximal stiffness for lightly massed DOFs during pseudo-transient continuation
        tangent_method="assembled", #assembled uses element tangents; finite_difference differentiates the complete residual
        beam_tangent_method="finite_difference", #only for tangent_method='assembled': assembled uses pyfe3d KC0; finite_difference differentiates beam force
        fd_epsilon=1e-6,            #relative finite-difference perturbation for tangent_method="finite_difference"
        tension_smoothing=0.0,      #smooth positive-part length scale for tension-only springs/pulleys; 0 keeps exact active-set behavior
        line_search=False,          #if True, backtrack the accepted step until the residual norm does not increase
        line_search_reduction=0.5,  #step-length reduction factor used by line_search
        line_search_min=1e-4,       #minimum step-length factor used by line_search
        restore_best=True,          #if not converged, restore the state with the lowest residual norm found
        print_info = True,          #print solver timing and convergence info
        

    ):
        #set timing information
        start_time = time.perf_counter()
        timings = {
            "update_internal_forces": 0.0,
            "update_stiffness": 0.0,
            "linear_solve": 0.0,
        }
        if tangent_method not in ("assembled", "finite_difference"):
            raise ValueError("tangent_method must be 'assembled' or 'finite_difference'.")
        if beam_tangent_method not in ("assembled", "finite_difference"):
            raise ValueError("beam_tangent_method must be 'assembled' or 'finite_difference'.")
        if convergence_criteria not in ("crisfield", "residual"):
            raise ValueError("convergence_criteria must be 'crisfield' or 'residual'.")
        if step_control not in ("global", "component"):
            raise ValueError("step_control must be 'global' or 'component'.")
        if fd_epsilon <= 0:
            raise ValueError("fd_epsilon must be positive.")
        if tension_smoothing < 0:
            raise ValueError("tension_smoothing must be non-negative.")
        if pseudo_dt is not None:
            if not np.isscalar(pseudo_dt):
                raise ValueError("pseudo_dt must be a finite positive scalar or None.")
            try:
                pseudo_dt = float(pseudo_dt)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "pseudo_dt must be a finite positive scalar or None."
                ) from error
            if not np.isfinite(pseudo_dt) or pseudo_dt <= 0:
                raise ValueError("pseudo_dt must be a finite positive scalar or None.")
        if not np.isscalar(k_reg_min):
            raise ValueError("k_reg_min must be a finite non-negative scalar.")
        try:
            k_reg_min = float(k_reg_min)
        except (TypeError, ValueError) as error:
            raise ValueError("k_reg_min must be a finite non-negative scalar.") from error
        if not np.isfinite(k_reg_min) or k_reg_min < 0:
            raise ValueError("k_reg_min must be a finite non-negative scalar.")
        if not (0.0 < line_search_reduction < 1.0):
            raise ValueError("line_search_reduction must be between 0 and 1.")
        if not (0.0 < line_search_min <= 1.0):
            raise ValueError("line_search_min must be between 0 and 1.")

        #set solver parameters
        self.__I_stiffness = I_stiffness
        self.__fd_epsilon = fd_epsilon
        for spring_element in self.spring_elements:
            spring_element.tension_smoothing = tension_smoothing
        relax = relax_init

        # Initialize displacement from the current state. This preserves an already
        # deformed configuration when solve() is called repeatedly for load stepping.
        displacement = self.coords_rotations_current - self.coords_rotations_init
        displacement_reference = displacement.copy()
        if pseudo_dt is None:
            regularization = np.zeros(self.N, dtype=DOUBLE)
        else:
            regularization = np.maximum(self.mass_diag / pseudo_dt**2, k_reg_min)
        self.iteration_history = []
        self.relax_history = []
        self.residual_norm_history = []
        self.regularized_residual_norm_history = []
        self.crisfield_history = []
        self.best_iteration = None
        self.best_residual_norm = np.inf
        self.best_regularized_residual_norm = np.inf
        self.best_crisfield = np.inf
        global_best_state = None
        converged = False

        def scaling(vec, D):
            """
                A. Peano and R. Riccioni, Automated discretisatton error
                control in finite element analysis. In Finite Elements m
                the Commercial Enviror&ent (Editei by J. 26.  Robinson),
                pp. 368-387. Robinson & Assoc., Verwood.  England (1978)
            """
            non_nulls = ~np.isclose(D, 0)
            vec = vec[non_nulls]
            D = D[non_nulls]
            return np.sqrt((vec*np.abs(1/D))@vec)
        
        #start of newton-raphson solver
        for iteration in range(max_iterations + 1):
            #calculate internal forces
            t0 = time.perf_counter()
            self.update_internal_forces()
            timings["update_internal_forces"] += time.perf_counter() - t0
            self.fe = self.__external_forces(fe)
            
            #update stiffness matrix, initially and every k_update iterations
            if iteration % k_update == 0:
                t0 = time.perf_counter()
                if tangent_method == "finite_difference":
                    self.Kbc = self.__finite_difference_tangent(
                        fe,
                        self.fi.copy(),
                        self.fe.copy(),
                    )
                else:
                    self.update_stiffness_matrix(beam_tangent_method=beam_tangent_method)
                timings["update_stiffness"] += time.perf_counter() - t0
            
            # The proximal term stabilizes the continuation subproblem only.
            # Keep the physical residual separate so final equilibrium can be
            # verified with pseudo_dt=None.
            physical_residual = self.fe - self.fi
            residual = physical_residual - regularization * (
                displacement - displacement_reference
            )
            if pseudo_dt is None:
                Ksolve = self.Kbc
            else:
                Ksolve = self.Kbc + diags(
                    regularization[self.bc], 0, format="csc"
                )
            Diagonal = Ksolve.diagonal()
            residual_norm = np.linalg.norm(physical_residual[self.bc])
            regularized_residual_norm = np.linalg.norm(residual[self.bc])
            crisfield_reference = max(
                scaling(self.fe[self.bc], Diagonal),
                scaling(self.fi[self.bc], Diagonal),
            )
            if np.isclose(crisfield_reference, 0.0):
                crisfield_test = (
                    0.0
                    if np.isclose(regularized_residual_norm, 0.0)
                    else regularized_residual_norm
                )
            else:
                crisfield_test = scaling(residual[self.bc], Diagonal) / crisfield_reference
            self.residual_norm_history.append(residual_norm)
            self.regularized_residual_norm_history.append(
                regularized_residual_norm
            )
            self.iteration_history.append(iteration)
            self.crisfield_history.append(crisfield_test)
            self.relax_history.append(relax)
            if residual_norm < self.best_residual_norm:
                self.best_iteration = iteration
                self.best_residual_norm = residual_norm
                self.best_crisfield = crisfield_test
            if regularized_residual_norm < self.best_regularized_residual_norm:
                self.best_regularized_residual_norm = regularized_residual_norm
                global_best_state = (
                    displacement.copy(),
                    self.coords_rotations_current.copy(),
                    self.coords_current.copy(),
                    self.fi.copy(),
                    self.fe.copy(),
                )

            #Pick convergence criteria
            if convergence_criteria == "crisfield":
                convergence_value = crisfield_test
            else:
                convergence_value = regularized_residual_norm

            #Check for convergence       
            if convergence_value < tolerance:
                if print_info:
                    print(
                        f"Converged after {iteration} iterations. Physical residual: {residual_norm:.3g} N, "
                        f"solve residual: {regularized_residual_norm:.3g} N, Crisfield: {crisfield_test:.3g}"
                    )
                converged = True
                break

            #check for max iterations reached
            if iteration == max_iterations:
                if print_info:
                    print(
                        f"Did not converge after {max_iterations} iterations. Physical residual: {residual_norm:.3g} N, "
                        f"solve residual: {regularized_residual_norm:.3g} N, Crisfield: {crisfield_test:.3g}"
                    )
                break

            #update relaxation factor if not converging over the last 20 steps
            if convergence_criteria == "crisfield":
                if iteration > 20 and self.crisfield_history[-1] >= np.min(
                    self.crisfield_history[-20:-1]
                ):
                    relax *= relax_update
            else:
                if iteration > 20 and self.regularized_residual_norm_history[-1] >= np.min(
                    self.regularized_residual_norm_history[-20:-1]
                ):
                    relax *= relax_update

            relax = max(relax,relax_min)

            #solve the linear system Ku=r for u (displacement delta), use spsolve with fallback on lsqr
            t0 = time.perf_counter()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    displacement_delta = spsolve(Ksolve, residual[self.bc])
                    #fall back on lsqr solver if spsolve generates warnings
                    if w:  
                        if print_info:
                            print(f"spsolve generated warnings: {[warning.message for warning in w]}. Falling back to lsqr solver.")
                        displacement_delta = lsqr(Ksolve, residual[self.bc], atol=1e-7, btol=1e-7)[0]
                except Exception as e:
                    #fall back on lsqr solver if spsolve fails
                    if print_info:
                        print(f"spsolve failed with error: {e}. Falling back to lsqr solver.")
                    displacement_delta = lsqr(Ksolve, residual[self.bc], atol=1e-7, btol=1e-7)[0]


            timings["linear_solve"] += time.perf_counter() - t0

            #relax displacement delta and apply step limits, then update displacement array
            displacement_step = displacement_delta * relax
            if step_control == "global":
                max_step = np.max(np.abs(displacement_step))
                if max_step > step_limit:
                    displacement_step *= step_limit / max_step
            else:
                displacement_step = np.clip(
                    displacement_step, -step_limit, step_limit
                )
            if line_search:
                displacement_base = displacement.copy()
                fi_base = self.fi.copy()
                fe_base = self.fe.copy()
                trial_best_residual_norm = regularized_residual_norm
                trial_best_state = None
                alpha = 1.0

                while alpha >= line_search_min:
                    displacement_trial = displacement_base.copy()
                    displacement_trial[self.bc] += alpha * displacement_step
                    self.coords_rotations_current = self.coords_rotations_init + displacement_trial
                    self.coords_current = self.coords_rotations_current[self.__coordmask]
                    self.update_internal_forces()
                    fe_trial = self.__external_forces(fe)
                    physical_residual_trial = fe_trial - self.fi
                    residual_trial = physical_residual_trial - regularization * (
                        displacement_trial - displacement_reference
                    )
                    residual_norm_trial = np.linalg.norm(residual_trial[self.bc])

                    if residual_norm_trial < trial_best_residual_norm:
                        trial_best_residual_norm = residual_norm_trial
                        trial_best_state = (
                            displacement_trial,
                            self.coords_rotations_current.copy(),
                            self.coords_current.copy(),
                            self.fi.copy(),
                            fe_trial.copy(),
                        )
                    alpha *= line_search_reduction

                if trial_best_state is None:
                    trial_best_state = (
                        displacement_base,
                        self.coords_rotations_init + displacement_base,
                        (self.coords_rotations_init + displacement_base)[self.__coordmask],
                        fi_base,
                        fe_base,
                    )

                (
                    displacement,
                    self.coords_rotations_current,
                    self.coords_current,
                    self.fi,
                    self.fe,
                ) = trial_best_state
            else:
                displacement[self.bc] += displacement_step
                self.coords_rotations_current = self.coords_rotations_init + displacement
                self.coords_current = self.coords_rotations_current[self.__coordmask]
            
        if restore_best and not converged and global_best_state is not None:
            (
                displacement,
                self.coords_rotations_current,
                self.coords_current,
                self.fi,
                self.fe,
            ) = global_best_state

        # Calculate runtime
        end_time = time.perf_counter()
        runtime = end_time - start_time

        if print_info:
            #print timing information
            print(f"Solver time: {runtime:.4f} s")
            iters = max(1, len(self.iteration_history))
            print("Timing summary (total / per-iter) [s]:")
            for k, v in timings.items():
                print(f"  {k:22s}: {v:.4f} / {v/iters:.6f}")

        return converged, runtime

    def reset(self):
        #Resets the structure to the initial conditions
        self.coords_current = self.coords_init
        self.coords_rotations_current = self.coords_rotations_init

    def modify_get_spring_rest_length(self, spring_ids = [], new_l0s = []): #TODO move outside of class and into functions
        #allows for modifying the rest length of a spring (usefull for power and steering lines), and returns all rest lengths
        for spring_id, new_l0 in zip(spring_ids, new_l0s):
            self.spring_elements[spring_id].l0 = new_l0
        rest_lengths = np.array([spring.l0 for spring in self.spring_elements])
        return rest_lengths   




