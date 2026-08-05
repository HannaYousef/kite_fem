import numpy as np
from pyfe3d import Spring, SpringProbe, DOF

ACTIVE_LENGTH_RTOL = 1e-9

class SpringElement:
    def __init__(self, n1 : int, n2 : int, init_k_KC0 : int):
        #initialising pyfe3d spring element
        springprobe = SpringProbe()
        self.spring = Spring(springprobe)
        self.spring.init_k_KC0 = init_k_KC0
        self.spring.n1 = n1
        self.spring.n2 = n2
        self.spring.c1 = DOF * n1
        self.spring.c2 = DOF * n2
        self.update_KC0v_only = 0
        self.tension_smoothing = 0.0
        self.current_extension = 0.0
        self.current_tangent_stiffness = 0.0

    def set_spring_properties(self, l0 : float, k : float, springtype : str, i_other_pulley: int = 0):
        #setting spring properties
        self.l0 = l0 
        self.k = k
        self.spring.kxe = k  
        self.springtype = springtype.lower()

        #track indice of matching spring element in pulley system
        if self.springtype == "pulley":
            self.i_other_pulley = i_other_pulley

        #error if invalid spring type
        if self.springtype not in ("noncompressive", "default", "pulley"):
            raise ValueError("Invalid spring type. Choose from 'noncompressive', 'default', or 'pulley'.")
        
    def unit_vector(self, coords : np.ndarray):
        #calculate unit vector and length of the element
        xi = coords[self.spring.c2//2 + 0] - coords[self.spring.c1//2 + 0]
        xj = coords[self.spring.c2//2 + 1] - coords[self.spring.c1//2 + 1]
        xk = coords[self.spring.c2//2 + 2] - coords[self.spring.c1//2 + 2]
        l = np.linalg.norm([xi,xj,xk])
        if np.isclose(l, 0.0):
            raise ValueError(
                "Zero-length spring element detected; the element direction is undefined."
            )
        unit_vect = np.array([xi, xj, xk])/l
        return unit_vect,l
    
    def __update_rotation_matrix(self, coords : np.ndarray):
        #determine arbitrary vector on plane xy to describe coordinate system along with vector x
        unit_vect = self.unit_vector(coords)[0]
        xi, xj ,xk = unit_vect[0], unit_vect[1], unit_vect[2]      
        vxyi, vxyj, vxyk =  unit_vect[1], unit_vect[2], unit_vect[0] 
        if xi == xj  and xj == xk: # Edge case, if all are the same then KC0 returns NaN's
            vxyi *= -1
        #update element rotation matrix in pyfe3d
        self.spring.update_rotation_matrix(xi, xj, xk, vxyi, vxyj, vxyk)

    def update_KC0(self, KC0r : np.ndarray, KC0c : np.ndarray, KC0v : np.ndarray, coords : np.ndarray):
        #update element rotation matrix and adds contribution to global stiffness matrix
        self.__update_rotation_matrix(coords)
        self.spring.update_KC0(KC0r, KC0c, KC0v,self.update_KC0v_only)
        #set flag to only update KC0v from now on
        self.update_KC0v_only = 1
        return KC0r, KC0c, KC0v

    def update_current_stiffness_state(self, coords: np.ndarray, l_other_pulley: float = 0.0):
        #Set spring stiffness according to the current element state.
        unit_vector, l = self.unit_vector(coords)
        total_length = l + l_other_pulley
        raw_extension = total_length - self.l0
        active = True
        self.current_extension = raw_extension
        self.current_tangent_stiffness = self.k

        if self.springtype in ("noncompressive", "pulley") and self.tension_smoothing > 0.0:
            root = np.sqrt(raw_extension**2 + self.tension_smoothing**2)
            self.current_extension = 0.5 * (raw_extension + root)
            self.current_tangent_stiffness = self.k * 0.5 * (1.0 + raw_extension / root)
            active = self.current_tangent_stiffness > 0.0
        elif self.springtype in ("noncompressive", "pulley"):
            tolerance = ACTIVE_LENGTH_RTOL * max(1.0, abs(self.l0))
            active = total_length >= self.l0 - tolerance
            self.current_extension = max(raw_extension, 0.0)
            self.current_tangent_stiffness = self.k if active else 0.0

        self.spring.kxe = self.current_tangent_stiffness if active else 0.0
        return unit_vector, l, active

    def spring_internal_forces(self, coords: np.ndarray, l_other_pulley:float = 0.0):
        #Set spring stiffness
        unit_vector, l, active = self.update_current_stiffness_state(coords, l_other_pulley)
        # calculate spring force and allign with unit vector
        f_s = self.k * self.current_extension if active else 0.0
        fi = f_s * unit_vector

        #append with zeros for rotational DOF's
        fi = np.append(fi, [0, 0, 0])
        return fi
