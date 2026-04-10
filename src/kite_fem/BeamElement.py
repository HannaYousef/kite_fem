import numpy as np
from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCProbe, DOF

class BeamElement:
    def __init__(self, n1 : int, n2 : int, init_k_KC0 : int):
        #initialising pyfe3d BeamC element
        beamprobe = BeamCProbe()
        self.beam = BeamC(beamprobe)
        self.prop = BeamProp()
        self.beam.init_k_KC0 = init_k_KC0
        self.beam.n1 = n1
        self.beam.n2 = n2
        self.beam.c1 = DOF * n1
        self.beam.c2 = DOF * n2
        self.update_KC0v_only = 0

    def set_inflatable_beam_properties(self,d,p,L):
        #Set properties
        self.r = 0.5*d
        self.A = np.pi * self.r**2
        self.I = np.pi * self.r**4 / 4.0
        self.J = self.I*2   
        self.k = 8/9
        self.p = p
        self.L = L
        self.collapsed = False

        #Set Pyfe3d properties
        self.prop.A = self.A
        self.prop.Ay = 0
        self.prop.Az = 0
        self.prop.Iyy = self.I
        self.prop.Izz = self.I
        self.prop.J = self.I*2
        self.beam.length = L

        #Determine G and E
        self.update_inflatable_beam_properties()

    def update_inflatable_beam_properties(self):
        ##Determine GJ from rotation using empirical relations for 1 meter inflatable beams by Breukels (2011), "An engineering methodology for kite design"
        rotation = self.get_beam_rotation()
        #lower bound on rotation ensure GJ != 0
        if rotation <= 0.03:
            rotation = 0.03

        #Breukels' equation matchin tip rotation and tip moment
        C13 = 1467
        C14 = 40.908
        C15 = -191.8
        C16 = 47.406
        C17 = -17703
        C18 = 358.05
        C19 = 0.0918
        c1 = ((C13*self.r+C14)*self.p+(C15*self.r+C16))
        c2 = ((C17*self.r**4)*np.log(self.p)+(C18*self.r**3+C19))
        T = c1*np.arctan(c2*rotation)

        #Calculate GJ from timoshenko beam equations with beam length L=1m
        GJ = T*1/(rotation)
        self.G = GJ/self.J
        self.prop.G = self.G
        if self.G < 0:
            print(
                "Invalid beam shear modulus detected: "
                f"G={self.G:.6e} Pa (d={2*self.r:.6f} m, p={self.p:.6f} bar, L={self.L:.6f} m)"
            )
            raise ValueError("Beam shear modulus G must be non-negative.")

        ##Determine EI from deflection
        deflection = self.get_beam_deflection()
        #lower bound on deflection ensure EI != 0
        if deflection <= 0.03:
            deflection = 0.03

        #Breukels' equation matching deflection and tip load
        C1 = 6582.82
        C2 = -272.43
        C3 = 40852.38
        C4 = 14.31
        C5 = 271865251.42
        C6 = 215.93
        C7 = 14021.79
        C8 = -589.05
        denom = (C1 * self.r + C2) * self.p**2 + (C3 * self.r**3 + C4)
        numer = (C5 * self.r**5 + C6) * self.p + (C7 * self.r + C8)
        P = denom * (1 - np.exp(-(numer / denom) * (deflection)))

        #Calculate EI from timoshenko beam equations with beam length L=1m
        EI = P*1**3/(3*(deflection-(P*1/((8/9)*self.A*self.G))))
        self.E = EI/self.I
        self.prop.E = self.E
        if self.E < 0:
            print(
                "Invalid beam Young's modulus detected: "
                f"E={self.E:.6e} Pa (d={2*self.r:.6f} m, p={self.p:.6f} bar, L={self.L:.6f} m)"
            )
            raise ValueError("Beam Young's modulus E must be non-negative.")


    def get_beam_deflection(self):
        #obtain deflection from displacement vector
        tipdeflection = np.array(self.beam.probe.ue[7:9])
        basedeflection = np.array(self.beam.probe.ue[1:3])
        #Scaling parameter to scale deflection to that of a 1m beam
        scale = (1/self.L)
        deflection = np.linalg.norm(tipdeflection - basedeflection)*scale
        
        # check collapse TODO: verify collapse
        C9 = 322.55
        C10 = 0.0239
        C11 = 5.3833
        C12 = 0.0461
        deflection_collapse = (C9*self.r**4+C10)*self.p+C11*self.r**2+C12
        #Set collapsed flag
        if deflection > deflection_collapse:
            self.collapsed = True
        else:
            self.collapsed = False

        return deflection
    
    def get_beam_rotation(self):
        #obtain rotation from displacement vector
        tiprotation = np.array(self.beam.probe.ue[9])
        baserotation = np.array(self.beam.probe.ue[3])
        #Scaling parameter to scale deflection to that of a 1m beam
        scale = (1/self.L)
        rotation = np.linalg.norm(tiprotation-baserotation)*scale
        return rotation
       
    def unit_vector(self, coords : np.ndarray):
        #calculate unit vector and length of the element
        xi = coords[self.beam.c2//2 + 0] - coords[self.beam.c1//2 + 0]
        xj = coords[self.beam.c2//2 + 1] - coords[self.beam.c1//2 + 1]
        xk = coords[self.beam.c2//2 + 2] - coords[self.beam.c1//2 + 2]
        l = (xi**2 + xj**2 + xk**2)**0.5
        unit_vect = np.array([xi, xj, xk])/l
        return unit_vect,l
    
    def update_rotation_matrix(self, coords : np.ndarray):
        #determine arbitrary vector on plane xy to describe coordinate system along with vector x
        unit_vect = self.unit_vector(coords)[0]
        xi, xj ,xk = unit_vect[0], unit_vect[1], unit_vect[2]
        vxyi, vxyj, vxyk =  unit_vect[1], unit_vect[2], unit_vect[0]
        if xi == xj  and xj == xk:
            vxyi *= -1
        #update element rotation matrix in pyfe3d
        self.beam.update_rotation_matrix(vxyi, vxyj, vxyk, coords)
    
    def update_KC0(self, KC0r : np.ndarray, KC0c : np.ndarray, KC0v : np.ndarray, coords : np.ndarray):
        #update element rotation matrix and adds contribution to global stiffness matrix
        self.update_rotation_matrix(coords)
        self.beam.update_KC0(KC0r, KC0c, KC0v,self.prop,self.update_KC0v_only)
        #set flag to only update KC0v from now on
        self.update_KC0v_only = 1
        return KC0r, KC0c, KC0v
    
    def beam_internal_forces(self, displacement: np.ndarray, coords: np.ndarray,fi: np.ndarray):
        #Update rotation matrix
        self.update_rotation_matrix(coords)
        #Updating displacement vector
        self.beam.update_probe_ue(displacement)
        #Update properties EI and GJ to match the displacement using experimental relations
        self.update_inflatable_beam_properties()
        #Calculate internal forces
        self.beam.update_fint(fi,self.prop)
        return fi
        
        
        
