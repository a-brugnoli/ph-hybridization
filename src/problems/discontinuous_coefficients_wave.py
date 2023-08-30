from src.problems.problem import Problem
from math import pi
import firedrake as fdrk

class DiscontinuousWave(Problem):
    "Maxwell eigenproblem"
    def __init__(self, n_elements_x, n_elements_y, quad=False):
        """Generate a mesh of a cube
        The boundary surfaces are numbered as follows:

        * 1: plane x == 0
        * 2: plane x == L
        * 3: plane y == 0
        * 4: plane y == L
        * 5: plane z == 0
        * 6: plane z == L
        """

        self.quad = quad
        self.dim = 2
        self.domain = fdrk.RectangleMesh(nx=n_elements_x, ny=n_elements_y, Lx=3, Ly=2,  quadrilateral=quad)
            
        self.x, self.y = fdrk.SpatialCoordinate(self.domain)
        
        self.normal_versor = fdrk.FacetNormal(self.domain)

        self.forcing = True
        self.material_coefficients = True


    def get_material_coefficients(self):

        pressure_coefficient = fdrk.conditional(fdrk.le(self.x, 1), 10, 1) 

        velocity_coefficent = 1

        return (pressure_coefficient, velocity_coefficent)
       
    
    def get_exact_solution(self):
        pass


    def get_forcing(self, time):
        assert isinstance(time, fdrk.Constant)

        force_pressure =  fdrk.conditional(fdrk.And(fdrk.And(fdrk.gt(self.x, 1.2), fdrk.lt(self.x, 1.4)), fdrk.lt(time, 0.2)), 1, 0) 
        return (force_pressure, fdrk.Constant((0,) * self.dim))
    

    def get_initial_conditions(self):
        pressure_field, velocity_field = fdrk.Constant(0), fdrk.Constant((0, 0))

        return (pressure_field, velocity_field)
    
    
    def get_boundary_conditions(self, time: fdrk.Constant):
        """
        Parameters:
            time: time variable for inhomogeneous bcs

        Returns:
            bd_dict : dictionary of boundary conditions for the problem at hand
        """

        null_bc = fdrk.Constant(0)

        null_bc_vec = fdrk.Constant((0,) * self.dim)

        bd_dict = {"dirichlet": (["on_boundary"], null_bc), "neumann":([], null_bc_vec)} 
        
        return bd_dict
    

    def __str__(self):
        return f"wave_discontinuous"