from src.problems.problem import Problem
from math import pi
import firedrake as fdrk
from firedrake.petsc import PETSc
from src.meshing.fischera_corner import fichera_corner

class MaxwellFichera(Problem):
    "Maxwell eigenproblem"
    def __init__(self, mesh_size):
        """Generate a mesh of a cube
        The boundary surfaces are numbered as follows:

        * 1: plane x == 0
        * 2: plane x == L
        * 3: plane y == 0
        * 4: plane y == L
        * 5: plane z == 0
        * 6: plane z == L
        """

        self.dim=3

        fichera_corner(mesh_size)

        self.domain = fdrk.Mesh("fichera_corner.msh")
    
        self.x, self.y, self.z = fdrk.SpatialCoordinate(self.domain)

        self.normal_versor = fdrk.FacetNormal(self.domain)

        self.bc_type = "electric"

        self.forcing = True
        self.material_coefficients = True


    def get_material_coefficients(self):
        return (fdrk.Constant(2), fdrk.Constant(3/2))

        
    def get_exact_solution(self, time: fdrk.Constant):


        exact_electric = fdrk.as_vector([fdrk.sin(2*time - 3*self.z), fdrk.sin(2*time - 3*self.x), fdrk.sin(2*time - 3*self.y)])
        exact_magnetic = fdrk.as_vector([fdrk.sin(2*time - 3*self.y), fdrk.sin(2*time - 3*self.z), fdrk.sin(2*time - 3*self.x)])

        return (exact_electric, exact_magnetic)
    


    def get_forcing(self, time):
        assert isinstance(time, fdrk.Constant)

        exact_current = fdrk.as_vector([fdrk.cos(2*time - 3*self.z), fdrk.cos(2*time - 3*self.x), fdrk.cos(2*time - 3*self.y)])
        return (exact_current, fdrk.Constant((0,0,0)))
    

    def get_initial_conditions(self):
        electric_field, magnetic_field = self.get_exact_solution(time = fdrk.Constant(0))

        return (electric_field, magnetic_field)


    def get_boundary_conditions(self, time: fdrk.Constant):
        """
        Parameters:
            time: time variable for inhomogeneous bcs

        Returns:
            bd_dict : dictionary of boundary conditions for the problem at hand
        """
        exact_electric, _ = self.get_exact_solution(time)

        null_bc = fdrk.Constant((0,0,0))
        bd_dict = {"electric": (["on_boundary"], exact_electric), "magnetic":([], null_bc)} 

        return bd_dict


    def __str__(self):
        return f"fichera_maxwell_bc_{self.bc_type}"