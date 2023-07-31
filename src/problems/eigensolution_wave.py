from src.problems.problem import Problem
from math import pi
import firedrake as fdrk
from firedrake.petsc import PETSc

class EigensolutionWave3D(Problem):
    "Maxwell eigenproblem"
    def __init__(self, n_elements_x, n_elements_y, n_elements_z, bc_type="mixed"):
        """Generate a mesh of a cube
        The boundary surfaces are numbered as follows:

        * 1: plane x == 0
        * 2: plane x == L
        * 3: plane y == 0
        * 4: plane y == L
        * 5: plane z == 0
        * 6: plane z == L
        """

        self.domain = fdrk.UnitCubeMesh(nx=n_elements_x, \
                                        ny=n_elements_y, \
                                        nz=n_elements_z)
        self.x, self.y, self.z = fdrk.SpatialCoordinate(self.domain)

        self.bc_type = bc_type

        self.normal_versor = fdrk.FacetNormal(self.domain)

        PETSc.Sys.Print(str(self))
       

    def get_exact_solution(self, time: fdrk.Constant):
        om = 1
        om_t = fdrk.sqrt(3*om ** 2)

        ft = 2 * fdrk.sin(om_t * time) + 3 * fdrk.cos(om_t * time)
        dft = fdrk.diff(ft, time) # om_t * (2 * cos(om_t * time) - 3 * sin(om_t * time))

        g_fun = fdrk.cos(om * self.x) * fdrk.sin(om * self.y) * fdrk.sin(om * self.z)

        grad_g = fdrk.grad(g_fun)

        exact_pressure = g_fun * dft
        exact_velocity = grad_g * ft
        return (exact_pressure, exact_velocity)
    

    def get_initial_conditions(self):
        pressure_field, velocity_field = self.get_exact_solution(time = fdrk.Constant(0))

        return (pressure_field, velocity_field)


    def get_boundary_conditions(self, time: fdrk.Constant):
        """
        Parameters:
            time: time variable for inhomogeneous bcs

        Returns:
            bd_dict : dictionary of boundary conditions for the problem at hand
        """
        exact_pressure, exact_velocity = self.get_exact_solution(time)

        null_bc = fdrk.Constant((0,0,0))
        if self.bc_type == "dirichlet":
            bd_dict = {"dirichlet": (["on_boundary"], exact_pressure), "neumann":([], null_bc)} 
        elif self.bc_type == "neumann":
            bd_dict = {"dirichlet": ([], null_bc), "neumann": (["on_boundary"], exact_velocity)}
        elif self.bc_type == "mixed":
            bd_dict = {"dirichlet": ([1,3,5], exact_pressure), "neumann": ([2,4,6], exact_velocity)}
        else:
            raise ValueError(f"{self.bc_type} is not a valid value for bc")
        
        return bd_dict


    def __str__(self):
        return f"Eigensolution Wave 3D. Boundary conditions {self.bc_type}"