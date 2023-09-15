from src.problems.problem import Problem
from math import pi
import firedrake as fdrk
from firedrake.petsc import PETSc

class EigensolutionWave(Problem):
    "Maxwell eigenproblem"
    def __init__(self, n_elements_x, n_elements_y, n_elements_z, bc_type="mixed", dim=3, quad=False):
        """Generate a mesh of a cube
        The boundary surfaces are numbered as follows:

        * 1: plane x == 0
        * 2: plane x == L
        * 3: plane y == 0
        * 4: plane y == L
        * 5: plane z == 0
        * 6: plane z == L
        """

        self.dim=dim
        self.quad = quad

        if dim==3:
            if quad:
                PETSc.Sys.Print("Hexahedral mesh requested")
                quad_mesh = fdrk.UnitSquareMesh(n_elements_x, n_elements_y, quadrilateral=quad)
                self.domain = fdrk.ExtrudedMesh(quad_mesh, layers=n_elements_z)

            else:
                self.domain = fdrk.UnitCubeMesh(nx=n_elements_x, 
                                            ny=n_elements_y, 
                                            nz=n_elements_z)
            
            self.x, self.y, self.z = fdrk.SpatialCoordinate(self.domain)
        elif dim==2:
            if quad:
                PETSc.Sys.Print("Quadrilateral mesh requested")

            self.domain = fdrk.UnitSquareMesh(nx=n_elements_x, 
                                              ny=n_elements_y, 
                                              quadrilateral=quad)
            
            self.x, self.y = fdrk.SpatialCoordinate(self.domain)
        else:
            PETSc.Sys.Print("Invalid dimension")

        self.bc_type = bc_type

        self.normal_versor = fdrk.FacetNormal(self.domain)

        self.forcing = True

        self.material_coefficients = False


    def get_material_coefficients(self):
        return (fdrk.Constant(1), fdrk.Constant(1))


    def get_exact_solution(self, time: fdrk.Constant):
        omega_space = 1
        if self.dim==3:
            g_fun = fdrk.sin(omega_space * self.x) * fdrk.sin(omega_space * self.y) * fdrk.sin(omega_space * self.z)
        else:
            g_fun = fdrk.sin(omega_space * self.x) * fdrk.sin(omega_space * self.y)
        grad_g = fdrk.grad(g_fun)

        ft, dft = self._get_time_function(time)
        exact_pressure = g_fun * dft
        exact_velocity = grad_g * ft

    
        return (exact_pressure, exact_velocity)
    

    def get_forcing(self, time):
        assert isinstance(time, fdrk.Constant)

        exact_pressure, exact_velocity = self.get_exact_solution(time)

        force_pressure = fdrk.diff(exact_pressure, time) - fdrk.div(exact_velocity)

        force_velocity = fdrk.diff(exact_velocity, time) - fdrk.grad(exact_pressure)

        return (force_pressure, force_velocity)
    

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

        null_bc = fdrk.Constant(0)

        null_bc_vec = fdrk.Constant((0,) * self.dim)

        if self.bc_type == "dirichlet":
            if self.dim==3 and self.quad:
                bd_dict = {"dirichlet": (["on_boundary", "top", "bottom"], exact_pressure), "neumann":([], null_bc_vec)} 
            else:
                bd_dict = {"dirichlet": (["on_boundary"], exact_pressure), "neumann":([], null_bc_vec)} 
        elif self.bc_type == "neumann":
            if self.dim==3 and self.quad:
                bd_dict = {"dirichlet": ([], null_bc), "neumann": (["on_boundary", "top", "bottom"], exact_velocity)}
            else:
                bd_dict = {"dirichlet": ([], null_bc), "neumann": (["on_boundary"], exact_velocity)}

        elif self.bc_type == "mixed":
            if self.dim==3:
                if self.quad:
                    bd_dict = {"dirichlet": ([1,3,"bottom"], exact_pressure), "neumann": ([2,4,"top"], exact_velocity)}
                else:
                    bd_dict = {"dirichlet": ([1,3,5], exact_pressure), "neumann": ([2,4,6], exact_velocity)}
            else:
                bd_dict = {"dirichlet": ([1,3], exact_pressure), "neumann": ([2,4], exact_velocity)}

        else:
            raise ValueError(f"{self.bc_type} is not a valid value for bc")
        
        return bd_dict
    

    def __str__(self):
        return f"eigensolution_wave_{self.dim}d_bc_{self.bc_type}"