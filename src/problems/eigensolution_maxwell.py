from src.problems.problem import Problem
from math import pi
import firedrake as fdrk

class EigensolutionMaxwell3D(Problem):
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

        

    def get_exact_solution(self, time: fdrk.Constant):
        om_x, om_y, om_z = 1, 1, 1
        om_t = fdrk.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)

        ft = fdrk.sin(om_t * time) / om_t
        dft = fdrk.cos(om_t * time)

        g_x = -fdrk.cos(om_x * self.x) * fdrk.sin(om_y * self.y) * fdrk.sin(om_z * self.z)
        g_y = fdrk.Constant(0.0)
        g_z = fdrk.sin(om_x * self.x) * fdrk.sin(om_y * self.y) * fdrk.cos(om_z * self.z)

        g_fun = fdrk.as_vector([g_x, g_y, g_z])

        curl_g = fdrk.as_vector([om_y * fdrk.sin(om_x * self.x) * fdrk.cos(om_y * self.y) \
                                  * fdrk.cos(om_z * self.z),
                            -(om_x + om_z) * fdrk.cos(om_x * self.x) \
                                * fdrk.sin(om_y * self.y) * fdrk.cos(om_z * self.z),
                            om_y * fdrk.cos(om_x * self.x) * fdrk.cos(om_y * self.y) \
                                * fdrk.sin(om_z * self.z)])
        # curl_g = curl(g_fun)

        exact_electric = g_fun * dft
        exact_magnetic = -curl_g * ft
        return (exact_electric, exact_magnetic)
    

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
        exact_electric, exact_magnetic = self.get_exact_solution(time)

        null_bc = fdrk.Constant((0,0,0))
        if self.bc_type == "electric":
            bd_dict = {"electric": (["on_boundary"], exact_electric), "magnetic":([], null_bc)} 
        elif self.bc_type == "magnetic":
            bd_dict = {"electric": ([], null_bc), "magnetic": (["on_boundary"], exact_magnetic)}
        elif self.bc_type == "mixed":
            bd_dict = {"electric": ([2,4,6], exact_electric), "magnetic": ([1,3,5], exact_magnetic)}
        else:
            raise ValueError(f"{self.bc_type} is not a valid value for bc")
        
        return bd_dict


    def __str__(self):
        return "EigensolutionMaxwell3D"