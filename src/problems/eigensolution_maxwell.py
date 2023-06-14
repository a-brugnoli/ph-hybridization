from src.problems.problem import Problem
from math import pi
import firedrake as fdrk

class EigensolutionMaxwell3D(Problem):
    "Maxwell eigenproblem"
    def __init__(self, n_elements_x, n_elements_y, n_elements_z):
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
        

    def get_exact_solution(self, time: fdrk.Constant):
        om_x = 1
        om_y = 1
        om_z = 1

        om_t = fdrk.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)

        ft = fdrk.sin(om_t * time) / om_t
        dft = fdrk.cos(om_t * time)

        g_x = -fdrk.cos(om_x * self.x) * fdrk.sin(om_y * self.y) * fdrk.sin(om_z * self.z)
        g_y = fdrk.Constant(0.0)
        g_z = fdrk.sin(om_x * self.x) * fdrk.sin(om_y * self.y) * fdrk.cos(om_z * self.z)

        g_fun = fdrk.as_vector([g_x,
                                g_y,
                                g_z])

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


    def get_boundary_conditions(self, time: fdrk.Constant, case="mixed"):
        """
        Parameters:
            case (string): type of boundary condition ("electric, magnetic, mixed")

        Returns:
            bd_dict : dictionary of boundary conditions for the problem at hand
        """
        exact_electric, exact_magnetic = self.get_exact_solution(time)
        if case == "electric":
            bd_dict = {"electric": ["on_boundary"], "magnetic":[]} 
        elif case == "magnetic":
            bd_dict = {"electric": [], "magnetic": ["on_boundary"]}
        elif case == "mixed":
            bd_dict = {"electric": [2,4,6], "magnetic":[1,3,5]}
        
        return bd_dict, exact_electric, exact_magnetic


    def __str__(self):
        return "EigensolutionMaxwell3D"