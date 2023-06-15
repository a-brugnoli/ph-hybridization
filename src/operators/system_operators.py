from .spaces_deRham import deRhamElements, deRhamSpaces
import firedrake as fdrk

class SystemOperators:
    def __init__(self, type, domain: fdrk.MeshGeometry, pol_degree):
        """
        Constructor for the MaxwellOperators class
        Parameters
            type (string) : "primal" or "dual", the kind of discretization (primal is u1 or B2)
            reynold (float) : the reciprocal of the magnetic Reynolds number
        """
        self.type=type
        self.domain = domain
        self.pol_degree = pol_degree

        self.CG_element, self.NED_element, self.RT_element, self.DG_element = \
            deRhamElements(domain, pol_degree).values()
        
        self.CG_space, self.NED_space, self.RT_space, self.DG_space = \
            deRhamSpaces(domain, pol_degree).values()

        self._set_space()


    def _set_space(self):
        self.fullspace=None


    def get_initial_conditions():
        pass


    def essential_boundary_conditions():
        pass


    def dynamics():
        pass


    def control():
        pass


    def operator_implicit_midpoint(self, time_step, testfunctions, trialfunctions):
        """
        Construct operators arising from the implicit midpoint discretization
        A x = b
        """
        mass_operator, dynamics_operator = \
                        self.dynamics(self, testfunctions, trialfunctions)
        
        lhs_operator = mass_operator - 0.5 * time_step * dynamics_operator
        
        return lhs_operator
    

    def functional_implicit_midpoint(self, time_step, testfunctions, functions, control):

        mass_functional, dynamics_functional = self.dynamics(self, testfunctions, functions)

        natural_control = self.control(testfunctions, control)

        rhs_functional = mass_functional + 0.5 * time_step * dynamics_functional \
                                    + time_step * natural_control
        
        return rhs_functional
    

    def __str__(self) -> str:
        pass