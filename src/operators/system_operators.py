from .spaces_deRham import deRhamElements, deRhamSpaces
import firedrake as fdrk
from abc import ABC, abstractmethod

class SystemOperators(ABC):
    def __init__(self, type_discretization, type_formulation, domain: fdrk.MeshGeometry, pol_degree):
        """
        Constructor for the MaxwellOperators class
        Parameters
            type (string) : "primal" or "dual", the kind of discretization (primal is u1 or B2)
            reynold (float) : the reciprocal of the magnetic Reynolds number
        """
        
        if type_discretization!="mixed" and type_discretization!="hybrid":
            raise ValueError(f"Discretization type {type_discretization} is not a valid value")
        
        if type_formulation!="primal" and type_formulation!="dual":
            raise ValueError(f"Formulation type {type_formulation} is not a valid value")

        self.type_discretization=type_discretization
        self.type_formulation=type_formulation

        self.domain = domain
        self.pol_degree = pol_degree
        self.normal_versor = fdrk.FacetNormal(self.domain)

        self.CG_element, self.NED_element, self.RT_element, self.DG_element = \
            deRhamElements(domain, pol_degree).values()
        
        self.CG_space, self.NED_space, self.RT_space, self.DG_space = \
            deRhamSpaces(domain, pol_degree).values()

        self._set_space()


    @abstractmethod
    def _set_space(self):
        self.fullspace=None


    @abstractmethod
    def get_initial_conditions():
        pass


    @abstractmethod
    def essential_boundary_conditions():
        pass

    @abstractmethod
    def natural_boundary_conditions():
        pass


    @abstractmethod
    def dynamics():
        pass


    @abstractmethod
    def control():
        pass


    def operator_implicit_midpoint(self, time_step, testfunctions, trialfunctions):
        """
        Construct operators arising from the implicit midpoint discretization
        A x = b
        """
        mass_operator, dynamics_operator = \
                        self.dynamics(testfunctions, trialfunctions)
        
        lhs_operator = mass_operator - 0.5 * time_step * dynamics_operator
        
        return lhs_operator
    

    def functional_implicit_midpoint(self, time_step, testfunctions, functions, control):

        mass_functional, dynamics_functional = self.dynamics(testfunctions, functions)

        natural_control = self.control(testfunctions, control)

        rhs_functional = mass_functional + 0.5 * time_step * dynamics_functional \
                                    + time_step * natural_control
        
        return rhs_functional
    
    @abstractmethod
    def __str__(self) -> str:
        pass