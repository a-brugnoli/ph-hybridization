from .spaces_deRham import deRhamElements, deRhamSpaces


class SystemOperators:
    def __init__(self, type, domain, pol_degree):
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


    def _set_space():
        pass


    def __str__(self) -> str:
        pass