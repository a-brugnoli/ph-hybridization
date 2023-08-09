import firedrake as fdrk
from abc import ABC, abstractmethod


class Problem(ABC):
    def __init__(self):
        self.domain = None
        self.x, self.y, self.z = None, None, None
        self.bc_type=None
        self.forcing = False
        self.dim=None
        self.normal_versor = None



    @abstractmethod
    def get_initial_conditions(self):
        pass


    @abstractmethod
    def get_exact_solution(self, time: fdrk.Constant):
        pass
    

    @abstractmethod
    def get_forcing(self, time: fdrk.Constant):
        pass

    
    @abstractmethod
    def get_boundary_conditions(self, time: fdrk.Constant):
        pass