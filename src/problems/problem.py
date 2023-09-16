import firedrake as fdrk
from abc import ABC, abstractmethod
from math import pi

class Problem(ABC):
    def __init__(self):
        self.domain = None
        self.x, self.y, self.z = None, None, None
        self.bc_type=None
        self.forcing = None
        self.dim=None
        self.normal_versor = None
        self.quad = None
        self.material_coefficients = None



    @abstractmethod
    def get_initial_conditions(self):
        pass


    @abstractmethod
    def get_exact_solution(self, time: fdrk.Constant):
        pass
    
    @abstractmethod    
    def get_material_coefficients(self):
        pass

    @abstractmethod
    def get_forcing(self, time: fdrk.Constant):
        pass

    
    @abstractmethod
    def get_boundary_conditions(self, time: fdrk.Constant):
        pass

    def _get_manufactured_time_function(self, time: fdrk.Constant):
        f_time = 1/2*time**2

        df_dtime = fdrk.diff(f_time, time)
        return f_time, df_dtime
    

    def _get_eigensolution_time_function(self, time: fdrk.Constant, omega):
        f_time = fdrk.sin(omega*time) + fdrk.cos(omega*time)

        df_dtime = fdrk.diff(f_time, time)
        return f_time, df_dtime
    