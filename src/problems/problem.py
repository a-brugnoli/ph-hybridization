import firedrake as fdrk
class Problem:
    def __init__(self):
        self.domain = None
        self.x, self.y, self.z = None, None, None
        
    
    def get_initial_conditions(self):
        pass


    def get_exact_solution(self, time: fdrk.Constant):
        pass
    

    def get_boundary_conditions(self, time: fdrk.Constant):
        pass