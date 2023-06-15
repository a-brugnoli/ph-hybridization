import firedrake as fdrk
from .solver import Solver
from src.problems.problem import Problem
from src.operators.maxwell_hybrid import MaxwellHybrid
from src.operators.maxwell_mixed import MaxwellMixed
from firedrake.petsc import PETSc

class HamiltonianWaveSolver(Solver):
    def __init__(self, 
                 problem: Problem, 
                 pol_degree=1, 
                 type_system="Maxwell",
                 type_discretization="hybrid",
                 type_formulation="primal", solver_parameters={}):
        """
        Constructor for the solver class 
        Parameters:
            problem (Problem) : a problem instance 
            pol_degree (int) : integer for the polynomial degree of the finite elements
            type_discretization (string) : "hybrid" or "mixed"
            type_formulation (string) :  "primal" or "dual" 
            solver_parameter (dictionary) : dictionary containing the solver parameter  
                polynomial degree (int), time step (float), final time (float)
        """

        self.problem = problem
        self.pol_degree = pol_degree
        self.solver_parameters = solver_parameters
        self.type_discretization = type_discretization

        if type_system=="Maxwell":
            if type_discretization=="hybrid" :
                self.operators = MaxwellHybrid(type_formulation, problem.domain, pol_degree)
            elif type_discretization=="mixed":
                self.operators = MaxwellMixed(type_formulation, problem.domain, pol_degree)
            else:
                raise ValueError(f"The discretization {type_discretization} does not exist")
        else:
            ValueError(f"The system {type_system} does not exist")

        self._set_spaces()
        self._set_initial_conditions()
        self._set_boundary_conditions()


    def _set_spaces(self):
        self.space_operators = self.operators.fullspace
        self.tests = fdrk.TestFunctions(self.space_operators)
        self.trials = fdrk.TrialFunctions(self.space_operators)

        self.state_old = fdrk.Function(self.space_operators)
        self.state_new = fdrk.Function(self.space_operators)
        self.state_midpoint = fdrk.Function(self.space_operators)

        PETSc.Sys.Print(f"Dimension of space for {str(self.operators)}: \
                        {self.space_operators.dim()} ")
                 

    def _set_initial_conditions(self):
        """
        Setup spaces, initial values, boundary conditions
        """
        (first_field_at0, second_field_at0) = self.problem.get_initial_conditions()

        tuple_initial_conditions = self.operators.assign_initial_conditions(self, first_field_at0, second_field_at0)

        for counter, field in enumerate(tuple_initial_conditions):
            self.state_old.sub(counter).assign(field)

        self.state_new.assign(self.state_old)
        self.time_system = fdrk.Constant(0)
        
        PETSc.Sys.Print("Inital conditions set")
        PETSc.Sys.Print("\n")


    def _set_boundary_conditions(self, time:fdrk.Constant):
        self.bcs_list = self.operators.essential_boundary_conditions(self. problem, time)

        PETSc.Sys.Print("Boundary conditions set")  
        PETSc.Sys.Print("\n")


    def integrate(self, time_step):
        """
        Time step of the implicit midpoint (non linear)

        Parameters
            log_invariants (Boolean): if True logs all the invariants
            log_variables (Boolean): if True logs all the variables
        """

        time_midpoint = fdrk.Constant(float(self.time_system) + time_step/2)

        first_field_midpoint, second_field_midpoint = \
            self.problem.get_exact_solution(time_midpoint)

        
        states_old = self.state_old.subfunctions
        states_new = fdrk.split(self.state_new)

        # TO BE DONE
        A_operator = self.operators.operator_implicit_midpoint(self, time_step, \
                    self.tests, self.trials)
        
        b_functional = self.operators.functional_implicit_midpoint(self, time_step, \
                    self.tests, states_old, control="TO BE DONE")

        linear_problem = fdrk.LinearVariationalProblem(A_operator, b_functional, self.state_new, bcs=self.bcs_list)
        solver =  fdrk.LinearVariationalSolver(linear_problem, solver_parameters=self.solver_parameters)

        solver.solve()

        self.state_midpoint.assign(0.5*(self.state_new + self.state_old))
        
        
        # Compute invariants before new assign
        if log_invariants: 
            dict_invariants = self.compute_invariants()
        else: dict_invariants = None

        return dict_invariants
    

    def update_variables(self, actual_time):
        self.time.assign(float(actual_time))
        self.state_old.assign(self.state_new)
