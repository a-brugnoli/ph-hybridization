import firedrake as fdrk
from .solver import Solver
from src.problems.problem import Problem
from src.operators.maxwell_operators import MaxwellOperators
from firedrake.petsc import PETSc

class HamiltonianWaveSolver(Solver):
    def __init__(self, 
                 problem: Problem, 
                 pol_degree=1, 
                 type_system="Maxwell",
                 type_discretization="hybrid",
                 type_formulation="primal", 
                 solver_parameters={'ksp_type': 'preonly',
                                    'pc_type': 'lu',
                                    'pc_factor_mat_solver_type': 'mumps'}
                ):
        """
        Constructor for the solver class using implicit Midpoint
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

        if type_system=="Maxwell":
            if type_discretization=="hybrid" :
                self.operators = MaxwellOperators(type_discretization, type_formulation, problem.domain, pol_degree)
            elif type_discretization=="mixed":
                self.operators = MaxwellOperators(type_discretization, type_formulation, problem.domain, pol_degree)
            else:
                raise ValueError(f"The discretization {type_discretization} does not exist")
        else:
            ValueError(f"System type {type_system} is not a valid option")

        self.time_old = fdrk.Constant(0)
        self.time_midpoint = fdrk.Constant(0)
        self.time_new = fdrk.Constant(0)
        self.actual_time = fdrk.Constant(0)

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

        PETSc.Sys.Print(f"Dimension of space for {str(self.operators)}: {self.space_operators.dim()} ")
                 

    def _set_initial_conditions(self):
        """
        Setup spaces, initial values, boundary conditions
        """
        expression_t0 = self.problem.get_initial_conditions()

        tuple_initial_conditions = self.operators.get_initial_conditions(expression_t0)

        for counter, field in enumerate(tuple_initial_conditions):
            self.state_old.sub(counter).assign(field)

        PETSc.Sys.Print(f"System {self.operators.type_discretization} {self.operators.type_formulation}: inital conditions set")


    def _set_boundary_conditions(self):
        self.essential_bcs = self.operators.essential_boundary_conditions(self.problem, time=self.time_new)

        self.natural_bcs = self.operators.natural_boundary_conditions(self.problem, time=self.time_midpoint)

        PETSc.Sys.Print(f"System {self.operators.type_discretization} {self.operators.type_formulation}: boundary conditions set")


    def integrate(self, time_step):
        """
        Time step of the implicit midpoint (non linear)

        Parameters
            log_invariants (Boolean): if True logs all the invariants
            log_variables (Boolean): if True logs all the variables
        """

        self.time_midpoint.assign(float(self.time_old) + time_step/2)
        self.time_new.assign(float(self.time_old) + time_step)
        
        states_old = self.state_old.subfunctions

        A_operator = self.operators.operator_implicit_midpoint(time_step, \
                    self.tests, self.trials)
        
        b_functional = self.operators.functional_implicit_midpoint(time_step, \
                    self.tests, states_old, control=self.natural_bcs)

        if self.operators.type_discretization=="mixed":
            linear_problem = fdrk.LinearVariationalProblem(A_operator, b_functional, self.state_new, bcs=self.essential_bcs)
            solver =  fdrk.LinearVariationalSolver(linear_problem, solver_parameters=self.solver_parameters)

            solver.solve()
        else:
            self.state_new = self.solve_hybrid(A_operator, b_functional, bcs=self.essential_bcs, \
                                               global_space=self.operators.space_global,
                                               mixed_local_space=self.operators.mixedspace_local)

        self.state_midpoint.assign(0.5*(self.state_new + self.state_old))
        self.actual_time.assign(self.time_new)
       

    def update_variables(self):
        self.state_old.assign(self.state_new)
        self.time_old.assign(self.actual_time)




    def solve_hybrid(self, a_form, b_form, bcs, global_space: fdrk.FunctionSpace, mixed_local_space : fdrk.MixedFunctionSpace):

        n_block_loc = mixed_local_space.num_sub_spaces()
        _A = fdrk.Tensor(a_form)
        _F = fdrk.Tensor(b_form)
        # Extracting blocks for Slate expression of the reduced system
        A = _A.blocks
        F = _F.blocks
        S = A[n_block_loc, n_block_loc] - A[n_block_loc, :n_block_loc] * A[:n_block_loc, :n_block_loc].inv * A[:n_block_loc, n_block_loc]
        E = F[n_block_loc] - A[n_block_loc, :n_block_loc] * A[:n_block_loc, :n_block_loc].inv * F[:n_block_loc]

        # Assemble and solve: SΛ = E
        Smat = fdrk.assemble(S, bcs=bcs)
        Evec = fdrk.assemble(E)
        lambda_h = fdrk.Function(global_space)
        fdrk.solve(Smat, lambda_h, Evec, solver_parameters=self.solver_parameters)

        
        # Intermediate expressions
        Lambda = fdrk.AssembledVector(lambda_h)  # Local coefficient vector for Λ
        # Local solve expressions
        x_h = fdrk.assemble(A[:n_block_loc, :n_block_loc].inv *
                        (F[:n_block_loc] - A[:n_block_loc, n_block_loc] * Lambda))

        # x_sys = A[:n_block_loc, :n_block_loc].solve(F[:n_block_loc] - A[:n_block_loc, n_block_loc] * Lambda,\
        #                                             decomposition="PartialPivLU")
        # x_h = Function(mixed_local_space)  # Function to store the result: x_loc
        # assemble(x_sys, x_h)

        sol = fdrk.Function(self.space_operators)
        for ii in range(n_block_loc):
            sol.sub(ii).assign(x_h.sub(ii))
        sol.sub(n_block_loc).assign(lambda_h)

        return sol


