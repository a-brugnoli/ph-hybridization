import firedrake as fdrk
from .solver import Solver
from src.problems.problem import Problem
from src.operators.maxwell_operators import MaxwellOperators
from src.operators.wave_operators import WaveOperators
from firedrake.petsc import PETSc
import gc

class HamiltonianWaveSolver(Solver):
    def __init__(self, 
                 problem: Problem, 
                 system,
                 time_step,
                 pol_degree=1, 
                 discretization="hybrid",
                 formulation="primal", 
                 solver_parameters={}, 
                 verbose=False
                ):
        """
        Constructor for the solver class using implicit Midpoint
        Parameters:
            problem (Problem) : a problem instance 
            pol_degree (int) : integer for the polynomial degree of the finite elements
            discretization (string) : "hybrid" or "mixed"
            formulation (string) :  "primal" or "dual" 
            solver_parameter (dictionary) : dictionary containing the solver parameter  
                polynomial degree (int), time step (float), final time (float)
        """

        self.problem = problem
        self.pol_degree = pol_degree
        self.solver_parameters = solver_parameters
        self.time_step = time_step
        self.verbose = verbose

        if system=="Maxwell":
            if discretization=="hybrid" :
                self.operators = MaxwellOperators(discretization, formulation, problem, pol_degree)
            elif discretization=="mixed":
                self.operators = MaxwellOperators(discretization, formulation, problem, pol_degree)
        elif system=="Wave":
            if discretization=="hybrid" :
                self.operators = WaveOperators(discretization, formulation, problem, pol_degree)
            elif discretization=="mixed":
                self.operators = WaveOperators(discretization, formulation, problem, pol_degree)
        else:
            ValueError(f"System type {system} is not a valid option")

        if self.verbose:
            PETSc.Sys.Print(f"{str(self.operators)}")
        self._set_spaces()
        self._set_initial_conditions()
        self._set_boundary_conditions()
        self._set_solver()


    def _set_spaces(self):
        self.space_operators = self.operators.fullspace
        self.tests = fdrk.TestFunctions(self.space_operators)
        self.trials = fdrk.TrialFunctions(self.space_operators)

        self.state_old = fdrk.Function(self.space_operators)
        self.state_new = fdrk.Function(self.space_operators)
        self.state_midpoint = fdrk.Function(self.space_operators)

        if self.verbose:
            PETSc.Sys.Print(f"Dimension of space: {self.space_operators.dim()} ")
                 

    def _set_initial_conditions(self):
        """
        Setup spaces, initial values, boundary conditions
        """
        expression_t0 = self.problem.get_initial_conditions()

        tuple_initial_conditions = self.operators.get_initial_conditions(expression_t0)

        for counter, field in enumerate(tuple_initial_conditions):
            self.state_old.sub(counter).assign(field)

        self.state_midpoint.assign(self.state_old)
        self.state_new.assign(self.state_old)

        self.time_old = fdrk.Constant(0)
        self.time_midpoint = fdrk.Constant(self.time_step/2)
        self.time_new = fdrk.Constant(self.time_step)
        self.actual_time = fdrk.Constant(0)

        if self.verbose:
            PETSc.Sys.Print(f"Inital conditions set")


    def _set_boundary_conditions(self):
        dict_essential_bcs = self.operators.essential_boundary_conditions(self.problem, time=self.time_new)

        self.space_bc = dict_essential_bcs["space"]
        self.value_bc = dict_essential_bcs["value"]
        self.list_id_bc = dict_essential_bcs["list_id"]

        self.essential_bcs = []
        for id_bc in self.list_id_bc:
            self.essential_bcs.append(fdrk.DirichletBC(self.space_bc, self.value_bc, id_bc))            

        self.natural_bcs = self.operators.natural_boundary_conditions(self.problem, time=self.time_midpoint)

        if self.verbose:
            PETSc.Sys.Print(f"Boundary conditions set")

    
    def _set_solver(self):
        states_old = self.state_old.subfunctions

        A_operator = self.operators.operator_implicit_midpoint(self.time_step, \
                    self.tests, self.trials)
        
        b_functional = self.operators.functional_implicit_midpoint(self.time_step, \
                    self.tests, states_old, control=self.natural_bcs)
        
        if self.problem.forcing:
            PETSc.Sys.Print("Problem with forcing term")

            # if self.verbose:
            #     PETSc.Sys.Print("Problem with forcing term")
            tuple_forcing = self.problem.get_forcing(self.time_midpoint)

            for counter, force in enumerate(tuple_forcing):
                PETSc.Sys.Print(f"Force {counter} : {force}")
                if force is not None:
                    b_functional += self.time_step*fdrk.inner(self.tests[counter], force)*fdrk.dx

        if self.operators.discretization=="mixed":

            linear_problem = fdrk.LinearVariationalProblem(A_operator, b_functional, self.state_new, bcs=self.essential_bcs)
            self.solver =  fdrk.LinearVariationalSolver(linear_problem, solver_parameters=self.solver_parameters)

        else:
            self.n_block_loc = self.operators.mixedspace_local.num_sub_spaces()
            _A = fdrk.Tensor(A_operator)
            _F = fdrk.Tensor(b_functional)
            # Extracting blocks for Slate expression of the reduced system
            self.A_blocks = _A.blocks
            self.F_blocks = _F.blocks

            self.A_global_operator = self.A_blocks[self.n_block_loc, self.n_block_loc] - self.A_blocks[self.n_block_loc, :self.n_block_loc] \
                * self.A_blocks[:self.n_block_loc, :self.n_block_loc].inv * self.A_blocks[:self.n_block_loc, self.n_block_loc]
            
            self.b_global_functional = self.F_blocks[self.n_block_loc] - self.A_blocks[self.n_block_loc, :self.n_block_loc] \
                * self.A_blocks[:self.n_block_loc, :self.n_block_loc].inv * self.F_blocks[:self.n_block_loc]

            # Global solver
            self.global_multiplier = fdrk.Function(self.operators.space_global)

            if "quadrilateral" in self.operators.cell_name and self.pol_degree>1:
            # if "quadrilateral" in self.operators.cell_name:
                if self.verbose:
                    PETSc.Sys.Print("Because of bug in DirichletBC, no solver set")
                pass
            else:
                linear_global_problem = fdrk.LinearVariationalProblem(self.A_global_operator, self.b_global_functional,\
                                                                      self.global_multiplier, bcs=self.essential_bcs)
                self.global_solver =  fdrk.LinearVariationalSolver(linear_global_problem, solver_parameters=self.solver_parameters)
                
                if self.verbose:
                    PETSc.Sys.Print(f"Solver set")

            

    def integrate(self):
        """
        Time step of the implicit midpoint (non linear)

        Parameters
            log_invariants (Boolean): if True logs all the invariants
            log_variables (Boolean): if True logs all the variables
        """

        if self.operators.discretization=="mixed":
            self.solver.solve()
        else:
            if "quadrilateral" in self.operators.cell_name and self.pol_degree>1:
            # if "quadrilateral" in self.operators.cell_name:
                # PETSc.Sys.Print("Projecting the boundary condition on the appropriate space")
                if isinstance(self.operators, WaveOperators):
                    if self.operators.formulation=="primal":
                        projected_value_bc = self.operators.project_RT_facet(self.value_bc, broken=False)
                    else:
                        projected_value_bc = self.operators.project_CG_facet(self.value_bc, broken=False)
                else:
                    projected_value_bc = self.operators.project_NED_facet(self.value_bc, broken=False)

                for iii in range(len(self.list_id_bc)):    
                    self.essential_bcs[iii].function_arg = projected_value_bc

                # PETSc.Sys.Print("Cleaning up memory")
                # gc.collect()
                # PETSc.garbage_cleanup(self.operators.domain._comm)
                # #PETSc.garbage_cleanup(self.operators.domain.comm)
                # PETSc.garbage_cleanup(PETSc.COMM_SELF)
                
            else:
                interpolated_value_bc = fdrk.interpolate(self.value_bc, self.space_bc)

                for iii in range(len(self.list_id_bc)):    
                    self.essential_bcs[iii].function_arg = interpolated_value_bc

                
            self.global_solver.solve()

            self._assemble_solution_hybrid()

        self.state_midpoint.assign(0.5*(self.state_new + self.state_old))
        self.actual_time.assign(self.time_new)
       

    def update_variables(self):
        self.state_old.assign(self.state_new)
        
        self.time_old.assign(self.actual_time)
        self.time_midpoint.assign(float(self.time_old) + self.time_step/2)
        self.time_new.assign(float(self.time_old) + self.time_step)
        


    def _assemble_solution_hybrid(self):
        if self.operators.discretization=="mixed":
            raise ValueError("Global to local assembly only valid for Hybrid system")

        # Intermediate expressions
        Lambda = fdrk.AssembledVector(self.global_multiplier)  # Local coefficient vector for Λ
        # Local solve expressions
        x_h = fdrk.assemble(self.A_blocks[:self.n_block_loc, :self.n_block_loc].inv *
                        (self.F_blocks[:self.n_block_loc] - self.A_blocks[:self.n_block_loc, self.n_block_loc] * Lambda))

        for ii in range(self.n_block_loc):
            self.state_new.sub(ii).assign(x_h.sub(ii))
        self.state_new.sub(self.n_block_loc).assign(self.global_multiplier)


    def dofs_essential_natural_bcs(self):
        """
        Extract dofs of essential and boundary conditions in Hybrid schemes
        """
        import numpy as np
        dofs_essential = []

        if self.operators.discretization=="mixed":
            raise ValueError(f" Function to extract dofs not defined for mixed discretization")
            

        for bc in self.essential_bcs:
            nodes_ess = bc.nodes

            dofs_essential = dofs_essential + list(nodes_ess)

        dofs_essential = list(set(dofs_essential))
        dofs_natural = list(set(self.operators.space_global.boundary_nodes("on_boundary")).difference(set(dofs_essential)))

        dofs_essential_offset = self.operators.mixedspace_local.dim() + np.array(dofs_essential)
        dofs_natural_offset = self.operators.mixedspace_local.dim() + np.array(dofs_natural)

        return dofs_essential_offset, dofs_natural_offset


