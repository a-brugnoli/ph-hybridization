from src.problems.eigensolution_maxwell import EigensolutionMaxwell3D
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver

from src.postprocessing.directory_figures import directories_problem
import math
from tqdm import tqdm
import firedrake as fdrk
from firedrake.petsc import PETSc
from mpi4py import MPI


def compute_error(n_elements, pol_degree, time_step=0.01, t_end=1, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()

    n_time_iter = math.ceil(t_end/time_step)

    problem = EigensolutionMaxwell3D(n_elements, n_elements, n_elements)

    directory_figures_problem, directory_figures_problem_matplotlib, directory_figures_problem_paraview = directories_problem(str(problem))
   
    solver_params = {}


    hybridsolver_primal = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, \
                                        type_discretization="hybrid", \
                                        type_formulation="primal", \
                                        solver_parameters= solver_params)

    hybridsolver_dual = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="dual", \
                                    solver_parameters= solver_params)


    for ii in tqdm(range(n_time_iter)):
        actual_time = (ii+1)*time_step

        hybridsolver_primal.integrate(time_step)
        hybridsolver_dual.integrate(time_step)
        
        hybridsolver_primal.update_variables()
        hybridsolver_dual.update_variables()
    
    PETSc.Sys.Print(f"Solution with {n_elements} elements and degree {pol_degree} computed")

    hybrid_electric_primal, hybrid_magnetic_primal = hybridsolver_primal.state_old.subfunctions[0:2]
    hybrid_electric_dual, hybrid_magnetic_dual = hybridsolver_dual.state_old.subfunctions[0:2]

    exact_electric, exact_magnetic = problem.get_exact_solution(fdrk.Constant(actual_time))

    