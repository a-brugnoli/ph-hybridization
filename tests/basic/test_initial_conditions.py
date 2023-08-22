from tests.basic.debug_solver import debug_wave
import math
from src.problems.eigensolution_wave import EigensolutionWave
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver
import firedrake as fdrk

n_elements = 3
pol_degree = 1

quad = True
problem_wave = EigensolutionWave(n_elements, n_elements, n_elements, quad=quad, dim=2, bc_type="dirichlet")

time_step =0.001
t_end = 10*time_step

n_time_iter = math.ceil(t_end/time_step)

system = "Wave"

mixedsolver_primal = HamiltonianWaveSolver(problem = problem_wave, pol_degree=pol_degree, \
                                        time_step=time_step, \
                                        discretization="mixed", \
                                        formulation="primal", \
                                        system=system)

mixedsolver_dual = HamiltonianWaveSolver(problem = problem_wave, pol_degree=pol_degree, \
                                        time_step=time_step, \
                                        discretization="mixed", \
                                        formulation="dual", \
                                        system=system)

hybridsolver_primal = HamiltonianWaveSolver(problem = problem_wave, pol_degree=pol_degree, \
                                        time_step=time_step, \
                                        discretization="hybrid", \
                                        formulation="primal", \
                                        system=system)

hybridsolver_dual = HamiltonianWaveSolver(problem = problem_wave, pol_degree=pol_degree, \
                                        time_step=time_step, \
                                        discretization="hybrid", \
                                        formulation="dual", \
                                        system=system)

mixed_first_primal, mixed_second_primal = mixedsolver_primal.state_old.subfunctions
hybrid_first_primal, hybrid_second_primal, _, _ = hybridsolver_primal.state_old.subfunctions

mixed_first_dual, mixed_second_dual = mixedsolver_dual.state_old.subfunctions
hybrid_first_dual, hybrid_second_dual, _, _ = hybridsolver_dual.state_old.subfunctions

errorvalue_first_primal = fdrk.norm(mixed_first_primal - hybrid_first_primal)
errorvalue_second_primal = fdrk.norm(mixed_second_primal - hybrid_second_primal)

errorvalue_first_dual = fdrk.norm(mixed_first_dual - hybrid_first_dual)
errorvalue_second_dual = fdrk.norm(mixed_second_dual - hybrid_second_dual)

print(f"Error at 0 first primal: {errorvalue_first_primal}")
print(f"Error at 0 second primal: {errorvalue_second_primal}")
print(f"Error at 0 first dual: {errorvalue_first_dual}")
print(f"Error at 0 second dual: {errorvalue_second_dual}")

