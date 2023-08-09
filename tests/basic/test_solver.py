from tests.basic.debug_solver import debug_wave
import math
from src.problems.eigensolution_wave import EigensolutionWave3D
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver
from tqdm import tqdm
import firedrake as fdrk

n_elements = 3
pol_degree = 1

problem_wave_dirichlet = EigensolutionWave3D(n_elements, n_elements, n_elements, bc_type="dirichlet")

problem_wave_neumann = EigensolutionWave3D(n_elements, n_elements, n_elements, bc_type="neumann")

time_step = 0.001
t_end = 1

n_time_iter = math.ceil(t_end/time_step)

system = "Wave"

mixedsolver_primal = HamiltonianWaveSolver(problem = problem_wave_dirichlet, pol_degree=pol_degree, \
                                        time_step=time_step, \
                                        discretization="mixed", \
                                        formulation="primal", \
                                        system=system)

mixedsolver_dual = HamiltonianWaveSolver(problem = problem_wave_neumann, pol_degree=pol_degree, \
                                        time_step=time_step, \
                                        discretization="mixed", \
                                        formulation="dual", \
                                        system=system)

hybridsolver_primal = HamiltonianWaveSolver(problem = problem_wave_dirichlet, pol_degree=pol_degree, \
                                        time_step=time_step, \
                                        discretization="hybrid", \
                                        formulation="primal", \
                                        system=system)

hybridsolver_dual = HamiltonianWaveSolver(problem = problem_wave_neumann, pol_degree=pol_degree, \
                                        time_step=time_step, \
                                        discretization="hybrid", \
                                        formulation="dual", \
                                        system=system)

tol = 1e-9
for ii in tqdm(range(1,n_time_iter+1)):
    time_test = ii*time_step

    mixedsolver_primal.integrate()
    mixedsolver_dual.integrate()

    hybridsolver_primal.integrate()
    hybridsolver_dual.integrate()

    mixed_first_primal, mixed_second_primal = mixedsolver_primal.state_new.subfunctions
    hybrid_first_primal, hybrid_second_primal, _, _ = hybridsolver_primal.state_new.subfunctions
    
    mixed_first_dual, mixed_second_dual = mixedsolver_dual.state_new.subfunctions
    hybrid_first_dual, hybrid_second_dual, _, _ = hybridsolver_dual.state_new.subfunctions

    assert fdrk.norm(mixed_first_primal - hybrid_first_primal) < tol
    assert fdrk.norm(mixed_second_primal - hybrid_second_primal)  < tol

    assert fdrk.norm(mixed_first_dual - hybrid_first_dual)  < tol
    assert fdrk.norm(mixed_second_dual - hybrid_second_dual) < tol

    debug_wave(mixedsolver_primal, time_test)
    debug_wave(mixedsolver_dual, time_test)

    debug_wave(hybridsolver_primal, time_test)
    debug_wave(hybridsolver_dual, time_test)

    mixedsolver_primal.update_variables()
    hybridsolver_primal.update_variables()

    mixedsolver_dual.update_variables()
    hybridsolver_dual.update_variables()

