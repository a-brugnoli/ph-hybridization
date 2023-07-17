import firedrake as fdrk
from src.problems.eigensolution_wave import EigensolutionWave3D
from src.problems.eigensolution_maxwell import EigensolutionMaxwell3D
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver

import matplotlib.pyplot as plt
from src.postprocessing import basic_plotting
import math
from tqdm import tqdm

n_elements = 2**3
pol_degree = 1
time_step = 0.01
t_end = time_step
n_time_iter = math.ceil(t_end/time_step)


eigen_wave = EigensolutionWave3D(n_elements, n_elements, n_elements)
eigen_maxwell = EigensolutionMaxwell3D(n_elements, n_elements, n_elements)

case = "Wave"
if case=="Maxwell":
    problem = eigen_maxwell
else:
    problem = eigen_wave

mixedsolver_primal = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, \
                                    type_discretization="mixed", \
                                    type_formulation="primal", \
                                    type_system=case)

mixedsolver_dual = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, \
                                    type_discretization="mixed", \
                                    type_formulation="dual", \
                                    type_system=case)

hybridsolver_primal = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="primal", \
                                    type_system=case)

hybridsolver_dual = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="dual", \
                                    type_system=case)


for ii in tqdm(range(n_time_iter)):
    actual_time = (ii+1)*time_step

    mixedsolver_primal.integrate(time_step)
    mixedsolver_dual.integrate(time_step)

    hybridsolver_primal.integrate(time_step)
    hybridsolver_dual.integrate(time_step)
    
    mixedsolver_primal.update_variables()
    mixedsolver_dual.update_variables()

    hybridsolver_primal.update_variables()
    hybridsolver_dual.update_variables()



