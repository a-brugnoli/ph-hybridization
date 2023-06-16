from src.problems.eigensolution_maxwell import EigensolutionMaxwell3D
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver

import matplotlib.pyplot as plt
from src.postprocessing import basic_plotting
from src.postprocessing.directory_figures import directories_problem
import math
from tqdm import tqdm
import firedrake as fdrk

save_fig = True
debug = True

n_elements = 2**3
pol_degree = 1
time_step = 0.01
t_end = time_step
n_time_iter = math.ceil(t_end/time_step)

eigen_maxwell = EigensolutionMaxwell3D(n_elements, n_elements, n_elements)

solver_params={}

mixedsolver_primal = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="mixed", \
                                    type_formulation="primal", \
                                    solver_parameters= solver_params)

mixedsolver_dual = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="mixed", \
                                    type_formulation="dual", \
                                    solver_parameters= solver_params)

hybridsolver_primal = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="primal", \
                                    solver_parameters= solver_params)

hybridsolver_dual = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="dual", \
                                    solver_parameters= solver_params)



for ii in tqdm(range(n_time_iter)):
    actual_time = (ii+1)*time_step

    mixedsolver_primal.integrate(time_step)
    mixedsolver_dual.integrate(time_step)
    
    mixedsolver_primal.update_variables()
    mixedsolver_dual.update_variables()



