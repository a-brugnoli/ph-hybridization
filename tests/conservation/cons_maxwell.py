from src.problems.eigensolution_maxwell import EigensolutionMaxwell3D
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver

import matplotlib.pyplot as plt
from src.postprocessing import basic_plotting
from src.postprocessing.directory_figures import directories_problem
import math
from tqdm import tqdm
import firedrake as fdrk
import numpy as np
from mpi4py import MPI

n_elements = 4
pol_degree = 3
time_step = 0.01
t_end = 100*time_step
n_time_iter = math.ceil(t_end/time_step)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

eigen_maxwell = EigensolutionMaxwell3D(n_elements, n_elements, n_elements)
norm_versor = eigen_maxwell.normal_versor

time = fdrk.Constant(0)
exact_electric, exact_magnetic = eigen_maxwell.get_exact_solution(time)

hybridsolver_primal = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="primal")

hybridsolver_dual = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="dual")

# Exact quantities
exact_energy = 0.5 * (fdrk.inner(exact_electric, exact_electric) * fdrk.dx + fdrk.inner(exact_magnetic, exact_magnetic) * fdrk.dx)
exact_bdflow = -fdrk.dot(fdrk.cross(exact_electric, exact_magnetic), norm_versor) * fdrk.ds


# electric_primal_midpoint, magnetic_primal_midpoint = hybridsolver_primal.state_midpoint.subfunctions[0:2]
# electric_dual_midpoint, magnetic_dual_midpoint = hybridsolver_dual.state_midpoint.subfunctions[0:2]

# electric_primal_midpoint, magnetic_primal_midpoint = hybridsolver_primal.state_midpoint.subfunctions[0:2]
# electric_dual_midpoint, magnetic_dual_midpoint = hybridsolver_dual.state_midpoint.subfunctions[0:2]

electric_primal_midpoint, magnetic_primal_midpoint = hybridsolver_primal.state_midpoint.subfunctions[0:2]
electric_dual_midpoint, magnetic_dual_midpoint = hybridsolver_dual.state_midpoint.subfunctions[0:2]
# Power balance combining primal and dual
discrete_bdflow = -fdrk.dot(fdrk.cross(electric_dual_midpoint, magnetic_primal_midpoint), norm_versor) * fdrk.ds

discrete_energyrate_midpoint = 1 / time_step * (fdrk.dot(Enmid_1, En1_2 - En_2) * fdrk.dx + fdrk.dot(Hnmid_1, Hn1_2 - Hn_2) * fdrk.dx)


if rank==0:
    directory_results_problem, directory_results_problem_matplotlib, _ = directories_problem(eigen_maxwell)

    time_vec = np.linspace(time_step, time_step * n_time_iter, n_time_iter)

    powerbalance_conservation = np.zeros((n_time_iter, ))
    error_exact_inter_powerbalnce = np.zeros((n_time_iter, ))

    div_electric_field = np.zeros((n_time_iter, ))
    div_magnetic_field = np.zeros((n_time_iter, ))

    
for ii in tqdm(range(n_time_iter)):
    actual_time = (ii+1)*time_step

    hybridsolver_primal.integrate(time_step)
    hybridsolver_dual.integrate(time_step)


    hybridsolver_primal.update_variables()
    hybridsolver_dual.update_variables()


    
    if rank==0:
        time.assign(actual_time)


