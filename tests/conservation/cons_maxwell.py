from src.problems.eigensolution_maxwell import EigensolutionMaxwell3D
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver

from src.postprocessing import basic_plotting
import matplotlib.pyplot as plt
import os
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

hybridsolver_primal = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="primal")

hybridsolver_dual = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="dual")


time_midpoint = hybridsolver_primal.time_midpoint
time_old = hybridsolver_primal.time_old
time_new = hybridsolver_primal.time_new

exact_electric_midpoint, exact_magnetic_midpoint = eigen_maxwell.get_exact_solution(time_midpoint)
exact_electric_old, exact_magnetic_old = eigen_maxwell.get_exact_solution(time_old)
exact_electric_new, exact_magnetic_new = eigen_maxwell.get_exact_solution(time_new)

# Exact quantities
exact_bdflow = fdrk.dot(fdrk.cross(exact_magnetic_midpoint, exact_electric_midpoint), norm_versor) * fdrk.ds

exact_energyrate = 1/time_step * (fdrk.dot(exact_electric_midpoint, exact_electric_new - exact_electric_old) * fdrk.dx\
                                + fdrk.dot(exact_magnetic_midpoint, exact_magnetic_new - exact_magnetic_old) * fdrk.dx)

# Variables at different time steps
electric_primal_old, magnetic_primal_old, _, _ = hybridsolver_primal.state_old.subfunctions
electric_dual_old, magnetic_dual_old, _, _ = hybridsolver_dual.state_old.subfunctions

electric_primal_midpoint, magnetic_primal_midpoint, _, _ = hybridsolver_primal.state_midpoint.subfunctions
electric_dual_midpoint, magnetic_dual_midpoint, _, _ = hybridsolver_dual.state_midpoint.subfunctions

electric_primal_new, magnetic_primal_new, _, _ = hybridsolver_primal.state_new.subfunctions
electric_dual_new, magnetic_dual_new, _, _ = hybridsolver_dual.state_new.subfunctions

# Power balance combining primal and dual
discrete_bdflow = fdrk.dot(fdrk.cross(magnetic_primal_midpoint, electric_dual_midpoint), norm_versor) * fdrk.ds

discrete_energyrate = 1/time_step * (fdrk.dot(electric_dual_midpoint, electric_primal_new - electric_primal_old) * fdrk.dx\
                            + fdrk.dot(magnetic_primal_midpoint, magnetic_dual_new - magnetic_dual_old) * fdrk.dx)

if rank==0:
    directory_results = os.path.dirname(os.path.abspath(__file__)) + '/results/'
    # Check if the directory exists
    if not os.path.exists(directory_results):
        # If it doesn't exist, create it
        os.makedirs(directory_results)

    time_vec = np.linspace(time_step, time_step * n_time_iter, n_time_iter)

    powerbalance_conservation = np.zeros((n_time_iter, ))
    error_exact_inter_powerbalance = np.zeros((n_time_iter, ))

    exact_bdflow_vec = np.zeros((n_time_iter, ))
    discrete_bdflow_vec = np.zeros((n_time_iter, ))

    exact_energyrate_vec = np.zeros((n_time_iter, ))
    discrete_energyrate_vec = np.zeros((n_time_iter, ))

    div_electric_field = np.zeros((n_time_iter, ))
    div_magnetic_field = np.zeros((n_time_iter, ))

    
for ii in tqdm(range(n_time_iter)):
    actual_time = (ii+1)*time_step

    hybridsolver_primal.integrate(time_step)
    hybridsolver_dual.integrate(time_step)


    if rank==0:
        error_exact_inter_powerbalance[ii] = fdrk.assemble(exact_bdflow-discrete_bdflow)
        powerbalance_conservation[ii] = fdrk.assemble(discrete_energyrate-discrete_bdflow)

        exact_bdflow_vec[ii] = fdrk.assemble(exact_bdflow)
        discrete_bdflow_vec[ii] = fdrk.assemble(discrete_bdflow)

        exact_energyrate_vec[ii] = fdrk.assemble(exact_energyrate)
        discrete_energyrate_vec[ii] = fdrk.assemble(discrete_energyrate)

        div_electric_field[ii] = fdrk.norm(fdrk.div(electric_primal_new))
        div_magnetic_field[ii] = fdrk.norm(fdrk.div(magnetic_dual_new))


    hybridsolver_primal.update_variables()
    hybridsolver_dual.update_variables()


if rank==0:

    basic_plotting.plot_signal(time_vec, error_exact_inter_powerbalance,
                                        title=r"Error numerical and exact boundary flow",
                                        save_path=f"{directory_results}error_bdflow_maxwell")

    basic_plotting.plot_signal(time_vec, powerbalance_conservation,
                                        title=r"Power balance conservation",
                                        save_path=f"{directory_results}power_balance_maxwell")

    basic_plotting.plot_signal(time_vec, div_electric_field, 
                                        title=r"Norm of the divergence $E_h^2$",
                                        save_path=f"{directory_results}div_electric")
    
    basic_plotting.plot_signal(time_vec, div_magnetic_field,  
                                        title=r"Norm of the divergence $H^2_h$",
                                        save_path=f"{directory_results}div_magnetic")

    
    plt.show()