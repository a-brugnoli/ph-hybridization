from src.problems.eigensolution_maxwell import EigensolutionMaxwell
from src.problems.eigensolution_wave import EigensolutionWave
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
time_step = 1/500
t_end = 1
n_time_iter = math.ceil(t_end/time_step)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dim=3
quad = False 

case = input("Which model (Wave or Maxwell)? ")
if case=="Maxwell":
    problem = EigensolutionMaxwell(n_elements, n_elements, n_elements, quad=quad)
else:
    problem = EigensolutionWave(n_elements, n_elements, n_elements, quad=quad, dim=dim)

norm_versor = problem.normal_versor

hybridsolver_primal = HamiltonianWaveSolver(problem = problem, 
                                            pol_degree=pol_degree,
                                            time_step=time_step, 
                                            system=case, 
                                            discretization="hybrid", 
                                            formulation="primal")

hybridsolver_dual = HamiltonianWaveSolver(problem = problem, 
                                          pol_degree=pol_degree, 
                                            time_step=time_step, 
                                            system=case,
                                            discretization="hybrid", 
                                            formulation="dual")


time_midpoint = hybridsolver_primal.time_midpoint
time_old = hybridsolver_primal.time_old
time_new = hybridsolver_primal.time_new

exact_first_midpoint, exact_second_midpoint = problem.get_exact_solution(time_midpoint)
exact_first_old, exact_second_old = problem.get_exact_solution(time_old)
exact_first_new, exact_second_new = problem.get_exact_solution(time_new)


# Exact quantities

if case=="Maxwell":
    if problem.domain.extruded:
        exact_bdflow = fdrk.dot(fdrk.cross(exact_second_midpoint, exact_first_midpoint), norm_versor) * fdrk.ds_v \
                      +fdrk.dot(fdrk.cross(exact_second_midpoint, exact_first_midpoint), norm_versor) * fdrk.ds_tb
    else:
        exact_bdflow = fdrk.dot(fdrk.cross(exact_second_midpoint, exact_first_midpoint), norm_versor) * fdrk.ds
else:
    if problem.domain.extruded:
        exact_bdflow = exact_first_midpoint * fdrk.dot(exact_second_midpoint, norm_versor) * fdrk.ds_v \
                      +exact_first_midpoint * fdrk.dot(exact_second_midpoint, norm_versor) * fdrk.ds_tb
    else:
        exact_bdflow = exact_first_midpoint * fdrk.dot(exact_second_midpoint, norm_versor) * fdrk.ds


exact_energyrate = 1/time_step * (fdrk.dot(exact_first_midpoint, exact_first_new - exact_first_old) * fdrk.dx\
                                + fdrk.dot(exact_second_midpoint, exact_second_new - exact_second_old) * fdrk.dx)

# Variables at different time steps
first_primal_old, second_primal_old, _, _ = hybridsolver_primal.state_old.subfunctions
first_dual_old, second_dual_old, _, _ = hybridsolver_dual.state_old.subfunctions

first_primal_midpoint, second_primal_midpoint, _, _ = hybridsolver_primal.state_midpoint.subfunctions
first_dual_midpoint, second_dual_midpoint, _, _ = hybridsolver_dual.state_midpoint.subfunctions

first_primal_new, second_primal_new, _, _ = hybridsolver_primal.state_new.subfunctions
first_dual_new, second_dual_new, _, _ = hybridsolver_dual.state_new.subfunctions

# Power balance combining primal and dual
if case=="Maxwell":
    if problem.domain.extruded:
        discrete_bdflow = fdrk.dot(fdrk.cross(second_primal_midpoint, first_dual_midpoint), norm_versor) * fdrk.ds_v \
                         +fdrk.dot(fdrk.cross(second_primal_midpoint, first_dual_midpoint), norm_versor) * fdrk.ds_tb

    else:
        discrete_bdflow = fdrk.dot(fdrk.cross(second_primal_midpoint, first_dual_midpoint), norm_versor) * fdrk.ds
else:
    if problem.domain.extruded:
        discrete_bdflow = first_dual_midpoint * fdrk.dot(second_primal_midpoint, norm_versor) * fdrk.ds_v \
                         +first_dual_midpoint * fdrk.dot(second_primal_midpoint, norm_versor) * fdrk.ds_tb
    else:
        discrete_bdflow = first_dual_midpoint * fdrk.dot(second_primal_midpoint, norm_versor) * fdrk.ds


if problem.forcing:
    forcing_first_midpoint, forcing_second_midpoint = problem.get_forcing(time_midpoint)

    exact_volumeflow = fdrk.inner(exact_first_midpoint, forcing_first_midpoint) * fdrk.dx \
                   + fdrk.inner(exact_second_midpoint, forcing_second_midpoint) * fdrk.dx
    
    discrete_volumeflow = fdrk.inner(first_dual_midpoint, forcing_first_midpoint) * fdrk.dx \
                   + fdrk.inner(second_primal_midpoint, forcing_second_midpoint) * fdrk.dx
else:
    exact_volume_flow = 0


discrete_energyrate = 1/time_step * (fdrk.dot(first_dual_midpoint, first_primal_new - first_primal_old) * fdrk.dx\
                            + fdrk.dot(second_primal_midpoint, second_dual_new - second_dual_old) * fdrk.dx)

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

    if case=="Maxwell":
        div_first_primal = np.zeros((n_time_iter, ))
        div_second_dual = np.zeros((n_time_iter, ))
    else:
        curl_second_dual = np.zeros((n_time_iter, ))

    
for ii in tqdm(range(n_time_iter)):
    actual_time = (ii+1)*time_step

    hybridsolver_primal.integrate()
    hybridsolver_dual.integrate()


    if rank==0:
        error_exact_inter_powerbalance[ii] = fdrk.assemble(exact_bdflow-discrete_bdflow)
        powerbalance_conservation[ii] = fdrk.assemble(discrete_energyrate-discrete_bdflow)

        exact_bdflow_vec[ii] = fdrk.assemble(exact_bdflow)
        discrete_bdflow_vec[ii] = fdrk.assemble(discrete_bdflow)

        exact_energyrate_vec[ii] = fdrk.assemble(exact_energyrate)
        discrete_energyrate_vec[ii] = fdrk.assemble(discrete_energyrate)

        if case=="Maxwell":
            div_first_primal[ii] = fdrk.norm(fdrk.div(first_primal_new))
            div_second_dual[ii] = fdrk.norm(fdrk.div(second_dual_new))
        else:
            curl_second_dual[ii] = fdrk.norm(fdrk.curl(second_dual_new))


    hybridsolver_primal.update_variables()
    hybridsolver_dual.update_variables()


if rank==0:

    basic_plotting.plot_signal(time_vec, error_exact_inter_powerbalance,
                                        title=r"Error numerical and exact boundary flow",
                                        save_path=f"{directory_results}error_bdflow_{case}")

    basic_plotting.plot_signal(time_vec, powerbalance_conservation,
                                        title=r"Power balance conservation",
                                        save_path=f"{directory_results}power_balance_{case}")

    if case=="Maxwell":
        basic_plotting.plot_signal(time_vec, div_first_primal, 
                                            title=r"$H^{\mathrm{div}}(\mathcal{T}_h)$ seminorm of $E_h^2$",
                                            save_path=f"{directory_results}div_electric_{case}")
        
        basic_plotting.plot_signal(time_vec, div_second_dual,  
                                        title=r"$H^{\mathrm{div}}(\mathcal{T}_h)$ seminorm of $H^2_h$",
                                        save_path=f"{directory_results}div_magnetic_{case}")
    else:
        basic_plotting.plot_signal(time_vec, curl_second_dual,  
                                        title=r"$H^{\mathrm{curl}}(\mathcal{T}_h)$ seminorm of $u^1_h$",
                                        save_path=f"{directory_results}curl_velocity_{case}")

    
    plt.show()