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

n_elements = 3
pol_degree = 2
time_step = 0.01
t_end = 5*time_step
n_time_iter = math.ceil(t_end/time_step)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


eigen_maxwell = EigensolutionMaxwell3D(n_elements, n_elements, n_elements)

directory_results_problem, directory_results_problem_matplotlib, _ = directories_problem(eigen_maxwell)

solver_params={}

mixedsolver_primal = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="mixed", \
                                    type_formulation="primal", \
                                    solver_parameters= solver_params)


hybridsolver_primal = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="primal", \
                                    solver_parameters= solver_params)

mixedsolver_dual = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="mixed", \
                                    type_formulation="dual", \
                                    solver_parameters= solver_params)


hybridsolver_dual = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="dual", \
                                    solver_parameters= solver_params)

if rank==0:
    time_vec = np.linspace(time_step, time_step * time_step, n_time_iter)
    error_electric_primal = np.zeros((n_time_iter))
    error_magnetic_primal = np.zeros((n_time_iter))
    error_electric_dual = np.zeros((n_time_iter))
    error_magnetic_dual = np.zeros((n_time_iter))



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

    mixed_electric_primal, mixed_magnetic_primal = mixedsolver_primal.state_old.subfunctions

    mixed_electric_dual, mixed_magnetic_dual = mixedsolver_dual.state_old.subfunctions

    hybrid_electric_primal, hybrid_magnetic_primal = hybridsolver_primal.state_old.subfunctions[0:2]
    
    hybrid_electric_dual, hybrid_magnetic_dual = hybridsolver_dual.state_old.subfunctions[0:2]

    errorvalue_electric_primal = fdrk.norm(mixed_electric_primal - hybrid_electric_primal)
    errorvalue_magnetic_primal = fdrk.norm(mixed_magnetic_primal - hybrid_magnetic_primal)

    errorvalue_electric_dual = fdrk.norm(mixed_electric_dual - hybrid_electric_dual)
    errorvalue_magnetic_dual = fdrk.norm(mixed_magnetic_dual - hybrid_magnetic_dual)

    if rank==0:
        error_electric_primal[ii] = errorvalue_electric_primal
        error_magnetic_primal[ii] = errorvalue_magnetic_primal
        error_electric_dual[ii] = errorvalue_electric_dual
        error_magnetic_dual[ii] = errorvalue_magnetic_dual


if rank==0:

    basic_plotting.plot_signal(time_vec, error_electric_primal,
                                        title="$E^2$",
                                        save_path=f"{directory_results_problem_matplotlib}equiv_mixedhybrid_maxwell")

    basic_plotting.plot_signal(time_vec, error_magnetic_primal,
                                        title="$H^1$",
                                        save_path=f"{directory_results_problem_matplotlib}equiv_mixedhybrid_maxwell")

    basic_plotting.plot_signal(time_vec, error_electric_dual,
                                        title="$E^1$",
                                        save_path=f"{directory_results_problem_matplotlib}equiv_mixedhybrid_maxwell")

    basic_plotting.plot_signal(time_vec, error_magnetic_dual,
                                        title="$H^2$",
                                        save_path=f"{directory_results_problem_matplotlib}equiv_mixedhybrid_maxwell")

    # basic_plotting.plot_signals(time_vec, error_electric_primal,
    #                                     error_magnetic_primal, 
    #                                     error_electric_dual, 
    #                                     error_magnetic_dual,
    #                                     legend=["1", "2", "3", "4"], 
    #                                     title=r"Equivalence mixed/hybrid Maxwell",
    #                                     save_path=f"{directory_results_problem_matplotlib}equiv_mixedhybrid_maxwell")
    
    plt.show()



# legend=[r"$||E^2_{\mathrm{cont}} - E^2_{\mathrm{hyb}}||_{L^2}$", 
#         r"$||H^1_{\mathrm{cont}} - H^1_{\mathrm{hyb}}||_{L^2}$",
#         r"$||E^1_{\mathrm{cont}} - E^1_{\mathrm{hyb}}||_{L^2}$",
#         r"$||H^2_{\mathrm{cont}} - H^2_{\mathrm{hyb}}||_{L^2}$"], 