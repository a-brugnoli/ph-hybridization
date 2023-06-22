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
from firedrake.petsc import PETSc

n_elements = 4
pol_degree = 3
time_step = 0.01
t_end = 100*time_step
n_time_iter = math.ceil(t_end/time_step)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

eigen_maxwell = EigensolutionMaxwell3D(n_elements, n_elements, n_elements)

time = fdrk.Constant(0)
exact_electric = eigen_maxwell.get_exact_solution(time)[0]
exact_magnetic = eigen_maxwell.get_exact_solution(time)[1]


mixedsolver_primal = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="mixed", \
                                    type_formulation="primal")


hybridsolver_primal = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="primal")

mixedsolver_dual = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="mixed", \
                                    type_formulation="dual")


hybridsolver_dual = HamiltonianWaveSolver(problem = eigen_maxwell, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="dual")


electric_mixed_primal_0, magnetic_mixed_primal_0    = mixedsolver_primal.state_old.subfunctions
electric_hybrid_primal_0, magnetic_hybrid_primal_0, normtrace_magnetic, tangtrace_electric  = hybridsolver_primal.state_old.subfunctions

err_electric_primal_0 = fdrk.norm(electric_mixed_primal_0 - electric_hybrid_primal_0)
err_magnetic_primal_0 = fdrk.norm(magnetic_mixed_primal_0 - magnetic_hybrid_primal_0)

PETSc.Sys.Print(f"Error initial electric primal hybrid/mixed {err_electric_primal_0}")
PETSc.Sys.Print(f"Error initial magnetic primal hybrid/mixed {err_magnetic_primal_0}")

electric_mixed_dual_0, magnetic_mixed_dual_0    = mixedsolver_dual.state_old.subfunctions
electric_hybrid_dual_0, magnetic_hybrid_dual_0, normtrace_magnetic, tangtrace_electric  = hybridsolver_dual.state_old.subfunctions

err_electric_dual_0 = fdrk.norm(electric_mixed_dual_0 - electric_hybrid_dual_0)
err_magnetic_dual_0 = fdrk.norm(magnetic_mixed_dual_0 - magnetic_hybrid_dual_0)

PETSc.Sys.Print(f"Error initial electric dual hybrid/mixed {err_electric_dual_0}")
PETSc.Sys.Print(f"Error initial magnetic dual hybrid/mixed {err_magnetic_dual_0}")


if rank==0:
    directory_results_problem, directory_results_problem_matplotlib, _ = directories_problem(eigen_maxwell)

    time_vec = np.linspace(time_step, time_step * n_time_iter, n_time_iter)

    error_electric_primal = np.zeros((n_time_iter, ))
    error_magnetic_primal = np.zeros((n_time_iter, ))

    error_electric_dual = np.zeros((n_time_iter, ))
    error_magnetic_dual = np.zeros((n_time_iter, ))

    value_mixed_electric_primal = np.zeros((n_time_iter, ))
    value_mixed_magnetic_primal = np.zeros((n_time_iter, ))

    value_hybrid_electric_primal = np.zeros((n_time_iter, ))
    value_hybrid_magnetic_primal = np.zeros((n_time_iter, ))

    value_mixed_electric_dual = np.zeros((n_time_iter, ))
    value_mixed_magnetic_dual = np.zeros((n_time_iter, ))

    value_hybrid_electric_dual = np.zeros((n_time_iter, ))
    value_hybrid_magnetic_dual = np.zeros((n_time_iter, ))
    
    value_exact_electric = np.zeros((n_time_iter, ))
    value_exact_magnetic = np.zeros((n_time_iter, ))



for ii in tqdm(range(n_time_iter)):
    actual_time = (ii+1)*time_step

    mixedsolver_primal.integrate(time_step)
    hybridsolver_primal.integrate(time_step)

    mixedsolver_primal.update_variables()
    hybridsolver_primal.update_variables()

    mixed_electric_primal, mixed_magnetic_primal = mixedsolver_primal.state_old.subfunctions
    hybrid_electric_primal, hybrid_magnetic_primal = hybridsolver_primal.state_old.subfunctions[0:2]

    errorvalue_electric_primal = fdrk.norm(mixed_electric_primal - hybrid_electric_primal)
    errorvalue_magnetic_primal = fdrk.norm(mixed_magnetic_primal - hybrid_magnetic_primal)


    mixedsolver_dual.integrate(time_step)
    hybridsolver_dual.integrate(time_step)
    
    mixedsolver_dual.update_variables()
    hybridsolver_dual.update_variables()

    
    mixed_electric_dual, mixed_magnetic_dual = mixedsolver_dual.state_old.subfunctions
    hybrid_electric_dual, hybrid_magnetic_dual = hybridsolver_dual.state_old.subfunctions[0:2]

    
    errorvalue_electric_dual = fdrk.norm(mixed_electric_dual - hybrid_electric_dual)
    errorvalue_magnetic_dual = fdrk.norm(mixed_magnetic_dual - hybrid_magnetic_dual)


    if rank==0:
        time.assign(actual_time)

        error_electric_primal[ii] = errorvalue_electric_primal
        error_magnetic_primal[ii] = errorvalue_magnetic_primal

        error_electric_dual[ii] = errorvalue_electric_dual
        error_magnetic_dual[ii] = errorvalue_magnetic_dual

        point = (1/2, 1/2, 1/2)
        value_mixed_electric_primal[ii] = mixed_electric_primal.at(point)[0]
        value_hybrid_electric_primal[ii] = hybrid_electric_primal.at(point)[0]

        value_mixed_electric_dual[ii] = mixed_electric_dual.at(point)[0]
        value_hybrid_electric_dual[ii] = hybrid_electric_dual.at(point)[0]

        value_exact_electric[ii] = exact_electric[0](point)


        value_mixed_magnetic_primal[ii] = mixed_magnetic_primal.at(point)[0]
        value_hybrid_magnetic_primal[ii] = hybrid_magnetic_primal.at(point)[0]
        value_mixed_magnetic_dual[ii] = mixed_magnetic_dual.at(point)[0]
        value_hybrid_magnetic_dual[ii] = hybrid_magnetic_dual.at(point)[0]
        
        value_exact_magnetic[ii] = exact_magnetic[0](point)


if rank==0:

    basic_plotting.plot_signal(time_vec, error_electric_primal,
                                        title=r"$||E^2_{\mathrm{mix}} - E^2_{\mathrm{hyb}}||_{L^2}$",
                                        save_path=f"{directory_results_problem_matplotlib}equiv_mixedhybrid_maxwell_E2")

    basic_plotting.plot_signal(time_vec, error_magnetic_primal,
                                        title=r"$||H^1_{\mathrm{mix}} - H^1_{\mathrm{hyb}}||_{L^2}$",
                                        save_path=f"{directory_results_problem_matplotlib}equiv_mixedhybrid_maxwell_H1")

    basic_plotting.plot_signal(time_vec, error_electric_dual,
                                        title=r"$||E^1_{\mathrm{mix}} - E^1_{\mathrm{hyb}}||_{L^2}$",
                                        save_path=f"{directory_results_problem_matplotlib}equiv_mixedhybrid_maxwell_E1")

    basic_plotting.plot_signal(time_vec, error_magnetic_dual,
                                        title=r"$||H^2_{\mathrm{mix}} - H^2_{\mathrm{hyb}}||_{L^2}$",
                                        save_path=f"{directory_results_problem_matplotlib}equiv_mixedhybrid_maxwell_H2")

    basic_plotting.plot_signals(time_vec, value_mixed_electric_primal, value_hybrid_electric_primal,\
                                        value_mixed_electric_dual, value_hybrid_electric_dual, value_exact_electric, \
                                        legend=["mixed primal", "hybrid primal", "mixed dual", "hybrid dual", "exact"], title="E at point")

    basic_plotting.plot_signals(time_vec, value_mixed_magnetic_primal, value_hybrid_magnetic_primal,\
                                        value_mixed_magnetic_dual, value_hybrid_magnetic_dual, value_exact_magnetic,\
                                        legend=["mixed primal", "hybrid primal", "mixed dual", "hybrid dual", "exact"],  title="H at point")

    basic_plotting.plot_signals(time_vec, error_electric_primal,
                                        error_magnetic_primal, 
                                        error_electric_dual, 
                                        error_magnetic_dual,
                                        legend=[r"$||E^2_{\mathrm{mix}} - E^2_{\mathrm{hyb}}||_{L^2}$", 
                                                r"$||H^1_{\mathrm{mix}} - H^1_{\mathrm{hyb}}||_{L^2}$",
                                                r"$||E^1_{\mathrm{mix}} - E^1_{\mathrm{hyb}}||_{L^2}$",
                                                r"$||H^2_{\mathrm{mix}} - H^2_{\mathrm{hyb}}||_{L^2}$"],  
                                        title=r"Equivalence mixed/hybrid Maxwell",
                                        save_path=f"{directory_results_problem_matplotlib}equiv_mixedhybrid_maxwell")

    
    plt.show()


