from src.problems.eigensolution_maxwell import EigensolutionMaxwell3D
from src.problems.eigensolution_wave import EigensolutionWave3D
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver
from src.postprocessing import basic_plotting
import matplotlib.pyplot as plt

import os
import math
from tqdm import tqdm
import firedrake as fdrk
import numpy as np
from mpi4py import MPI
from firedrake.petsc import PETSc
from tests.basic.debug_solver import debug_wave


n_elements = 3
pol_degree = 1
time_step = 0.01
t_end = 10*time_step
n_time_iter = math.ceil(t_end/time_step)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

case = "Wave"
if case=="Maxwell":
    problem = EigensolutionMaxwell3D(n_elements, n_elements, n_elements)
else:
    problem = EigensolutionWave3D(n_elements, n_elements, n_elements, bc_type="neumann")

time = fdrk.Constant(0)
exact_first, exact_second = problem.get_exact_solution(time)

mixedsolver_primal = HamiltonianWaveSolver(problem = problem, 
                                           system=case, 
                                           time_step=time_step, 
                                           pol_degree=pol_degree, 
                                            discretization="mixed", 
                                            formulation="primal")

hybridsolver_primal = HamiltonianWaveSolver(problem = problem,
                                            system=case,
                                            time_step=time_step,
                                            pol_degree=pol_degree, 
                                            discretization="hybrid", 
                                            formulation="primal")

mixedsolver_dual = HamiltonianWaveSolver(problem = problem, 
                                         system=case, 
                                         time_step=time_step, 
                                         pol_degree=pol_degree, 
                                         discretization="mixed", 
                                         formulation="dual")

hybridsolver_dual = HamiltonianWaveSolver(problem = problem,
                                        system=case, 
                                        time_step=time_step, 
                                        pol_degree=pol_degree, 
                                        discretization="hybrid", 
                                        formulation="dual")

PETSc.Sys.Print(f"Size Mixed: {mixedsolver_primal.space_operators.dim()}. Size Hybrid {hybridsolver_primal.space_operators.dim()}")

if rank==0:
    directory_results = os.path.dirname(os.path.abspath(__file__)) + '/results/'
    if not os.path.exists(directory_results):
        # If it doesn't exist, create it
        os.makedirs(directory_results)

    time_vec = np.linspace(time_step, time_step * n_time_iter, n_time_iter)

    error_first_primal = np.zeros((n_time_iter, ))
    error_second_primal = np.zeros((n_time_iter, ))

    error_first_dual = np.zeros((n_time_iter, ))
    error_second_dual = np.zeros((n_time_iter, ))

    value_mixed_first_primal = np.zeros((n_time_iter, ))
    value_mixed_second_primal = np.zeros((n_time_iter, ))

    value_hybrid_first_primal = np.zeros((n_time_iter, ))
    value_hybrid_second_primal = np.zeros((n_time_iter, ))

    value_mixed_first_dual = np.zeros((n_time_iter, ))
    value_mixed_second_dual = np.zeros((n_time_iter, ))

    value_hybrid_first_dual = np.zeros((n_time_iter, ))
    value_hybrid_second_dual = np.zeros((n_time_iter, ))
    
    value_exact_first = np.zeros((n_time_iter, ))
    value_exact_second = np.zeros((n_time_iter, ))

    

for ii in tqdm(range(n_time_iter)):
    actual_time = (ii+1)*time_step

    mixedsolver_primal.integrate()
    hybridsolver_primal.integrate()

    mixedsolver_dual.integrate()
    hybridsolver_dual.integrate()

    mixedsolver_primal.update_variables()
    hybridsolver_primal.update_variables()
    
    mixedsolver_dual.update_variables()
    hybridsolver_dual.update_variables()

    mixed_first_primal, mixed_second_primal = mixedsolver_primal.state_old.subfunctions
    hybrid_first_primal, hybrid_second_primal, _, _ = hybridsolver_primal.state_old.subfunctions
    
    mixed_first_dual, mixed_second_dual = mixedsolver_dual.state_old.subfunctions
    hybrid_first_dual, hybrid_second_dual, _, _ = hybridsolver_dual.state_old.subfunctions

    errorvalue_first_primal = fdrk.norm(mixed_first_primal - hybrid_first_primal)
    errorvalue_second_primal = fdrk.norm(mixed_second_primal - hybrid_second_primal)

    errorvalue_first_dual = fdrk.norm(mixed_first_dual - hybrid_first_dual)
    errorvalue_second_dual = fdrk.norm(mixed_second_dual - hybrid_second_dual)


    if rank==0:
        time.assign(actual_time)

        error_first_primal[ii] = errorvalue_first_primal
        error_second_primal[ii] = errorvalue_second_primal

        error_first_dual[ii] = errorvalue_first_dual
        error_second_dual[ii] = errorvalue_second_dual

        point = (1/7, 1/10, 2/3)

        if case=="Maxwell":
            value_mixed_first_primal[ii] = mixed_first_primal.at(point)[0]
            value_hybrid_first_primal[ii] = hybrid_first_primal.at(point)[0]

            value_mixed_first_dual[ii] = mixed_first_dual.at(point)[0]
            value_hybrid_first_dual[ii] = hybrid_first_dual.at(point)[0]
            value_exact_first[ii] = exact_first[0](point)
        else:
            value_mixed_first_primal[ii] = mixed_first_primal.at(point)
            value_hybrid_first_primal[ii] = hybrid_first_primal.at(point)

            value_mixed_first_dual[ii] = mixed_first_dual.at(point)
            value_hybrid_first_dual[ii] = hybrid_first_dual.at(point)
            value_exact_first[ii] = exact_first(point)

        value_mixed_second_primal[ii] = mixed_second_primal.at(point)[0]
        value_hybrid_second_primal[ii] = hybrid_second_primal.at(point)[0]
        value_mixed_second_dual[ii] = mixed_second_dual.at(point)[0]
        value_hybrid_second_dual[ii] = hybrid_second_dual.at(point)[0]
        
        value_exact_second[ii] = exact_second[0](point)


if case=="Maxwell":
    first_primal = "E^2"
    first_dual = "E^1"

    second_primal = "H^1"
    second_dual = "H^2"
else: 
    first_primal = "p^3"
    first_dual = "p^0"

    second_primal = "u^2"
    second_dual = "u^1"


basic_plotting.plot_signal(time_vec, error_first_primal,
                                    title=r"$||{field}_{{\mathrm{{mix}}}} - {field}_{{\mathrm{{hyb}}}}||_{{L^2}}$".format(field=first_primal),
                                    save_path=f"{directory_results}equiv_mixedhybrid_{case}_{first_primal}")

basic_plotting.plot_signal(time_vec, error_second_primal,
                                    title=r"$||{field}_{{\mathrm{{mix}}}} - {field}_{{\mathrm{{hyb}}}}||_{{L^2}}$".format(field=second_primal),
                                    save_path=f"{directory_results}equiv_mixedhybrid_{case}_{second_primal}")

basic_plotting.plot_signal(time_vec, error_first_dual,
                                    title=r"$||{field}_{{\mathrm{{mix}}}} - {field}_{{\mathrm{{hyb}}}}||_{{L^2}}$".format(field=first_dual),
                                    save_path=f"{directory_results}equiv_mixedhybrid_{case}_{first_dual}")

basic_plotting.plot_signal(time_vec, error_second_dual,
                                    title=r"$||{field}_{{\mathrm{{mix}}}} - {field}_{{\mathrm{{hyb}}}}||_{{L^2}}$".format(field=second_dual),
                                    save_path=f"{directory_results}equiv_mixedhybrid__{case}_{second_dual}")

basic_plotting.plot_signals(time_vec, value_mixed_first_primal, value_hybrid_first_primal,\
                                    value_mixed_first_dual, value_hybrid_first_dual, value_exact_first, \
                                    legend=["mixed primal", "hybrid primal", "mixed dual", "hybrid dual", "exact"], title="First field at point")

basic_plotting.plot_signals(time_vec, value_mixed_second_primal, value_hybrid_second_primal,\
                                    value_mixed_second_dual, value_hybrid_second_dual, value_exact_second,\
                                    legend=["mixed primal", "hybrid primal", "mixed dual", "hybrid dual", "exact"],  title="Second field at point")

basic_plotting.plot_signals(time_vec, error_first_primal,
                                    error_second_primal, 
                                    error_first_dual, 
                                    error_second_dual,
                                    legend=[r"$||{field}_{{\mathrm{{mix}}}} - {field}_{{\mathrm{{hyb}}}}||_{{L^2}}$".format(field=first_primal), 
                                            r"$||{field}_{{\mathrm{{mix}}}} - {field}_{{\mathrm{{hyb}}}}||_{{L^2}}$".format(field=second_primal),
                                            r"$||{field}_{{\mathrm{{mix}}}} - {field}_{{\mathrm{{hyb}}}}||_{{L^2}}$".format(field=first_dual),
                                            r"$||{field}_{{\mathrm{{mix}}}} - {field}_{{\mathrm{{hyb}}}}||_{{L^2}}$".format(field=second_dual)],  
                                    title=f"Equivalence mixed/hybrid {case}",
                                    save_path=f"{directory_results}equiv_mixedhybrid_{case}")

plt.show()
