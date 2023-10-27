import firedrake as fdrk
from src.problems.fichera_maxwell import MaxwellFichera
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver
import os
import math
from tqdm import tqdm
import firedrake as fdrk
import numpy as np
from os.path import expanduser
from math import pi
import matplotlib.pyplot as plt
from src.postprocessing import basic_plotting
from src.preprocessing.parser import *

pol_degree = 2
t_end = pi
time_step = t_end/100

n_time_iter = math.ceil(t_end/time_step)

mesh_size = 1/8
problem = MaxwellFichera(mesh_size)

coeff_electric, coeff_magnetic = problem.get_material_coefficients()

hybridsolver_primal = HamiltonianWaveSolver(problem = problem, 
                                            pol_degree=pol_degree,
                                            time_step=time_step, 
                                            system="Maxwell", 
                                            discretization="hybrid", 
                                            formulation="primal")

hybridsolver_dual = HamiltonianWaveSolver(problem = problem, 
                                          pol_degree=pol_degree, 
                                            time_step=time_step, 
                                            system="Maxwell",
                                            discretization="hybrid", 
                                            formulation="dual")

time_midpoint = hybridsolver_primal.time_midpoint
time_old = hybridsolver_primal.time_old
time_new = hybridsolver_primal.time_new

# Variables at different time steps
electric_primal_old, magnetic_primal_old, normal_primal_old, tangential_primal_old = hybridsolver_primal.state_old.subfunctions
electric_dual_old, magnetic_dual_old, normal_dual_old, tangential_dual_old = hybridsolver_dual.state_old.subfunctions

electric_primal_midpoint, _, _, _ = hybridsolver_primal.state_midpoint.subfunctions
electric_dual_midpoint, _, _, _ = hybridsolver_dual.state_midpoint.subfunctions

# electric_primal_new, magnetic_primal_new, _, _ = hybridsolver_primal.state_new.subfunctions
# electric_dual_new, magnetic_dual_new, _, _ = hybridsolver_dual.state_new.subfunctions

if save_out:
    directory_paraview = expanduser("~") + f"/StoreResults/Ph_Hybridization/{str(problem)}/Paraview/"
    if not os.path.exists(directory_paraview):
        # If it doesn't exist, create it
        os.makedirs(directory_paraview)

    directory_matplotlib = expanduser("~") + f"/StoreResults/Ph_Hybridization/{str(problem)}/Matplotlib/"
    if not os.path.exists(directory_matplotlib):
        # If it doesn't exist, create it
        os.makedirs(directory_matplotlib)

    outfile_hybrid_primal = fdrk.File(f"{directory_paraview}/Fields_hybrid_primal.pvd")
    outfile_hybrid_dual = fdrk.File(f"{directory_paraview}/Fields_hybrid_dual.pvd")

    outfile_hybrid_primal.write(electric_primal_old, magnetic_primal_old, \
                                normal_primal_old, tangential_primal_old, time=float(hybridsolver_primal.time_old))
    outfile_hybrid_dual.write(electric_dual_old, magnetic_dual_old, \
                            normal_dual_old, tangential_dual_old, time=float(hybridsolver_dual.time_old))


time_vec = np.linspace(0, time_step * n_time_iter, n_time_iter+1)

energy_primal_vec = np.zeros((n_time_iter+1, ))
energy_dual_vec = np.zeros((n_time_iter+1, ))

energy_primal = fdrk.inner(electric_primal_old, coeff_electric * electric_primal_old) * fdrk.dx \
               +fdrk.inner(magnetic_primal_old, coeff_magnetic * magnetic_primal_old) * fdrk.dx \

energy_dual = fdrk.inner(electric_dual_old, coeff_electric * electric_dual_old) * fdrk.dx \
              +fdrk.inner(magnetic_dual_old, coeff_magnetic * magnetic_dual_old) * fdrk.dx \
              
energy_primal_vec[0] = fdrk.assemble(energy_primal)
energy_dual_vec[0] = fdrk.assemble(energy_dual)

save_figure_time = [t_end/4, t_end/2, 3*t_end/4, t_end]
kk=0

for ii in tqdm(range(1,n_time_iter+1)):
    actual_time = ii*time_step

    hybridsolver_primal.integrate()
    hybridsolver_dual.integrate()

    hybridsolver_primal.update_variables()
    hybridsolver_dual.update_variables()

    energy_primal_vec[ii] = fdrk.assemble(energy_primal)
    energy_dual_vec[ii] = fdrk.assemble(energy_dual)

    if save_out:
        if abs(actual_time - save_figure_time[kk])< 1e-9:
            outfile_hybrid_primal.write(electric_primal_old, magnetic_primal_old, \
                                normal_primal_old, tangential_primal_old, time=float(hybridsolver_primal.time_old))
            outfile_hybrid_dual.write(electric_dual_old, magnetic_dual_old, \
                            normal_dual_old, tangential_dual_old, time=float(hybridsolver_dual.time_old))
   
            kk = kk +1

basic_plotting.plot_signals(time_vec, energy_primal_vec, energy_dual_vec,\
                            legend=["primal", "dual"], title="Energy", save_path=f"{directory_matplotlib}energy.eps")
