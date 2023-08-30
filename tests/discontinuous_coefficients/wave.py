from src.problems.discontinuous_coefficients_wave import DiscontinuousWave
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver

import os
import math
from tqdm import tqdm
import firedrake as fdrk
import numpy as np
from os.path import expanduser

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.postprocessing import basic_plotting

save_output_file = input("Do you want to save in your home directory into the folder StoreResults? ")

n_elements_x = 32
n_elements_y = 32


pol_degree = 2
time_step = 4/2000
t_end = 4

n_time_iter = math.ceil(t_end/time_step)

quad = True

problem = DiscontinuousWave(n_elements_x, n_elements_y, quad=quad)

coeff_velocity, coeff_sigma = problem.get_material_coefficients()

hybridsolver_primal = HamiltonianWaveSolver(problem = problem, 
                                            pol_degree=pol_degree,
                                            time_step=time_step, 
                                            system="Wave", 
                                            discretization="hybrid", 
                                            formulation="primal")

hybridsolver_dual = HamiltonianWaveSolver(problem = problem, 
                                          pol_degree=pol_degree, 
                                            time_step=time_step, 
                                            system="Wave",
                                            discretization="hybrid", 
                                            formulation="dual")


displacement_primal_old = fdrk.Function(hybridsolver_primal.operators.fullspace.sub(0))
displacement_dual_old = fdrk.Function(hybridsolver_dual.operators.fullspace.sub(0))

displacement_primal_old.assign(fdrk.Constant(0))
displacement_dual_old.assign(fdrk.Constant(0))

time_midpoint = hybridsolver_primal.time_midpoint
time_old = hybridsolver_primal.time_old
time_new = hybridsolver_primal.time_new

# Variables at different time steps
velocity_primal_old, sigma_primal_old, normal_primal_old, tangential_primal_old = hybridsolver_primal.state_old.subfunctions
velocity_dual_old, sigma_dual_old, normal_dual_old, tangential_dual_old = hybridsolver_dual.state_old.subfunctions

velocity_primal_midpoint, _, _, _ = hybridsolver_primal.state_midpoint.subfunctions
velocity_dual_midpoint, _, _, _ = hybridsolver_dual.state_midpoint.subfunctions

# velocity_primal_new, sigma_primal_new, _, _ = hybridsolver_primal.state_new.subfunctions
# velocity_dual_new, sigma_dual_new, _, _ = hybridsolver_dual.state_new.subfunctions

if save_output_file:
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

    outfile_hybrid_primal.write(velocity_primal_old, sigma_primal_old, \
                                normal_primal_old, tangential_primal_old, time=float(hybridsolver_primal.time_old))
    outfile_hybrid_dual.write(velocity_dual_old, sigma_dual_old, \
                            normal_dual_old, tangential_dual_old, time=float(hybridsolver_dual.time_old))

    output_freq = 20

    frames_displacement_primal = []
    frames_displacement_dual = []

time_vec = np.linspace(0, time_step * n_time_iter, n_time_iter+1)

energy_primal_vec = np.zeros((n_time_iter+1, ))
energy_dual_vec = np.zeros((n_time_iter+1, ))

energy_primal = fdrk.inner(velocity_primal_old, coeff_velocity * velocity_primal_old) * fdrk.dx \
               +fdrk.inner(sigma_primal_old, coeff_sigma * sigma_primal_old) * fdrk.dx \

energy_dual = fdrk.inner(velocity_dual_old, coeff_velocity * velocity_dual_old) * fdrk.dx \
              +fdrk.inner(sigma_dual_old, coeff_sigma * sigma_dual_old) * fdrk.dx \
              
energy_primal_vec[0] = fdrk.assemble(energy_primal)
energy_dual_vec[0] = fdrk.assemble(energy_dual)

save_figure_time = [0.2, 0.6, 1.8, 4]
kk=0

for ii in tqdm(range(n_time_iter)):
    actual_time = (ii+1)*time_step

    hybridsolver_primal.integrate()
    hybridsolver_dual.integrate()

    displacement_primal_old.assign(displacement_primal_old + time_step*velocity_primal_midpoint)
    displacement_dual_old.assign(displacement_dual_old + time_step*velocity_dual_midpoint)

    hybridsolver_primal.update_variables()
    hybridsolver_dual.update_variables()

    energy_primal_vec[ii+1] = fdrk.assemble(energy_primal)
    energy_dual_vec[ii+1] = fdrk.assemble(energy_dual)

    
    outfile_hybrid_primal.write(velocity_primal_old, sigma_primal_old, \
                                normal_primal_old, tangential_primal_old, time=float(hybridsolver_primal.time_old))
    outfile_hybrid_dual.write(velocity_dual_old, sigma_dual_old, \
                            normal_dual_old, tangential_dual_old, time=float(hybridsolver_dual.time_old))

    if save_output_file:
        if ii % output_freq == 0:
            frames_displacement_primal.append(displacement_primal_old.copy(deepcopy=True))
            frames_displacement_dual.append(displacement_dual_old.copy(deepcopy=True))

        if actual_time == save_figure_time[kk]:
            save_path_primal = f"{directory_matplotlib}displacement_primal_{save_figure_time[kk]}"
            save_path_dual = f"{directory_matplotlib}displacement_dual_{save_figure_time[kk]}"

            fig = plt.figure()
            axes = fig.add_subplot(projection='3d')
            # Adding labels
            axes.set_xlabel('x')
            axes.set_ylabel('y')
            fdrk.trisurf(displacement_primal_old, axes=axes)
            # axes.set_title('velocity primal')
            
            plt.savefig(save_path_primal, dpi='figure', format='eps')

            fdrk.trisurf(displacement_dual_old, axes=axes)
        # axes.set_title('velocity dual')

            plt.savefig(save_path_dual, dpi='figure', format='eps')

            kk+=1

basic_plotting.plot_signals(time_vec, energy_primal_vec, energy_dual_vec,\
                            legend=["primal", "dual"], title="Energy", save_path=f"{directory_matplotlib}energy.eps")
plt.show()

if save_output_file:

    nsp = 64
    fn_plotter = fdrk.FunctionPlotter(problem.domain, num_sample_points=nsp)

    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = fdrk.tripcolor(displacement_primal_old, num_sample_points=nsp, axes=axes)
    fig.colorbar(colors)

    def animate(q):
        colors.set_array(fn_plotter(q))

    interval = 1e3 * output_freq * time_step
    animation_primal = FuncAnimation(fig, animate, frames=frames_displacement_primal, interval=interval)
    animation_dual = FuncAnimation(fig, animate, frames=frames_displacement_dual, interval=interval)
    try:
        animation_primal.save(f"{directory_matplotlib}displacement_primal.mp4", writer="ffmpeg")
        animation_dual.save(f"{directory_matplotlib}displacement_dual.mp4", writer="ffmpeg")
    except:
        print("Failed to write movie! Try installing `ffmpeg`.")