from src.problems.eigensolution_maxwell import EigensolutionMaxwell
from src.problems.eigensolution_wave import EigensolutionWave
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver
import pandas as pd

pol_degree_vec = [1,2,3] 
cases = ["Wave", "Maxwell"]
time_step = 1/500
quad = True

results_mixed_wave = {"Primal": {1: [], 2: [], 3:[]}, "Dual": {1: [], 2: [], 3:[]}}
results_hybrid_wave = {"Primal": {1: [], 2: [], 3:[]}, "Dual": {1: [], 2: [], 3:[]}}
results_reduction_wave =  {"Primal": {1: [], 2: [], 3:[]}, "Dual": {1: [], 2: [], 3:[]}}

results_mixed_maxwell = {"Primal": {1: [], 2: [], 3:[]}, "Dual": {1: [], 2: [], 3:[]}}
results_hybrid_maxwell = {"Primal": {1: [], 2: [], 3:[]}, "Dual": {1: [], 2: [], 3:[]}}
results_reduction_maxwell =  {"Primal": {1: [], 2: [], 3:[]}, "Dual": {1: [], 2: [], 3:[]}}

for case in cases:
    
    for pol_degree in pol_degree_vec:

        if pol_degree==1:
            n_elem_vector = [1, 2, 4, 8, 16] 
        elif pol_degree==2:
            n_elem_vector = [1, 2, 4, 8]
        elif pol_degree==3:
            n_elem_vector = [1, 2, 4]

        for n_elem in n_elem_vector:
        
            if case=="Maxwell":
                problem = EigensolutionMaxwell(n_elem, n_elem, n_elem, quad=quad)
            else:
                problem = EigensolutionWave(n_elem, n_elem, n_elem, quad=quad, dim=3)


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


            dim_fullspace_mixed_primal = mixedsolver_primal.operators.fullspace.dim()
            dim_globalspace_hybrid_primal = hybridsolver_primal.operators.space_global.dim()

            dim_fullspace_mixed_dual = mixedsolver_dual.operators.fullspace.dim()
            dim_globalspace_hybrid_dual = hybridsolver_dual.operators.space_global.dim()

            # print(f"{case} equation, deg_ree {pol_degree}, number of elements {n_elem}")
            # print(f"Dimension mixed primal: {dim_fullspace_mixed_primal}")
            # print(f"Dimension hybrid primal: {dim_globalspace_hybrid_primal}")
            # print(f"Reduction primal: {dim_globalspace_hybrid_primal/dim_fullspace_mixed_primal*100}")

            # print(f"Dimension mixed dual: {dim_fullspace_mixed_dual}")
            # print(f"Dimension hybrid dual: {dim_globalspace_hybrid_dual}")
            # print(f"Reduction dual: {dim_globalspacehybrid_dual/dim_fullspace_mixed_dual*100}")

            if case=="Wave":
                results_mixed_wave["Primal"][pol_degree].append(dim_fullspace_mixed_primal)
                results_hybrid_wave["Primal"][pol_degree].append(dim_globalspace_hybrid_primal)
                results_reduction_wave["Primal"][pol_degree].append(dim_globalspace_hybrid_primal/dim_fullspace_mixed_primal*100)

                results_mixed_wave["Dual"][pol_degree].append(dim_fullspace_mixed_dual)
                results_hybrid_wave["Dual"][pol_degree].append(dim_globalspace_hybrid_dual)
                results_reduction_wave["Dual"][pol_degree].append(dim_globalspace_hybrid_dual/dim_fullspace_mixed_dual*100)
            else:
                results_mixed_maxwell["Primal"][pol_degree].append(dim_fullspace_mixed_primal)
                results_hybrid_maxwell["Primal"][pol_degree].append(dim_globalspace_hybrid_primal)
                results_reduction_maxwell["Primal"][pol_degree].append(dim_globalspace_hybrid_primal/dim_fullspace_mixed_primal*100)

                results_mixed_maxwell["Dual"][pol_degree].append(dim_fullspace_mixed_dual)
                results_hybrid_maxwell["Dual"][pol_degree].append(dim_globalspace_hybrid_dual)
                results_reduction_maxwell["Dual"][pol_degree].append(dim_globalspace_hybrid_dual/dim_fullspace_mixed_dual*100)


print(f"Wave equation")
print(f"Dimension mixed: {results_mixed_wave}")
print(f"Dimension hybrid: {results_hybrid_wave}")
print(f"Reduction: {results_reduction_wave}")

print(f"Maxwell equation")
print(f"Dimension mixed: {results_mixed_maxwell}")
print(f"Dimension hybrid: {results_hybrid_maxwell}")
print(f"Reduction dual: {results_reduction_maxwell}")
