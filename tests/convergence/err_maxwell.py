from src.problems.eigensolution_maxwell import EigensolutionMaxwell3D
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver
import math
from tqdm import tqdm
import firedrake as fdrk
from firedrake.petsc import PETSc


def compute_error(n_elements, pol_degree, time_step=0.01, t_end=1):
    n_time_iter = math.ceil(t_end/time_step)

    problem = EigensolutionMaxwell3D(n_elements, n_elements, n_elements)
    cell_diameter = fdrk.CellDiameter(problem.domain)
    normal_versor = problem.normal_versor
    
    hybridsolver_primal = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, \
                                        type_discretization="hybrid", \
                                        type_formulation="primal")

    hybridsolver_dual = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, \
                                    type_discretization="hybrid", \
                                    type_formulation="dual")

    for ii in tqdm(range(n_time_iter)):
        actual_time = (ii+1)*time_step

        hybridsolver_primal.integrate(time_step)
        hybridsolver_dual.integrate(time_step)
        
        hybridsolver_primal.update_variables()
        hybridsolver_dual.update_variables()
    
    PETSc.Sys.Print(f"Solution with {n_elements} elements and degree {pol_degree} computed")

    exact_electric, exact_magnetic = problem.get_exact_solution(fdrk.Constant(actual_time))

    # Error primal
    electric_primal, magnetic_primal, electric_normal_primal, magnetic_tangential_primal = hybridsolver_primal.state_old.subfunctions

    errorL2_electric_primal = fdrk.norm(exact_electric - electric_primal)
    errorL2_magnetic_primal = fdrk.norm(exact_magnetic - magnetic_primal)
    errorHdiv_electric_primal = fdrk.errornorm(exact_electric, electric_primal, norm_type="Hdiv")
    errorHcurl_magnetic_primal = fdrk.errornorm(exact_magnetic, magnetic_primal, norm_type="Hcurl")
    error_tangential_primal = hybridsolver_primal.operators.trace_norm(exact_magnetic-magnetic_tangential_primal, cell_diameter, normal_versor)
    projected_exact_electric = hybridsolver_primal.operators.project_NED_facetbroken(exact_electric)
    error_normal_primal = hybridsolver_primal.operators.trace_norm(projected_exact_electric-electric_normal_primal, cell_diameter, normal_versor)

    # Error dual
    electric_dual, magnetic_dual, magnetic_normal_dual, electric_tangential_dual = hybridsolver_dual.state_old.subfunctions

    errorL2_electric_dual = fdrk.norm(exact_electric - electric_dual)
    errorL2_magnetic_dual = fdrk.norm(exact_magnetic - magnetic_dual)
    errorHcurl_electric_dual = fdrk.errornorm(exact_electric, electric_dual, norm_type="Hcurl")
    errorHdiv_magnetic_dual = fdrk.errornorm(exact_magnetic, magnetic_primal, norm_type="Hdiv")
    error_tangential_dual = hybridsolver_primal.operators.trace_norm(exact_magnetic-electric_tangential_dual, cell_diameter, normal_versor)
    projected_exact_magnetic = hybridsolver_primal.operators.project_NED_facetbroken(exact_magnetic)
    error_normal_dual = hybridsolver_primal.operators.trace_norm(projected_exact_magnetic-magnetic_normal_dual, cell_diameter, normal_versor)


    error_dict = {
        "errorL2_electric_primal": errorL2_electric_primal, 
        "errorL2_magnetic_primal": errorL2_magnetic_primal, 
        "errorHdiv_electric_primal": errorHdiv_electric_primal,
        "errorHcurl_magnetic_primal": errorHcurl_magnetic_primal,
        "error_tangential_primal": error_tangential_primal,
        "error_normal_primal": error_normal_primal,
        "errorL2_electric_dual": errorL2_electric_dual, 
        "errorL2_magnetic_dual": errorL2_magnetic_dual, 
        "errorHcurl_electric_dual": errorHcurl_electric_dual,
        "errorHdiv_magnetic_dual": errorHdiv_magnetic_dual,
        "error_tangential_dual": error_tangential_dual,
        "error_normal_dual": error_normal_dual,
        }

    return error_dict