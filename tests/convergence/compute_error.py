from src.problems.eigensolution_maxwell import EigensolutionMaxwell3D
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver
import math
from tqdm import tqdm
import firedrake as fdrk
from firedrake.petsc import PETSc


def compute_error(n_elements, pol_degree, bc_type, time_step=0.01, t_end=1, type_system="Maxwell"):
    """
    Returns the Linfinity norm in time of the error
    """
    n_time_iter = math.ceil(t_end/time_step)

    if type_system=="Maxwell":
        problem = EigensolutionMaxwell3D(n_elements, n_elements, n_elements, bc_type=bc_type)
        
        hybridsolver_primal = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree,
                                                    type_system="Maxwell", 
                                                    type_discretization="hybrid", 
                                                    type_formulation="primal")

        hybridsolver_dual = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, 
                                                    type_system="Maxwell", 
                                                    type_discretization="hybrid", 
                                                    type_formulation="dual")
        
    else:
        raise NotImplementedError("Not implemented for wave")

    state_exact = problem.get_exact_solution(fdrk.Constant(0))
    error_dict = dict_error_maxwell(state_exact, hybridsolver_primal, hybridsolver_dual)

    for ii in tqdm(range(n_time_iter)):
        actual_time = (ii+1)*time_step

        hybridsolver_primal.integrate(time_step)
        hybridsolver_dual.integrate(time_step)
        
        hybridsolver_primal.update_variables()
        hybridsolver_dual.update_variables()

        state_exact_actual = problem.get_exact_solution(fdrk.Constant(actual_time))
        error_dict_actual = dict_error_maxwell(state_exact_actual, hybridsolver_primal, hybridsolver_dual)

        # Computation of the Linfinity norm in time
        for key_error, value_error in error_dict_actual.items():
            if error_dict[key_error]< value_error:
                error_dict[key_error]= value_error

    PETSc.Sys.Print(f"Solution with {n_elements} elements, pol degree {pol_degree} and bcs {bc_type} computed")

    # state_exact = problem.get_exact_solution(fdrk.Constant(actual_time))
    # error_dict = dict_error_maxwell(state_exact, hybridsolver_primal, hybridsolver_dual)

    return error_dict


def dict_error_maxwell(state_exact, solver_primal: HamiltonianWaveSolver, solver_dual: HamiltonianWaveSolver):
    exact_electric, exact_magnetic = state_exact
    
    # Error primal
    electric_primal, magnetic_primal, electric_normal_primal, magnetic_tangential_primal = solver_primal.state_old.subfunctions

    error_L2_electric_primal = fdrk.norm(exact_electric-electric_primal)
    error_L2_magnetic_primal = fdrk.norm(exact_magnetic-magnetic_primal)

    error_Hdiv_electric_primal = fdrk.norm(exact_electric-electric_primal, norm_type="Hdiv")
    error_Hcurl_magnetic_primal = fdrk.norm(exact_magnetic-magnetic_primal, norm_type="Hcurl")

    error_tangential_primal = solver_primal.operators.trace_norm(exact_magnetic-magnetic_tangential_primal)
    projected_exact_electric = solver_primal.operators.project_NED_facetbroken(exact_electric)
    error_normal_primal = solver_primal.operators.trace_norm(projected_exact_electric-electric_normal_primal)

    # Error dual
    electric_dual, magnetic_dual, magnetic_normal_dual, electric_tangential_dual = solver_dual.state_old.subfunctions

    error_L2_electric_dual = fdrk.norm(exact_electric - electric_dual)
    error_L2_magnetic_dual = fdrk.norm(exact_magnetic - magnetic_dual)

    error_Hcurl_electric_dual = fdrk.norm(exact_electric-electric_dual, norm_type="Hcurl")
    error_Hdiv_magnetic_dual = fdrk.norm(exact_magnetic-magnetic_dual, norm_type="Hdiv")

    error_tangential_dual = solver_dual.operators.trace_norm(exact_electric-electric_tangential_dual)
    projected_exact_magnetic = solver_dual.operators.project_NED_facetbroken(exact_magnetic)
    error_normal_dual = solver_dual.operators.trace_norm(projected_exact_magnetic-magnetic_normal_dual)

    # Error dual field
    error_L2_electric_df = fdrk.norm(electric_primal - electric_dual)
    error_L2_magnetic_df = fdrk.norm(magnetic_primal - magnetic_dual)


    error_dictionary = {
        "error_L2_electric_primal": error_L2_electric_primal, 
        "error_L2_magnetic_primal": error_L2_magnetic_primal, 
        "error_Hdiv_electric_primal": error_Hdiv_electric_primal,
        "error_Hcurl_magnetic_primal": error_Hcurl_magnetic_primal,
        "error_tangential_primal": error_tangential_primal,
        "error_normal_primal": error_normal_primal,

        "error_L2_electric_dual": error_L2_electric_dual, 
        "error_L2_magnetic_dual": error_L2_magnetic_dual, 
        "error_Hcurl_electric_dual": error_Hcurl_electric_dual,
        "error_Hdiv_magnetic_dual": error_Hdiv_magnetic_dual,
        "error_tangential_dual": error_tangential_dual,
        "error_normal_dual": error_normal_dual,
        
        "error_L2_electric_df": error_L2_electric_df,
        "error_L2_magnetic_df": error_L2_magnetic_df,
        }

    return error_dictionary