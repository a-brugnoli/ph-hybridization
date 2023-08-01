from src.problems.eigensolution_maxwell import EigensolutionMaxwell3D
from src.problems.eigensolution_wave import EigensolutionWave3D
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver
import math
from tqdm import tqdm
import firedrake as fdrk
from firedrake.petsc import PETSc


def compute_error(n_elements, pol_degree, bc_type, type_system, time_step=0.01, t_end=1, type_discretization="hybrid"):
    """
    Returns the Linfinity norm in time of the error
    """
    n_time_iter = math.ceil(t_end/time_step)

    if type_system=="Maxwell":
        problem = EigensolutionMaxwell3D(n_elements, n_elements, n_elements, bc_type=bc_type)
    elif type_system=="Wave":
        problem = EigensolutionWave3D(n_elements, n_elements, n_elements, bc_type=bc_type)
    else: 
        raise TypeError("Physics not valid")
        
    hybridsolver_primal = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, time_step=time_step,
                                                type_system=type_system, 
                                                type_discretization=type_discretization, 
                                                type_formulation="primal")

    hybridsolver_dual = HamiltonianWaveSolver(problem = problem, pol_degree=pol_degree, time_step=time_step,
                                                type_system=type_system, 
                                                type_discretization=type_discretization, 
                                                type_formulation="dual")
    
    
    state_exact_0 = problem.get_exact_solution(fdrk.Constant(0))
    if type_system=="Maxwell":
            error_dict = dict_error_maxwell(state_exact_0, hybridsolver_primal, hybridsolver_dual)
    else:
            error_dict = dict_error_wave(state_exact_0, hybridsolver_primal, hybridsolver_dual)
        
    for ii in tqdm(range(n_time_iter)):
        actual_time = (ii+1)*time_step

        hybridsolver_primal.integrate()
        hybridsolver_dual.integrate()
        
        hybridsolver_primal.update_variables()
        hybridsolver_dual.update_variables()

        state_exact_actual = problem.get_exact_solution(fdrk.Constant(actual_time))

        if type_system=="Maxwell":
            error_dict_actual = dict_error_maxwell(state_exact_actual, hybridsolver_primal, hybridsolver_dual)
        else:
            error_dict_actual = dict_error_wave(state_exact_actual, hybridsolver_primal, hybridsolver_dual)

        # Computation of the Linfinity norm in time
        for key_error, value_error in error_dict_actual.items():
            if error_dict[key_error]< value_error:
                error_dict[key_error]= value_error

    PETSc.Sys.Print(f"Solution with {n_elements} elements, pol degree {pol_degree} and bcs {bc_type} computed")

    # assert abs(actual_time-float(hybridsolver_primal.time_old)) < 1e-9
    # assert abs(actual_time-float(hybridsolver_dual.time_old)) < 1e-9

    # state_exact_end = problem.get_exact_solution(fdrk.Constant(actual_time))

    # if type_system=="Maxwell":
    #     error_dict = dict_error_maxwell(state_exact_end, hybridsolver_primal, hybridsolver_dual)
    # else:
    #     error_dict = dict_error_wave(state_exact_end, hybridsolver_primal, hybridsolver_dual)
     
    return error_dict


def dict_error_maxwell(state_exact, solver_primal: HamiltonianWaveSolver, solver_dual: HamiltonianWaveSolver):
    exact_electric, exact_magnetic = state_exact
    
    # Error primal
    if solver_primal.operators.type_discretization=="hybrid":
        electric_primal, magnetic_primal, electric_normal_primal, magnetic_tangential_primal = solver_primal.state_old.subfunctions
        error_tangential_primal = solver_primal.operators.trace_norm_NED(exact_magnetic-magnetic_tangential_primal)
        projected_exact_electric = solver_primal.operators.project_NED_facetbroken(exact_electric)
        error_normal_primal = solver_primal.operators.trace_norm_NED(projected_exact_electric-electric_normal_primal)

    else:
        electric_primal, magnetic_primal = solver_primal.state_old.subfunctions


    error_L2_electric_primal = fdrk.norm(exact_electric-electric_primal)
    error_L2_magnetic_primal = fdrk.norm(exact_magnetic-magnetic_primal)

    error_Hdiv_electric_primal = fdrk.norm(exact_electric-electric_primal, norm_type="Hdiv")
    error_Hcurl_magnetic_primal = fdrk.norm(exact_magnetic-magnetic_primal, norm_type="Hcurl")
        
    # Error dual
    if solver_dual.operators.type_discretization=="hybrid":
        electric_dual, magnetic_dual, magnetic_normal_dual, electric_tangential_dual = solver_dual.state_old.subfunctions
        error_tangential_dual = solver_dual.operators.trace_norm_NED(exact_electric-electric_tangential_dual)
        projected_exact_magnetic = solver_dual.operators.project_NED_facetbroken(exact_magnetic)
        error_normal_dual = solver_dual.operators.trace_norm_NED(projected_exact_magnetic-magnetic_normal_dual)
    else:
        electric_dual, magnetic_dual = solver_dual.state_old.subfunctions

    error_L2_electric_dual = fdrk.norm(exact_electric - electric_dual)
    error_L2_magnetic_dual = fdrk.norm(exact_magnetic - magnetic_dual)

    error_Hcurl_electric_dual = fdrk.norm(exact_electric-electric_dual, norm_type="Hcurl")
    error_Hdiv_magnetic_dual = fdrk.norm(exact_magnetic-magnetic_dual, norm_type="Hdiv")
        
    # Error dual field
    error_L2_electric_df = fdrk.norm(electric_primal - electric_dual)
    error_L2_magnetic_df = fdrk.norm(magnetic_primal - magnetic_dual)


    if solver_primal.operators.type_discretization=="hybrid" and solver_dual.operators.type_discretization=="hybrid":

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
        
    else:
        error_dictionary = {
            "error_L2_electric_primal": error_L2_electric_primal, 
            "error_L2_magnetic_primal": error_L2_magnetic_primal, 
            "error_Hdiv_electric_primal": error_Hdiv_electric_primal,
            "error_Hcurl_magnetic_primal": error_Hcurl_magnetic_primal,
            
            "error_L2_electric_dual": error_L2_electric_dual, 
            "error_L2_magnetic_dual": error_L2_magnetic_dual, 
            "error_Hcurl_electric_dual": error_Hcurl_electric_dual,
            "error_Hdiv_magnetic_dual": error_Hdiv_magnetic_dual,
            
            "error_L2_electric_df": error_L2_electric_df,
            "error_L2_magnetic_df": error_L2_magnetic_df,
            }


    return error_dictionary



def dict_error_wave(state_exact, solver_primal: HamiltonianWaveSolver, solver_dual: HamiltonianWaveSolver):
    exact_pressure, exact_velocity = state_exact
    
    # Error primal
    if solver_primal.operators.type_discretization=="hybrid":
        pressure_primal, velocity_primal, pressure_normal_primal, velocity_tangential_primal = solver_primal.state_old.subfunctions
        error_tangential_primal = solver_primal.operators.trace_norm_RT(exact_velocity-velocity_tangential_primal)
        projected_exact_pressure = solver_primal.operators.project_RT_facetbroken(exact_pressure)
        error_normal_primal = solver_primal.operators.trace_norm_RT(projected_exact_pressure-pressure_normal_primal)

    else:
        pressure_primal, velocity_primal = solver_primal.state_old.subfunctions


    error_L2_pressure_primal = fdrk.norm(exact_pressure-pressure_primal)
    error_L2_velocity_primal = fdrk.norm(exact_velocity-velocity_primal)

    error_Hdiv_velocity_primal = fdrk.norm(exact_velocity-velocity_primal, norm_type="Hdiv")
        
    # Error dual
    if solver_dual.operators.type_discretization=="hybrid":
        pressure_dual, velocity_dual, velocity_normal_dual, pressure_tangential_dual = solver_dual.state_old.subfunctions
        error_tangential_dual = solver_dual.operators.trace_norm_CG(exact_pressure-pressure_tangential_dual)
        projected_exact_velocity = solver_dual.operators.project_CG_facetbroken(exact_velocity)
        error_normal_dual = solver_dual.operators.trace_norm_CG(projected_exact_velocity-velocity_normal_dual)
    else:
        pressure_dual, velocity_dual = solver_dual.state_old.subfunctions

    error_L2_pressure_dual = fdrk.norm(exact_pressure - pressure_dual)
    error_L2_velocity_dual = fdrk.norm(exact_velocity - velocity_dual)

    error_H1_pressure_dual = fdrk.norm(exact_pressure-pressure_dual, norm_type="H1")
    error_Hcurl_velocity_dual = fdrk.norm(exact_velocity-velocity_dual, norm_type="Hcurl")
        
    # Error dual field
    error_L2_pressure_df = fdrk.norm(pressure_primal - pressure_dual)
    error_L2_velocity_df = fdrk.norm(velocity_primal - velocity_dual)


    if solver_primal.operators.type_discretization=="hybrid" and solver_dual.operators.type_discretization=="hybrid":

        error_dictionary = {
            "error_L2_pressure_primal": error_L2_pressure_primal, 
            "error_L2_velocity_primal": error_L2_velocity_primal, 
            "error_Hdiv_velocity_primal": error_Hdiv_velocity_primal,
            "error_tangential_primal": error_tangential_primal,
            "error_normal_primal": error_normal_primal,

            "error_L2_pressure_dual": error_L2_pressure_dual, 
            "error_L2_velocity_dual": error_L2_velocity_dual, 
            "error_H1_pressure_dual": error_H1_pressure_dual,
            "error_Hcurl_velocity_dual": error_Hcurl_velocity_dual,
            "error_tangential_dual": error_tangential_dual,
            "error_normal_dual": error_normal_dual,
            
            "error_L2_pressure_df": error_L2_pressure_df,
            "error_L2_velocity_df": error_L2_velocity_df,
            }
        
    else:
        error_dictionary = {
            "error_L2_pressure_primal": error_L2_pressure_primal, 
            "error_L2_velocity_primal": error_L2_velocity_primal, 
            "error_Hdiv_velocity_primal": error_Hdiv_velocity_primal,
            
            "error_L2_pressure_dual": error_L2_pressure_dual, 
            "error_L2_velocity_dual": error_L2_velocity_dual, 
            "error_H1_pressure_dual": error_H1_pressure_dual,
            "error_Hcurl_velocity_dual": error_Hcurl_velocity_dual,
            
            "error_L2_pressure_df": error_L2_pressure_df,
            "error_L2_velocity_df": error_L2_velocity_df,
            }


    return error_dictionary