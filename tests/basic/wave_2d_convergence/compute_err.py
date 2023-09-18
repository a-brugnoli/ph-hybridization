from firedrake import *
import math
from tqdm import tqdm
from firedrake.petsc import PETSc
from src.meshing.unitsquare import unstructured_unit_square

def dict_error_primal(exact_solution, approximate_solution: Function):

    ex_pressure, ex_velocity = exact_solution
    approx_pressure, approx_velocity = approximate_solution.subfunctions

    L2_error_pressure = norm(ex_pressure - approx_pressure)
    L2_error_velocity = norm(ex_velocity - approx_velocity)
    Hdiv_error_velocity = norm(ex_velocity - approx_velocity, norm_type="Hdiv")
    dict_error = {"L2_error_pressure": L2_error_pressure, "L2_error_velocity": L2_error_velocity, "Hdiv_error_velocity": Hdiv_error_velocity}

    return dict_error


def compute_err(n_elements, degree, time_step = 0.002, t_end = 0.1, quad=False):
    """
    Solve an analytical problem for the 2d wave equation with dirichlet
    """

    # mesh_size = 1/n_elements
    # unstructured_unitsquaremesh(mesh_size)
    # mesh = Mesh("unit_square_mesh.msh")

    mesh = UnitSquareMesh(n_elements, n_elements, quadrilateral=quad)
    x, y = SpatialCoordinate(mesh)
    normal = FacetNormal(mesh)

    def exact_solution(t: Constant):
    # Define exact eigensolution (homogeneous boundary condition)
        g_fun = sin(pi*x)*sin(pi*y)
        omega = pi*sqrt(2)

        # g_fun = sin(pi*x) + sin(pi*y)
        # omega = pi

        grad_g = grad(g_fun)
        
        f_time = sin(omega*t) + cos(omega*t)
        df_dtime = diff(f_time, t)

        exact_pressure = g_fun*df_dtime
        exact_velocity = grad_g*f_time

        return exact_pressure, exact_velocity
    

    def mass_form(test_pressure, pressure, test_velocity, velocity):
        return inner(test_pressure, pressure)*dx + inner(test_velocity, velocity)*dx 
    

    def j_form_div(test_pressure, pressure, test_velocity, velocity):
        return inner(test_pressure, div(velocity))*dx - inner(div(test_velocity), pressure)*dx
    

    # Set spaces (primal formulation)
    cell = mesh.ufl_cell()
    DG_element = FiniteElement("DG", cell, degree-1) 

    if quad:
        RT_element = FiniteElement("RTCF", cell, degree) 
    else:
        RT_element = FiniteElement("RT", cell, degree, variant=f"integral({degree+1})") 

    DG_space = FunctionSpace(mesh, DG_element)
    RT_space = FunctionSpace(mesh, RT_element)

    primal_space = DG_space * RT_space
    test_pressure_primal, test_velocity_primal = TestFunctions(primal_space)
    trial_pressure_primal, trial_velocity_primal = TrialFunctions(primal_space)

    state_primal_old = Function(primal_space)
    pressure_primal_old, velocity_primal_old =  state_primal_old.subfunctions

    state_primal_new = Function(primal_space)

    time = Constant(0)
    exact_state = exact_solution(time)
    exact_pressure, exact_velocity = exact_state

    if quad:
        pressure_primal_old.assign(project(exact_pressure, DG_space))
        velocity_primal_old.assign(project(exact_velocity, RT_space))
    else:
        pressure_primal_old.assign(interpolate(exact_pressure, DG_space))
        velocity_primal_old.assign(interpolate(exact_velocity, RT_space))

    mass_operator = mass_form(test_pressure_primal, trial_pressure_primal, test_velocity_primal, trial_velocity_primal)
    j_div_operator = j_form_div(test_pressure_primal, trial_pressure_primal, test_velocity_primal, trial_velocity_primal)

    a_operator_midpoint = mass_operator - 0.5 * time_step * j_div_operator

    mass_functional = mass_form(test_pressure_primal, pressure_primal_old, test_velocity_primal, velocity_primal_old)
    j_div_functional = j_form_div(test_pressure_primal, pressure_primal_old, test_velocity_primal, velocity_primal_old)

    time_midpoint = Constant(time_step/2)
    exact_pressure_midpoint, _ = exact_solution(time_midpoint)

    natural_control = dot(test_velocity_primal, normal) * exact_pressure_midpoint * ds

    b_functional_midpoint = mass_functional + 0.5 * time_step * j_div_functional + time_step * natural_control

    primal_problem = LinearVariationalProblem(a_operator_midpoint, b_functional_midpoint, state_primal_new)
    primal_solver = LinearVariationalSolver(primal_problem)

    n_time_iter = math.ceil(t_end/time_step)
    actual_t_end = n_time_iter*time_step

    error_dict_0 = dict_error_primal(exact_state, state_primal_old)
    error_dict_Linf = error_dict_0

    error_dict_L2 = {}
    for key_error_0, value_error_0 in error_dict_0.items():
        error_dict_L2[key_error_0] = value_error_0 ** 2 /2

    for ii in tqdm(range(n_time_iter)):

        primal_solver.solve()
        state_primal_old.assign(state_primal_new)

        actual_time = (ii+1)*time_step
        time.assign(actual_time)

        time_midpoint.assign(actual_time + time_step/2)

        error_dict_actual = dict_error_primal(exact_state, state_primal_old)

        # Computation of the Linfinity and L2 norm in time
        for key_error, value_error_actual in error_dict_actual.items():
            if error_dict_Linf[key_error]< value_error_actual:
                error_dict_Linf[key_error]= value_error_actual

            if abs(actual_time-actual_t_end)>1e-12:
                error_dict_L2[key_error] = error_dict_L2[key_error] + value_error_actual**2
            else:
                error_dict_L2[key_error] = error_dict_L2[key_error] + value_error_actual**2/2

    
    for key_error, value_error_L2 in error_dict_L2.items():
        error_dict_L2[key_error] = math.sqrt(value_error_L2*time_step)

    PETSc.Sys.Print(f"Solution with {n_elements} elements, pol degree {degree} computed")

    error_time = {"Linf": error_dict_Linf, "L2": error_dict_L2, "Tend": error_dict_actual}

    return error_time

