import firedrake as fdrk
from src.problems.analytical_maxwell import AnalyticalMaxwell
from src.problems.analytical_wave import AnalyticalWave
import numpy as np
from tqdm import tqdm

n_elements = 5

problem_wave = AnalyticalWave(n_elements, n_elements, n_elements, dim=2, quad=True)
problem_maxwell = AnalyticalMaxwell(n_elements, n_elements, n_elements,quad=True)

time = fdrk.Constant(0)

exact_pressure, exact_velocity = problem_wave.get_exact_solution(time)
exact_electric, exact_magnetic = problem_maxwell.get_exact_solution(time)

residual_first_wave = fdrk.diff(exact_pressure, time) - fdrk.div(exact_velocity)
residual_second_wave = fdrk.diff(exact_velocity, time) - fdrk.grad(exact_pressure)

residual_first_maxwell = fdrk.diff(exact_electric, time) - fdrk.curl(exact_magnetic)
residual_second_maxwell = fdrk.diff(exact_magnetic, time) + fdrk.curl(exact_electric)

time_vec = np.linspace(0, 10, num=11)

tol = 10**(-12)
for t in tqdm(time_vec):

    time.assign(t)


    assert fdrk.norm(residual_first_wave)< tol
    assert fdrk.norm(residual_second_wave)< tol


    assert fdrk.norm(residual_first_maxwell)< tol
    assert fdrk.norm(residual_second_maxwell)< tol