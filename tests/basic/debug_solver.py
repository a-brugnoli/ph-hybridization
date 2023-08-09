import firedrake as fdrk
from src.solvers.hamiltonian_solver import HamiltonianWaveSolver
from firedrake.petsc import PETSc

def debug_wave(solver: HamiltonianWaveSolver, time_new_test, tol=1e-9):
    time_step = solver.time_step

    assert abs(float(solver.time_new) - time_new_test) < tol
    assert abs(float(solver.time_midpoint) - (time_new_test - time_step/2)) < tol
    assert abs(float(solver.time_old) - (time_new_test - time_step)) < tol

    norm_versor = solver.problem.normal_versor
    formulation = solver.operators.formulation
    discretization = solver.operators.discretization

    pressure_exact_midpoint, velocity_exact_midpoint = solver.problem.get_exact_solution(solver.time_midpoint)
    pressure_exact_new, velocity_exact_new = solver.problem.get_exact_solution(solver.time_new)

    test_pressure, test_velocity = solver.tests[0:2]

    pressure_old, velocity_old = solver.state_old.subfunctions[0:2]
    pressure_midpoint, velocity_midpoint = solver.state_midpoint.subfunctions[0:2]
    pressure_new, velocity_new = solver.state_new.subfunctions[0:2]

    bc_type = solver.problem.bc_type

    natural_control = solver.operators.natural_boundary_conditions(solver.problem, time=solver.time_midpoint)

    if formulation =="primal":
        assert fdrk.norm(natural_control - pressure_exact_midpoint)< tol

        residual_pressure = fdrk.inner(test_pressure, (pressure_new-pressure_old)/time_step)*fdrk.dx \
                            -fdrk.inner(test_pressure, fdrk.div(velocity_midpoint))*fdrk.dx
            
        residual_velocity = fdrk.inner(test_velocity, (velocity_new-velocity_old)/time_step)*fdrk.dx \
                            +fdrk.inner(fdrk.div(test_velocity), pressure_midpoint) *fdrk.dx
            
        if bc_type == "dirichlet":
            if discretization == "mixed":
                residual_velocity -=fdrk.inner(fdrk.dot(test_velocity, norm_versor), natural_control)*fdrk.ds
            else:
                normaltrace_pessure_midpoint = solver.state_midpoint.subfunctions[2]
                control_local = fdrk.inner(fdrk.dot(test_velocity, norm_versor), fdrk.dot(normaltrace_pessure_midpoint, norm_versor))
                residual_velocity -=(control_local('+') + control_local('-')) * fdrk.dS + control_local * fdrk.ds
                
                # PETSc.Sys.Print("Warning: modifyng control debug")
                # residual_velocity -=fdrk.inner(fdrk.dot(test_velocity, norm_versor), natural_control)*fdrk.ds
        else:
            PETSc.Sys.Print("WARNING: debug essential conditions for primal system to be implemented")

    if formulation == "dual":
        assert fdrk.norm(natural_control - velocity_exact_midpoint)< tol

        residual_pressure = fdrk.inner(test_pressure, (pressure_new-pressure_old)/time_step)*fdrk.dx \
                            +fdrk.inner(fdrk.grad(test_pressure), velocity_midpoint)*fdrk.dx 
        
        residual_velocity = fdrk.inner(test_velocity, (velocity_new-velocity_old)/time_step)*fdrk.dx \
                            -fdrk.inner(test_velocity, fdrk.grad(pressure_midpoint) ) *fdrk.dx
        
        if bc_type == "neumann":
            if discretization == "mixed":
                residual_pressure -=fdrk.inner(test_pressure, fdrk.dot(natural_control, norm_versor))*fdrk.ds
            else:
                normaltrace_velocity_midpoint = solver.state_midpoint.subfunctions[2]
                control_local = fdrk.inner(test_pressure, normaltrace_velocity_midpoint)
                residual_pressure -=(control_local('+') + control_local('-')) * fdrk.dS + control_local * fdrk.ds

                # PETSc.Sys.Print("Warning: modifyng control debug")
                # residual_pressure -=fdrk.inner(test_pressure, fdrk.dot(natural_control, norm_versor))*fdrk.ds

        else:
            PETSc.Sys.Print("WARNING: Debug essential conditions for dual system to be implemented")


    max_res_pressure = fdrk.assemble(residual_pressure).vector().max()
    PETSc.Sys.Print(f"Max Residual pressure equation {formulation} dicretization {discretization}: {max_res_pressure}")
    max_res_velocity = fdrk.assemble(residual_velocity).vector().max()
    PETSc.Sys.Print(f"Max Residual velocity equation {formulation} dicretization {discretization}: {max_res_velocity}")

    assert max_res_pressure < tol and max_res_velocity < tol

    PETSc.Sys.Print(f"Wave {formulation} with bc {bc_type}: PASSED")
