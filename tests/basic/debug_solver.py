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
                if solver.operators.domain.extruded:
                    residual_velocity -=fdrk.inner(fdrk.dot(test_velocity, norm_versor), natural_control)*fdrk.ds_v
                    residual_velocity -=fdrk.inner(fdrk.dot(test_velocity, norm_versor), natural_control)*fdrk.ds_tb

                else:
                    residual_velocity -=fdrk.inner(fdrk.dot(test_velocity, norm_versor), natural_control)*fdrk.ds
            else:
                normal_pessure_midpoint, tangential_velocity_midpoint = solver.state_midpoint.subfunctions[2:4]
                test_normal_pressure, test_tangential_velocity = solver.tests[2:4]

                control_local = fdrk.inner(fdrk.dot(test_velocity, norm_versor), fdrk.dot(normal_pessure_midpoint, norm_versor))
                control_local_adj = fdrk.inner(fdrk.dot(test_normal_pressure, norm_versor), fdrk.dot(velocity_midpoint, norm_versor))

                control_global = fdrk.inner(fdrk.dot(test_normal_pressure, norm_versor), fdrk.dot(tangential_velocity_midpoint, norm_versor))
                control_global_adj = fdrk.inner(fdrk.dot(test_tangential_velocity, norm_versor), fdrk.dot(normal_pessure_midpoint, norm_versor))
                
                test_velocity_unbroken = fdrk.TestFunction(solver.operators.RT_space)

                residual_equivalence = fdrk.inner(test_velocity_unbroken, (velocity_new-velocity_old)/time_step)*fdrk.dx \
                            +fdrk.inner(fdrk.div(test_velocity_unbroken), pressure_midpoint) *fdrk.dx 
                
                if solver.operators.domain.extruded:
                    residual_velocity -=(control_local('+') + control_local('-')) * fdrk.dS_v + control_local * fdrk.ds_v
                    residual_velocity -=(control_local('+') + control_local('-')) * fdrk.dS_h + control_local * fdrk.ds_tb

                    form_control = fdrk.inner(fdrk.dot(test_tangential_velocity, norm_versor), natural_control)*fdrk.ds_v \
                                 + fdrk.inner(fdrk.dot(test_tangential_velocity, norm_versor), natural_control)*fdrk.ds_tb

                    residual_equivalence -=fdrk.inner(fdrk.dot(test_velocity_unbroken, norm_versor), natural_control)*fdrk.ds_v
                    residual_equivalence -=fdrk.inner(fdrk.dot(test_velocity_unbroken, norm_versor), natural_control)*fdrk.ds_tb

                else:
                    residual_equivalence -=fdrk.inner(fdrk.dot(test_velocity_unbroken, norm_versor), natural_control)*fdrk.ds

                    form_control = fdrk.inner(fdrk.dot(test_tangential_velocity, norm_versor), natural_control)*fdrk.ds

                    residual_velocity -=(control_local('+') + control_local('-')) * fdrk.dS + control_local * fdrk.ds

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
                if solver.operators.domain.extruded:
                    residual_pressure -=fdrk.inner(test_pressure, fdrk.dot(natural_control, norm_versor))*fdrk.ds_v
                    residual_pressure -=fdrk.inner(test_pressure, fdrk.dot(natural_control, norm_versor))*fdrk.ds_tb

                else:
                    residual_pressure -=fdrk.inner(test_pressure, fdrk.dot(natural_control, norm_versor))*fdrk.ds
            else:
                normal_velocity_midpoint, tangential_pressure_midpoint = solver.state_midpoint.subfunctions[2:4]
                test_normal_velocity, test_tangential_pressure = solver.tests[2:4]

                control_local = fdrk.inner(test_pressure, normal_velocity_midpoint)
                control_local_adj = fdrk.inner(test_normal_velocity, pressure_midpoint)


                control_global = fdrk.inner(test_normal_velocity, tangential_pressure_midpoint)
                control_global_adj = fdrk.inner(test_tangential_pressure, normal_velocity_midpoint)


                test_pressure_unbroken = fdrk.TestFunction(solver.operators.CG_space)

                residual_equivalence = fdrk.inner(test_pressure_unbroken, (pressure_new-pressure_old)/time_step)*fdrk.dx \
                                        +fdrk.inner(fdrk.grad(test_pressure_unbroken), velocity_midpoint)*fdrk.dx 
                
                if solver.operators.domain.extruded:
                    residual_pressure -=(control_local('+') + control_local('-')) * fdrk.dS_v + control_local * fdrk.ds_v
                    residual_pressure -=(control_local('+') + control_local('-')) * fdrk.dS_h + control_local * fdrk.ds_tb

                    form_control = fdrk.inner(test_tangential_pressure, fdrk.dot(natural_control, norm_versor))*fdrk.ds_v \
                                 + fdrk.inner(test_tangential_pressure, fdrk.dot(natural_control, norm_versor))*fdrk.ds_tb
                    
                    residual_equivalence -=fdrk.inner(test_pressure_unbroken, fdrk.dot(natural_control, norm_versor))*fdrk.ds_v 
                    residual_equivalence -=fdrk.inner(test_pressure_unbroken, fdrk.dot(natural_control, norm_versor))*fdrk.ds_tb

                else:
                    residual_pressure -=(control_local('+') + control_local('-')) * fdrk.dS + control_local * fdrk.ds

                    form_control = fdrk.inner(test_tangential_pressure, fdrk.dot(natural_control, norm_versor))*fdrk.ds

                    residual_equivalence -=fdrk.inner(test_pressure_unbroken, fdrk.dot(natural_control, norm_versor))*fdrk.ds


        else:
            PETSc.Sys.Print("WARNING: Debug essential conditions for dual system to be implemented")


    max_res_pressure = fdrk.assemble(residual_pressure).vector().max()
    PETSc.Sys.Print(f"Max Residual pressure equation {formulation} dicretization {discretization}: {max_res_pressure}")
    max_res_velocity = fdrk.assemble(residual_velocity).vector().max()
    PETSc.Sys.Print(f"Max Residual velocity equation {formulation} dicretization {discretization}: {max_res_velocity}")

    assert max_res_pressure < tol and max_res_velocity < tol

    if discretization=="hybrid":

        if solver.operators.domain.extruded:
            residual_normal = (control_local_adj('+') + control_local_adj('-')) * fdrk.dS_v + control_local_adj * fdrk.ds_v \
                            + (control_local_adj('+') + control_local_adj('-')) * fdrk.dS_h + control_local_adj * fdrk.ds_tb \
                            - ((control_global('+') + control_global('-')) * fdrk.dS_v + control_global * fdrk.ds_v) \
                            - ((control_global('+') + control_global('-')) * fdrk.dS_h + control_global * fdrk.ds_tb) 
            
            residual_tangential = (control_global_adj('+') + control_global_adj('-')) * fdrk.dS_v + control_global_adj * fdrk.ds_v \
                                + (control_global_adj('+') + control_global_adj('-')) * fdrk.dS_h + control_global_adj * fdrk.ds_tb - form_control
               
        else:
            residual_normal = (control_local_adj('+') + control_local_adj('-')) * fdrk.dS + control_local_adj * fdrk.ds \
                                                -((control_global('+') + control_global('-')) * fdrk.dS + control_global * fdrk.ds)
            
            residual_tangential = (control_global_adj('+') + control_global_adj('-')) * fdrk.dS + control_global_adj * fdrk.ds - form_control
                
        max_res_normal = fdrk.assemble(residual_normal).vector().max()
        PETSc.Sys.Print(f"Max Residual normal trace equation {formulation} dicretization {discretization}: {max_res_normal}")
        max_res_tangential = fdrk.assemble(residual_tangential).vector().max()
        PETSc.Sys.Print(f"Max Residual tangential equation {formulation} dicretization {discretization}: {max_res_tangential}")
        max_res_equivalence = fdrk.assemble(residual_equivalence).vector().max()
        PETSc.Sys.Print(f"Max Residual equivalence equation {formulation} dicretization {discretization}: {max_res_equivalence}")

        assert max_res_normal < tol and max_res_tangential < tol and max_res_equivalence < tol

    PETSc.Sys.Print(f"Wave {formulation} with bc {bc_type}: PASSED")
