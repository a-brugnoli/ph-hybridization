import firedrake as fdrk
import matplotlib.pyplot as plt

n_elem = 4
deg = 1
quad = True
mesh = fdrk.UnitSquareMesh(n_elem, n_elem, quadrilateral=quad)
normal_versor = fdrk.FacetNormal(mesh)
x, y = fdrk.SpatialCoordinate(mesh)

cell = mesh.ufl_cell()
CG_element = fdrk.FiniteElement("CG", cell, deg) 
broken_CG_element = fdrk.BrokenElement(CG_element)
facet_CG_element = CG_element[fdrk.facet]
brokenfacet_CG_element = fdrk.BrokenElement(facet_CG_element)

broken_CG_space = fdrk.FunctionSpace(mesh, broken_CG_element)
brokenfacet_CG_space = fdrk.FunctionSpace(mesh, brokenfacet_CG_element)
facet_CG_space = fdrk.FunctionSpace(mesh, facet_CG_element)

mixed_space = broken_CG_space * brokenfacet_CG_space * facet_CG_space

test_pressure, test_normaltrace, test_tangentialtrace = fdrk.TestFunctions(mixed_space)
trial_pressure, trial_normaltrace, trial_tangentialtrace = fdrk.TrialFunctions(mixed_space)

control_local = fdrk.inner(test_pressure, trial_normaltrace)
control_local_adj = fdrk.inner(test_normaltrace, trial_pressure)

control_global = fdrk.inner(test_normaltrace, trial_tangentialtrace)
control_global_adj = fdrk.inner(test_tangentialtrace, trial_normaltrace)

constr_local = + (control_local('+') + control_local('-')) * fdrk.dS + control_local * fdrk.ds \
                - ((control_local_adj('+') + control_local_adj('-')) * fdrk.dS + control_local_adj * fdrk.ds)

constr_global = + (control_global('+') + control_global('-')) * fdrk.dS + control_global * fdrk.ds \
                - ((control_global_adj('+') + control_global_adj('-')) * fdrk.dS + control_global_adj * fdrk.ds)

A_operator = fdrk.inner(fdrk.grad(test_pressure), fdrk.grad(trial_pressure)) * fdrk.dx \
                    - constr_local - constr_global

exact_solution = fdrk.sin(x)*fdrk.sin(y)

forcing = -fdrk.div(fdrk.grad(exact_solution))

b_functional = test_pressure*forcing*fdrk.dx 

solution = fdrk.Function(mixed_space)

bc_dirichlet = fdrk.DirichletBC(mixed_space.sub(2), exact_solution, "on_boundary")
problem = fdrk.LinearVariationalProblem(A_operator, b_functional, solution, bcs=bc_dirichlet)
solver =  fdrk.LinearVariationalSolver(problem)
solver.solve()

fdrk.trisurf(solution.sub(0))

plt.show()