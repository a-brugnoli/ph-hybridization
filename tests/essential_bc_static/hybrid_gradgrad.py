import firedrake as fdrk
import matplotlib.pyplot as plt
from math import pi

n_elem = 4
deg = 2
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

exact_solution = fdrk.sin(pi*x)*fdrk.sin(pi*y)

forcing = -fdrk.div(fdrk.grad(exact_solution))

b_functional = test_pressure*forcing*fdrk.dx + test_tangentialtrace*fdrk.dot(fdrk.grad(exact_solution), normal_versor)*fdrk.ds

bc_dirichlet = fdrk.DirichletBC(facet_CG_space, fdrk.Constant(0), "on_boundary")

# bc_dirichlet = fdrk.DirichletBC(facet_CG_space, exact_solution, "on_boundary")

n_block_loc = 2
_A = fdrk.Tensor(A_operator)
_F = fdrk.Tensor(b_functional)
# Extracting blocks for Slate expression of the reduced system
A_blocks = _A.blocks
F_blocks = _F.blocks

A_global_operator = A_blocks[n_block_loc, n_block_loc] - A_blocks[n_block_loc, :n_block_loc] \
* A_blocks[:n_block_loc, :n_block_loc].inv * A_blocks[:n_block_loc, n_block_loc]

b_global_functional = F_blocks[n_block_loc] - A_blocks[n_block_loc, :n_block_loc] \
* A_blocks[:n_block_loc, :n_block_loc].inv * F_blocks[:n_block_loc]

# Global solver
global_multiplier = fdrk.Function(facet_CG_space)
linear_global_problem = fdrk.LinearVariationalProblem(A_global_operator, b_global_functional, global_multiplier, bcs=bc_dirichlet)
global_solver =  fdrk.LinearVariationalSolver(linear_global_problem)


global_solver.solve()

solution = fdrk.Function(mixed_space)

# Intermediate expressions
Lambda = fdrk.AssembledVector(global_multiplier)  # Local coefficient vector for Λ
# Local solve expressions
x_h = fdrk.assemble(A_blocks[:n_block_loc, :n_block_loc].inv *
                (F_blocks[:n_block_loc] - A_blocks[:n_block_loc, n_block_loc] * Lambda))

for ii in range(n_block_loc):
    solution.sub(ii).assign(x_h.sub(ii))
solution.sub(n_block_loc).assign(global_multiplier)


fdrk.trisurf(solution.sub(0))

plt.show()