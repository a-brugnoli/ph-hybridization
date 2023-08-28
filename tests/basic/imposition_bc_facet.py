import firedrake as fdrk

n_elem = 4
deg = 1
quad = True

mesh = fdrk.UnitSquareMesh(n_elem, n_elem, quadrilateral=quad)
cell_diameter = fdrk.CellDiameter(mesh)
normal_versor = fdrk.FacetNormal(mesh)
x, y = fdrk.SpatialCoordinate(mesh)
cell = mesh.ufl_cell()

CG_element = fdrk.FiniteElement("CG", cell, deg) 
facet_CG_element = CG_element[fdrk.facet]

CG_space = fdrk.FunctionSpace(mesh, CG_element)
facet_CG_space = fdrk.FunctionSpace(mesh, facet_CG_element)

exact_solution = fdrk.sin(x)*fdrk.sin(y)

projected_facet_exact = fdrk.Function(facet_CG_space)
facet_test = fdrk.TestFunction(facet_CG_space)
facet_trial = fdrk.TrialFunction(facet_CG_space)

fdrk.solve(facet_test*facet_trial*fdrk.ds + facet_test("+")*facet_trial("+")*fdrk.dS
           == facet_test*exact_solution*fdrk.ds
           + facet_test("+")*exact_solution("+")*fdrk.dS, projected_facet_exact)

projected_exact = fdrk.project(exact_solution, CG_space)

boundary_integrand = cell_diameter * (projected_exact - projected_facet_exact) ** 2

square_norm = boundary_integrand('+') * fdrk.dS + boundary_integrand * fdrk.ds

print(f"Error: {fdrk.sqrt(fdrk.assemble(square_norm))}")