import gmsh

gmsh.initialize()

def fichera_corner(mesh_size):

    outer_cube = gmsh.model.occ.addBox(-1, -1, -1, 2, 2, 2)
    inner_cube = gmsh.model.occ.addBox(-1, -1, -1, 1, 1, 1)
    resulting_geometry = gmsh.model.occ.cut([(3, outer_cube)], [(3, inner_cube)])

    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(3)

    gmsh.write("fichera_corner.msh")

    gmsh.finalize()
