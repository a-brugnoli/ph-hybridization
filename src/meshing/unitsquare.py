import gmsh

def unstructured_unit_square(lc):
    gmsh.initialize()

    # lc = 0.1  # Characteristic mesh size
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    ll = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    pl = gmsh.model.geo.addPlaneSurface([ll])

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    gmsh.model.mesh.generate(2)  # 2D meshing

    gmsh.write("unit_square.msh")

    gmsh.finalize()


unstructured_unit_square(0.1)