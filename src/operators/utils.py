import firedrake as fdrk

def facet_form(integrand, extruded):
    if extruded:
        facet_form = (integrand('+') + integrand('-')) * fdrk.dS_v \
                    +(integrand('+') + integrand('-')) * fdrk.dS_h \
                    + integrand * fdrk.ds_v \
                    + integrand * fdrk.ds_tb 
    else:
        facet_form = (integrand('+') + integrand('-')) * fdrk.dS + integrand * fdrk.ds

    return facet_form


