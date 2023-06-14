import firedrake as fdrk
import ufl

def deRhamElements(domain, pol_degree):
    """
    Trimmed polynomial finite element de Rham sequence
    """
    cell = domain.ufl_cell()
    cont_element = ufl.FiniteElement("CG", cell, pol_degree) 

    if domain.geometric_dimension()==3:
        if str(cell)=='hexahedron':
            tang_cont_element = ufl.FiniteElement("NCE", cell, pol_degree)
            nor_cont_element = ufl.FiniteElement("NCF", cell, pol_degree)
        else:
            tang_cont_element = ufl.FiniteElement("N1curl", cell, pol_degree, variant='point') 
            nor_cont_element = ufl.FiniteElement("RT", cell, pol_degree, variant='integral')
            # tang_cont_element = ufl.FiniteElement("N1curl", cell, pol_degree) 
            # nor_cont_element = ufl.FiniteElement("RT", cell, pol_degree)
            
    else:
        if str(cell)=='quadrilateral':
            tang_cont_element = ufl.FiniteElement("RTCE", cell, pol_degree)
            nor_cont_element = ufl.FiniteElement("RTCF", cell, pol_degree)
        else:
            # tang_cont_element = ufl.FiniteElement("N1curl", cell, pol_degree, variant='point') 
            # nor_cont_element = ufl.FiniteElement("RT", cell, pol_degree, variant='point')
            tang_cont_element = ufl.FiniteElement("N1curl", cell, pol_degree) 
            nor_cont_element = ufl.FiniteElement("RT", cell, pol_degree)
            
    disc_element  = ufl.FiniteElement("DG", cell, pol_degree-1) 
    
    dict_elements = {"continuous": cont_element, 
                     "tangential continuous": tang_cont_element, 
                     "normal continuous": nor_cont_element, 
                     "discontinuous": disc_element}
    return dict_elements

def deRhamSpaces(domain, pol_degree):

    cont_element, tang_cont_element, nor_cont_element, disc_element = \
        deRhamElements(domain, pol_degree).values()


    cont_space = fdrk.FunctionSpace(domain, cont_element)
    tang_cont_space = fdrk.FunctionSpace(domain, tang_cont_element)
    nor_cont_space = fdrk.FunctionSpace(domain, nor_cont_element)
    disc_space = fdrk.FunctionSpace(domain, disc_element)

    dict_spaces = {"continuous": cont_space, "tangential continuous": tang_cont_space, "normal continuous": nor_cont_space, "discontinuous": disc_space}
    return dict_spaces
