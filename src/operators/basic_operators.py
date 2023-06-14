"""
Module containing the basic form need to construct the Navier-Stokes and Maxwell systems and their coupling.
Dimension of the manifold is n
"""
import ufl
import firedrake as fdrk

dx = ufl.Measure("dx")
ds = ufl.Measure("ds")


def mass(test_k, trial_k):
    """
    Return a variational form associated to a generic inner product

        Parameters: 
            test_k  (TestFunction) : a k form test function 
            trial_k (TrialFunction or Function): a k form (trial) function

        Returns:
            mass : form representing a generic inner product
                
    """
    
    mass = ufl.inner(test_k, trial_k)*dx
    return mass


def rotational(test, trial, rotor):
    """
    Return a variational form associated to the rotational form

    rotational = (test_k, func \cross trial_k)

        Parameters: 
            test_k  (TestFunction) : a test function 
            trial_k (TrialFunction or Function): a (trial) function
            rotor (TrialFunction or Function) : a (trial) function that undergoes the hat map (can be vector can be scalar epending on the geometric dimension)

        Returns:
            rotational : form representing the rotational term
                
    """
    if test.ufl_shape==():
        assert rotor.ufl_shape[0]==2 and trial.ufl_shape[0]==2
        rotational = ufl.inner(trial, _cross2d(test, rotor))*dx
    elif test.ufl_shape[0]==2:
        if rotor.ufl_shape==():
            rotational = ufl.inner(test,  _cross2d(rotor, trial))*dx
        elif trial.ufl_shape==():
            rotational = ufl.inner(test,  _cross2d(-trial, rotor))*dx  # Minus sign in this case on the scalar function
        else: raise("Dimensions are not compatible for rotational operator")
    else: 
        rotational = ufl.inner(test,  ufl.cross(rotor, trial))*dx
    
    return rotational


def strong_grad(test_1, trial_0):
    """
    Return a variational form associated to a strong gradient operator. 

        Parameters: 
            test_1  (TestFunction) : a 1 form test function 
            trial_0 (TrialFunction or Function): a 0 form (trial) function

        Returns:
            stgrad : form representing a strong gradient
                
    """
    stgrad = ufl.inner(test_1, ufl.grad(trial_0))*dx
    return stgrad


def strong_curl(test, trial):
    """
    Return a variational form associated to a strong curl operator. 

        Parameters: 
            test  (TestFunction) : a form test function (a 2 form or a 1 form)
            trial (TrialFunction or Function): a form (trial) function (can be a 1 form or a 0 form)

        Returns:
            stcurl : form representing a strong 3d curl
                
    """
    
    if trial.ufl_shape==():
        # The trial function is a scalar. The curl is given by the rotated gradient vector rot grad = (dy, -dx)^T
        stcurl = ufl.inner(test, _rotgrad2d(trial))*dx
    else:
        if trial.ufl_shape[0]==2:
            # The trial function is 2 dimensional. The curl is given by the row vector curl2d = (-dy, dx)
            stcurl = ufl.inner(test, _curl2d(trial))*dx
        else:
            # The trial function is 3 dimensional.
            stcurl = ufl.inner(test, ufl.curl(trial))*dx

    return stcurl


def strong_div(test_n, trial_1n):
    """
    Return a variational form associated to a strong divergence operator. 

        Parameters: 
            test_n  (TestFunction) : a n form test function 
            trial_1n (TrialFunction or Function): a n-1 form (trial) function

        Returns:
            stdiv : form representing a strong divergence
                
    """
    stdiv = ufl.inner(test_n, ufl.div(trial_1n))*dx
    return stdiv


def weak_grad(test_1n, trial_n):
    """
    Return a variational form associated to a weak gradient operator. 
    
        Parameters: 
            test_1n  (TestFunction) : a n-1 form test function 
            trial_n (TrialFunction or Function): a n form (trial) function

        Returns:
            wkgrad : form representing a weak gradient
                
    """

    wkgrad = ufl.inner( - ufl.div(test_1n), trial_n)*dx
    return wkgrad


def weak_curl(test, trial):
    """
    Return a variational form associated to a weak curl operator
    
        Parameters: 
            test  (TestFunction) : a form test function (can be a 1 or a 0 form)
            trial (TrialFunction or Function): a form (trial) function (can be a 2 or a 1 form )

        Returns:
            wkcurl : form representing a weak curl
                
    """
    if trial.ufl_shape==():
        # The trial fnuction is a scalar field
        wkcurl = ufl.inner(_curl2d(test), trial)*dx
    else: 
        if trial.ufl_shape[0]==2:
            # The trial fnuction is 2 dimensional 
            wkcurl = ufl.inner(_rotgrad2d(test), trial)*dx
        else:
            # The trial function is 3 dimensional
            wkcurl = ufl.inner(ufl.curl(test), trial)*dx
    return wkcurl


def weak_div(test_0, trial_1):
    """
    Return a variational form associated to a weak divergence operator. 
    
        Parameters: 
            test_0  (TestFunction) : a 0 form test function 
            trial_1 (TrialFunction or Function): a 1 form (trial) function

        Returns:
            wkdiv : form representing a weak divergence
                
    """

    wkdiv = ufl.inner(- ufl.grad(test_0), trial_1)*dx
    return wkdiv


def control_grad(test_1n, control_0):
    """
    Return a control operator associated with the integration by parts of the grad operator
    Parameters: 
            test_1n  (TestFunction) : a n-1 form test function 
            control_0 (Function): a 0 control function

        Returns:
            ctrl_grad : linear form representing a grad control
          
    """
        
    normal_vector = fdrk.FacetNormal(test_1n.ufl_function_space().ufl_domain())

    ctrl_grad = ufl.inner(test_1n, control_0*normal_vector)*ds

    return ctrl_grad


def control_curl(test, control):
    """
    Return a control operator associated with the integration by parts of the curl operator
    Parameters: 
            test  (TestFunction) : a form test function 
            control (Function): a control function

        Returns:
            ctrl_curl : linear form representing a curl control
          
    """
    
    normal_vector = fdrk.FacetNormal(test.ufl_function_space().ufl_domain())

    if test.ufl_shape==():
        # The test fnuction is a scalar field
        ctrl_curl = ufl.inner(_cross2d(test, normal_vector), control)*ds
    else: 
        if test.ufl_shape[0]==2:
            # The test fnuction is 2 dimensional 
            ctrl_curl = ufl.inner(test, _cross2d(control, normal_vector))*ds
        else:
            # The test function is 3 dimensional
            ctrl_curl = ufl.inner(test, ufl.cross(normal_vector, control))*ds

    return ctrl_curl


def control_div(test_0, control_1n):
    """
    Return a control operator associated with the integration by parts of the div operator
    Parameters: 
            test_0  (TestFunction) : a 0 form test function 
            control_1n (Function): a n-1 control function

        Returns:
            ctrl_div : linear form representing a div control
          
    """
        
    normal_vector = fdrk.FacetNormal(test_0.ufl_function_space().ufl_domain())

    ctrl_div = ufl.inner(test_0*normal_vector, control_1n)*ds

    return ctrl_div


# Utilities
def _cross2d(scalar_field, vector_field):
        """
        Compute the 2d cross of a scalar field (vector directed along z) and a vector field

        cross2d(scalar, vector) = scalar*[-vector_y, 
                                           vector_x]
        """

        return ufl.as_vector([-scalar_field*vector_field[1], scalar_field*vector_field[0]])


def _rotgrad2d(fun):
    """
    The rotated gradient is define as the the curl of a scalar field directed along z in 2+1 dimensions
    rotgrad2d = (dy, -dx)^T
    """

    assert fun.ufl_shape==()

    return ufl.as_vector([fun.dx(1), -fun.dx(0)])


def _curl2d(fun):
    """
    The curl 2d is defined as the curl of a planar vector field f = (fx, fy) in 2+1 dimensions
    curl2d = (-dy, dx)
    """

    assert fun.ufl_shape[0]==2
    curl_2d = - fun[0].dx(1) + fun[1].dx(0)
    return curl_2d