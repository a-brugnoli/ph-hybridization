from . import basic_operators
from .system_operators import SystemOperators
from src.problems.problem import Problem
import firedrake as fdrk
from ufl.tensors import ListTensor 
from firedrake.petsc import PETSc


class MaxwellMixed(SystemOperators):

    def __init__(self, type, domain, pol_degree):
        super().__init__(type, domain, pol_degree)

        self.normal_versor = fdrk.FacetNormal(self.domain)

        
    def _set_space(self):
        if self.domain.geometric_dimension()==3:
            broken_RT_element = fdrk.BrokenElement(self.RT_element)
            self.broken_RT_space = fdrk.FunctionSpace(self.domain, broken_RT_element)

            if self.type == "primal":
                # The electric field is discretized using broken Raviart-Thomas elements 
                # whereas the magnetic field uses Nedelec elements
                self.mixedspace = self.broken_RT_space * self.NED_space
            else:
                self.mixedspace = self.NED_space * broken_RT_element
        elif self.domain.geometric_dimension()==2:
            raise NotImplementedError("Maxwell works only in 3D")


    def assign_initial_conditions(self, electric_field, magnetic_field):
        if self.type == "primal":

            electric_mixed = fdrk.project(electric_field, self.mixedspace.sub(0))
            magnetic_mixed = fdrk.interpolate(magnetic_field, self.mixedspace.sub(1))
        elif self.type == "dual":

            electric_mixed = fdrk.interpolate(electric_field, self.mixedspace.sub(0))
            magnetic_mixed = fdrk.project(magnetic_field, self.mixedspace.sub(1))
        else:
            raise SyntaxError("Only primal or dual representation")
        
        return (electric_mixed, magnetic_mixed)
    

    def get_boundary_conditions(self, problem: Problem, time: fdrk.Constant):
        
        bc_essential = []

        bc_dictionary, electric_value, magnetic_value = problem.get_boundary_conditions(time)
        if self.type=="primal":
            element_NED_mixed = self.mixedspace.sub(1).ufl_element()
            assert "N1curl" + str(self.pol_degree) in element_NED_mixed

            list_id_magnetic = bc_dictionary["magnetic"]
            for id in list_id_magnetic:
                bc_essential.append(fdrk.DirichletBC(self.mixedspace(1),
                                                      magnetic_value, id))

        elif self.type=="dual":
            element_NED_mixed = self.mixedspace.sub(0).ufl_element()
            assert "N1curl" + str(self.pol_degree) in element_NED_mixed
            
            list_id_electric = bc_dictionary["electric"]
            for id in list_id_electric:
                bc_essential.append(fdrk.DirichletBC(self.mixedspace(0),
                                                      electric_value, id))
            
        else:
            raise SyntaxError("Only primal or dual representation")
        
        return bc_essential
    
  
    def operator_dynamics(self, testfunctions, functions):

        test_electric, test_magnetic = testfunctions
        electric_field, magnetic_field = functions

        mass = fdrk.inner(test_electric, electric_field) * fdrk.dx\
                    + fdrk.inner(test_magnetic, magnetic_field) * fdrk.dx
        
        if self.type=="primal":
            interconnection = fdrk.dot(test_electric, fdrk.curl(magnetic_field)) * fdrk.dx \
                - fdrk.dot(fdrk.curl(test_magnetic), electric_field) * fdrk.dx

        if self.type=="dual":
            interconnection = fdrk.dot(fdrk.curl(test_electric), magnetic_field) * fdrk.dx \
                - fdrk.dot(test_magnetic, fdrk.curl(electric_field)) * fdrk.dx
            
        return mass, interconnection
    
    def operator_control(self, testfunctions, control):
        """
        Returns the operators for maxwell equations
        Parameters
            testfunctions (TestFunctions) : a mixed test function from the appropriate function space
            control (Function) : a control function from the appropriate function space
        """
        
        test_electric, test_magnetic = testfunctions

        if self.type == "primal":
            natural_control = fdrk.dot(fdrk.cross(test_magnetic, control), self.normal_versor) * fdrk.ds
        else: 
            natural_control = -fdrk.dot(fdrk.cross(test_electric, control), self.normal_versor) * fdrk.ds

        return natural_control
    
 
        

    def __str__(self) -> str:
        return f"Maxwell Mixed Operators. Type {self.type}"

    