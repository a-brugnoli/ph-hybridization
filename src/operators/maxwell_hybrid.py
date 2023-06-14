from . import basic_operators
from .system_operators import SystemOperators
from src.problems.problem import Problem
import firedrake as fdrk
from ufl.tensors import ListTensor 
from firedrake.petsc import PETSc


class MaxwellHybrid(SystemOperators):

    def __init__(self, type, domain: fdrk.MeshGeometry, pol_degree: int):
        super().__init__(type, domain, pol_degree)

        self.normal_versor = fdrk.FacetNormal(self.domain)

        
    def _set_space(self):
        if self.domain.geometric_dimension()==3:
            broken_RT_element = fdrk.BrokenElement(self.RT_element)
            self.broken_RT_space = fdrk.FunctionSpace(self.domain, broken_RT_element)

            facet_NED_element = self.NED_element[fdrk.facet]
            broken_NED_element = fdrk.BrokenElement(self.NED_element)
            brokenfacet_NED_element = fdrk.BrokenElement(facet_NED_element)

            self.broken_NED_space = fdrk.FunctionSpace(self.domain, broken_NED_element)
            self.brokenfacet_NED_space = fdrk.FunctionSpace(self.domain, brokenfacet_NED_element)

            self.facet_NED_space = fdrk.FunctionSpace(self.domain, facet_NED_element)

            self.space_global = self.facet_NED_space

            if self.type == "primal":
                # The electric field is discretized using broken Raviart-Thomas elements 
                # whereas the magnetic field uses Nedelec elements

                self.mixedspace_local = self.broken_RT_space * self.broken_NED_space \
                                        * self.brokenfacet_NED_space 
            else:
                self.mixedspace_local = self.broken_NED_space * self.broken_RT_space, \
                    * self.brokenfacet_NED_space 
            
            self.hybridspace = self.mixedspace_local * self.space_global

        elif self.domain.geometric_dimension()==2:
            raise NotImplementedError("Maxwell works only in 3D")


    def assign_initial_conditions(self, electric_field, magnetic_field):
        electric_hybrid = fdrk.project(electric_field, self.mixedspace_local.sub(0))
        magnetic_hybrid = fdrk.project(magnetic_field, self.mixedspace_local.sub(1))

        if self.type == "primal":
            variable_normaltrace = self.project_NED_facetbroken(electric_field)
            variable_tangentialtrace = fdrk.interpolate(magnetic_field, self.space_global)
        
        elif self.type == "dual":
            variable_normaltrace = self.project_NED_facetbroken(magnetic_field)
            variable_tangentialtrace = fdrk.interpolate(electric_field, self.space_global)

        else:
            raise SyntaxError("Only primal or dual representation")

        return (electric_hybrid, magnetic_hybrid, variable_normaltrace, variable_tangentialtrace)

    

    def get_boundary_conditions(self, problem: Problem, time: fdrk.Constant):
        global_element = str(self.space_global.ufl_element())
        assert "N1curl" + str(self.pol_degree) in global_element
        
        bc_essential = []

        bc_dictionary, electric_value, magnetic_value = problem.get_boundary_conditions(time)
        if self.type=="primal":
            list_id_magnetic = bc_dictionary["magnetic"]
            for id in list_id_magnetic:
                bc_essential.append(fdrk.DirichletBC(self.space_global, magnetic_value, id))

        elif self.type=="dual":
            list_id_electric = bc_dictionary["electric"]
            for id in list_id_electric:
                bc_essential.append(fdrk.DirichletBC(self.space_global,
                                                      electric_value, id))
        else:
            raise SyntaxError("Only primal or dual representation")
        
        return bc_essential
    
    
    def operator_dynamics(self, testfunctions, functions):

        test_electric, test_magnetic, test_normaltrace, test_tangtrace = testfunctions
        electric_field, magnetic_field, normaltrace_field, tangtrace_field = functions

        mass = fdrk.inner(test_electric, electric_field) * fdrk.dx\
                    + fdrk.inner(test_magnetic, magnetic_field) * fdrk.dx
        
        control_global = fdrk.inner(fdrk.cross(test_normaltrace, self.normal_versor), \
                                    fdrk.cross(tangtrace_field, self.normal_versor))
        control_global_adj = fdrk.inner(fdrk.cross(test_tangtrace, self.normal_versor), \
                                    fdrk.cross(normaltrace_field, self.normal_versor))

        if self.type=="primal":
            interconnection = fdrk.dot(test_electric, fdrk.curl(magnetic_field)) * fdrk.dx \
                - fdrk.dot(fdrk.curl(test_magnetic), electric_field) * fdrk.dx

            control_local = fdrk.inner(fdrk.cross(test_magnetic, self.normal_versor), \
                                       fdrk.cross(normaltrace_field, self.normal_versor))
            control_local_adj = fdrk.inner(fdrk.cross(test_normaltrace, self.normal_versor), \
                                           fdrk.cross(magnetic_field, self.normal_versor))

            constr_local = (control_local('+') + control_local('-')) * fdrk.dS \
                  + control_local * fdrk.ds \
                - ((control_local_adj('+') + control_local_adj('-')) * fdrk.dS \
                   + control_local_adj * fdrk.ds)
            
            
            constr_global = (control_global('+') + control_global('-')) * fdrk.dS \
                        + control_global * fdrk.ds \
                        - ((control_global_adj('+') + control_global_adj('-')) * fdrk.dS \
                        + control_global_adj * fdrk.ds)
            
    
        if self.type=="dual":
            interconnection = fdrk.dot(fdrk.curl(test_electric), magnetic_field) * fdrk.dx \
                - fdrk.dot(test_magnetic, fdrk.curl(electric_field)) * fdrk.dx
            
            control_loc = -fdrk.inner(fdrk.cross(test_electric, self.normal_versor), \
                                       fdrk.cross(normaltrace_field, self.normal_versor))
            control_local_adj = -fdrk.inner(fdrk.cross(test_normaltrace, self.normal_versor), \
                                       fdrk.cross(electric_field, self.normal_versor))

            constr_local = (control_loc('+') + control_loc('-')) * fdrk.dS \
                + control_loc * fdrk.ds \
                - ((control_local_adj('+') + control_local_adj('-')) * fdrk.dS \
                + control_local_adj * fdrk.ds)
            
            constr_global = -(control_global('+') + control_global('-')) * fdrk.dS \
                        - control_global * fdrk.ds \
                        + ((control_global_adj('+') + control_global_adj('-')) * fdrk.dS \
                        - control_global_adj * fdrk.ds)

        return mass, interconnection, constr_local, constr_global
    
 
    def operator_control(self, testfunctions, control):
        """
        Returns the operators for maxwell equations
        Parameters
            testfunctions (TestFunctions) : a mixed test function from the appropriate function space
            control (Function) : a control function from the appropriate function space
        """
        _, _, _, test_tangtrace = testfunctions

        if self.type == "primal":
            natural_control = fdrk.dot(fdrk.cross(test_tangtrace, control), self.normal_versor) * fdrk.ds
        else: 
            natural_control = -fdrk.dot(fdrk.cross(test_tangtrace, control), self.normal_versor) * fdrk.ds

        return natural_control
    

    def project_NED_facetbroken(self, variable_to_project):
        # project normal trace of field on the broken facet NED space
        trial_function = fdrk.TrialFunction(self.brokenfacet_NED_space)
        test_function = fdrk.TestFunction(self.brokenfacet_NED_space)

        a_form = fdrk.inner(fdrk.cross(test_function, self.normal_versor), \
                            fdrk.cross(trial_function, self.normal_versor))
        a_operator = (a_form('+') + a_form('-')) * fdrk.dS + a_form * fdrk.ds

        l_form = fdrk.inner(fdrk.cross(test_function, self.normal_versor), \
                       fdrk.cross(fdrk.cross(variable_to_project, self.normal_versor), \
                                   self.normal_versor))
        l_functional = (l_form('+') + l_form('-')) * fdrk.dS + l_form * fdrk.ds

        A_matrix = fdrk.Tensor(a_operator)
        b_vector = fdrk.Tensor(l_functional)
        projected_variable = fdrk.assemble(A_matrix.inv * b_vector)

        return projected_variable
        

    def __str__(self) -> str:
        return f"Maxwell Hybrid Operators. Type {self.type}"

    