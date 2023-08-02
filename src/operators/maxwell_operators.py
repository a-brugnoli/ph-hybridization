from .system_operators import SystemOperators
from src.problems.problem import Problem
import firedrake as fdrk
from ufl.tensors import ListTensor 
from firedrake.petsc import PETSc

class MaxwellOperators(SystemOperators):

    def __init__(self, type_discretization, type_formulation, domain: fdrk.MeshGeometry, pol_degree: int):
        super().__init__(type_discretization, type_formulation, domain, pol_degree)
      

    def _set_space(self):
        if self.domain.geometric_dimension()==3:
            broken_RT_element = fdrk.BrokenElement(self.RT_element)
            broken_RT_space = fdrk.FunctionSpace(self.domain, broken_RT_element)

            if self.type_discretization=="hybrid":
                broken_NED_element = fdrk.BrokenElement(self.NED_element)
                broken_NED_space = fdrk.FunctionSpace(self.domain, broken_NED_element)

                facet_NED_element = self.NED_element[fdrk.facet]
                brokenfacet_NED_element = fdrk.BrokenElement(facet_NED_element)
                self.brokenfacet_NED_space = fdrk.FunctionSpace(self.domain, brokenfacet_NED_element)
                facet_NED_space = fdrk.FunctionSpace(self.domain, facet_NED_element)

                self.space_global = facet_NED_space

                if self.type_formulation == "primal":
                    self.mixedspace_local = broken_RT_space * broken_NED_space * self.brokenfacet_NED_space 
                else:
                    self.mixedspace_local = broken_NED_space * broken_RT_space * self.brokenfacet_NED_space 
            
                self.fullspace = self.mixedspace_local * self.space_global
            else:
                if self.type_formulation == "primal":
                # The electric field is discretized using broken Raviart-Thomas elements 
                # whereas the magnetic field uses Nedelec elements
                    self.fullspace = broken_RT_space * self.NED_space
                else:
                    self.fullspace = self.NED_space * broken_RT_space

        elif self.domain.geometric_dimension()==2:
            raise NotImplementedError("Maxwell works only in 3D")


    def get_initial_conditions(self, expression_initial: tuple):
        electric_field_exp, magnetic_field_exp = expression_initial

        # Interpolation on broken spacs has been fixed in recent versions of firedrake
        cell = self.domain.ufl_cell()
        if 'hexahedron' in str(cell) or "quadrilateral" in str(cell):
            electric = fdrk.project(electric_field_exp, self.fullspace.sub(0))
            magnetic = fdrk.project(magnetic_field_exp, self.fullspace.sub(1))

        else:
            electric = fdrk.interpolate(electric_field_exp, self.fullspace.sub(0))
            magnetic = fdrk.interpolate(magnetic_field_exp, self.fullspace.sub(1))

        if self.type_discretization=="hybrid":
            if self.type_formulation == "primal":
                exact_normaltrace = electric_field_exp
                exact_tangtrace = magnetic_field_exp                    

            else:
                exact_normaltrace = magnetic_field_exp
                exact_tangtrace = electric_field_exp 


            variable_normaltrace = self.project_NED_facetbroken(exact_normaltrace)

            if 'hexahedron' in str(cell) or "quadrilateral" in str(cell):
                variable_tangentialtrace = fdrk.project(exact_tangtrace, self.space_global)
            else:
                variable_tangentialtrace = fdrk.interpolate(exact_tangtrace, self.space_global)
            
            return (electric, magnetic, variable_normaltrace, variable_tangentialtrace)
        else:
            return (electric, magnetic)

    

    def essential_boundary_conditions(self, problem: Problem, time: fdrk.Constant):
        
        essential_bc = []

        bc_dictionary = problem.get_boundary_conditions(time)

        if self.type_formulation=="primal":
            
            tuple_bc_data = bc_dictionary["magnetic"]
            
            if self.type_discretization=="hybrid":
                space_bc = self.space_global
            else:
                space_bc = self.fullspace.sub(1)
            
        else:

            tuple_bc_data = bc_dictionary["electric"]
            
            if self.type_discretization=="hybrid":
                space_bc = self.space_global
            else:
                space_bc = self.fullspace.sub(0)
        
        global_element = str(space_bc.ufl_element())
        print(global_element)
        assert f"N1curl{str(self.pol_degree)}" in global_element or f"RTCE" in global_element

        list_id_bc = tuple_bc_data[0]
        value_bc = tuple_bc_data[1]

        for id in list_id_bc:
                essential_bc.append(fdrk.DirichletBC(space_bc, value_bc, id))
        
        return essential_bc
    

    def natural_boundary_conditions(self, problem: Problem, time: fdrk.Constant):
        bc_dictionary = problem.get_boundary_conditions(time)

        if self.type_formulation=="primal":
            natural_bc = bc_dictionary["electric"][1]
            
        elif self.type_formulation=="dual":
            natural_bc = bc_dictionary["magnetic"][1]
        
        return natural_bc
    
    
    def dynamics(self, testfunctions, functions):

        if self.type_discretization=="hybrid":
            test_electric, test_magnetic, test_normaltrace, test_tangtrace = testfunctions
            electric_field, magnetic_field, normaltrace_field, tangtrace_field = functions
        else:
            test_electric, test_magnetic = testfunctions
            electric_field, magnetic_field = functions

        mass = fdrk.inner(test_electric, electric_field) * fdrk.dx\
                    + fdrk.inner(test_magnetic, magnetic_field) * fdrk.dx
        
        if self.type_formulation=="primal":
            interconnection = fdrk.dot(test_electric, fdrk.curl(magnetic_field)) * fdrk.dx \
            - fdrk.dot(fdrk.curl(test_magnetic), electric_field) * fdrk.dx
        else:
            interconnection = fdrk.dot(fdrk.curl(test_electric), magnetic_field) * fdrk.dx \
                - fdrk.dot(test_magnetic, fdrk.curl(electric_field)) * fdrk.dx
        
        dynamics = interconnection

        if self.type_discretization=="hybrid":
            control_global = fdrk.inner(fdrk.cross(test_normaltrace, self.normal_versor), fdrk.cross(tangtrace_field, self.normal_versor))
            control_global_adj = fdrk.inner(fdrk.cross(test_tangtrace, self.normal_versor), fdrk.cross(normaltrace_field, self.normal_versor))

            if self.type_formulation=="primal":
                
                control_local = fdrk.inner(fdrk.cross(test_magnetic, self.normal_versor), fdrk.cross(normaltrace_field, self.normal_versor))
                control_local_adj = fdrk.inner(fdrk.cross(test_normaltrace, self.normal_versor), fdrk.cross(magnetic_field, self.normal_versor))

                constr_local = (control_local('+') + control_local('-')) * fdrk.dS + control_local * fdrk.ds \
                             - ((control_local_adj('+') + control_local_adj('-')) * fdrk.dS + control_local_adj * fdrk.ds)
                
                constr_global = (control_global('+') + control_global('-')) * fdrk.dS + control_global * fdrk.ds \
                                - ((control_global_adj('+') + control_global_adj('-')) * fdrk.dS + control_global_adj * fdrk.ds)

            else:   
                control_loc = -fdrk.inner(fdrk.cross(test_electric, self.normal_versor), fdrk.cross(normaltrace_field, self.normal_versor))
                control_local_adj = -fdrk.inner(fdrk.cross(test_normaltrace, self.normal_versor), fdrk.cross(electric_field, self.normal_versor))

                constr_local = ((control_loc('+') + control_loc('-')) * fdrk.dS + control_loc * fdrk.ds) \
                             - ((control_local_adj('+') + control_local_adj('-')) * fdrk.dS + control_local_adj * fdrk.ds)
                
                constr_global = -((control_global('+') + control_global('-')) * fdrk.dS + control_global * fdrk.ds) \
                                + ((control_global_adj('+') + control_global_adj('-')) * fdrk.dS + control_global_adj * fdrk.ds)

            dynamics += constr_local + constr_global
        
        return mass, dynamics
    
 
    def control(self, testfunctions, control):
        """
        Returns the forms for maxwell equations
        Parameters
            testfunctions (TestFunctions) : a mixed test function from the appropriate function space
            control (Function) : a control function from the appropriate function space
        """
        if self.type_discretization=="hybrid":
            test_control = testfunctions[-1]
        else:
            if self.type_formulation == "primal":
                test_control = testfunctions[1]
            else:
                test_control = testfunctions[0]


        if self.type_formulation == "primal":
            natural_control = + fdrk.dot(fdrk.cross(test_control, control), self.normal_versor) * fdrk.ds
        else: 
            natural_control = - fdrk.dot(fdrk.cross(test_control, control), self.normal_versor) * fdrk.ds

        return natural_control

    

    def project_NED_facetbroken(self, variable_to_project):

        if self.type_discretization!="hybrid":
            PETSc.Sys.Print("Formulation is not hybrid. Function not available")
            pass

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


    def trace_norm_NED(self, variable):

        boundary_integrand = self.cell_diameter * fdrk.cross(variable, self.normal_versor) ** 2
        return fdrk.sqrt(fdrk.assemble((boundary_integrand('+') + boundary_integrand('-')) * fdrk.dS + boundary_integrand * fdrk.ds))
    
        
    def __str__(self) -> str:
        return f"Maxwell Operators. Discretization {self.type_discretization}, Formulation {self.type_formulation}"

    