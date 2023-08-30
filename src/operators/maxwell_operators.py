from .system_operators import SystemOperators
from src.problems.problem import Problem
import firedrake as fdrk
from firedrake.petsc import PETSc
from .utils import facet_form

class MaxwellOperators(SystemOperators):

    def __init__(self, discretization, formulation, problem: Problem, pol_degree: int):
        super().__init__(discretization, formulation, problem, pol_degree)
      

    def _set_space(self):
        if self.domain.geometric_dimension()==3:
            broken_RT_element = fdrk.BrokenElement(self.RT_element)
            broken_RT_space = fdrk.FunctionSpace(self.domain, broken_RT_element)

            if self.discretization=="hybrid":
                broken_NED_element = fdrk.BrokenElement(self.NED_element)
                broken_NED_space = fdrk.FunctionSpace(self.domain, broken_NED_element)

                facet_NED_element = self.NED_element[fdrk.facet]
                brokenfacet_NED_element = fdrk.BrokenElement(facet_NED_element)
                self.brokenfacet_NED_space = fdrk.FunctionSpace(self.domain, brokenfacet_NED_element)
                self.facet_NED_space = fdrk.FunctionSpace(self.domain, facet_NED_element)

                self.space_global = self.facet_NED_space

                if self.formulation == "primal":
                    self.mixedspace_local = broken_RT_space * broken_NED_space * self.brokenfacet_NED_space 
                else:
                    self.mixedspace_local = broken_NED_space * broken_RT_space * self.brokenfacet_NED_space 
            
                self.fullspace = self.mixedspace_local * self.space_global
            else:
                if self.formulation == "primal":
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
        try:
            electric = fdrk.interpolate(electric_field_exp, self.fullspace.sub(0))
            magnetic = fdrk.interpolate(magnetic_field_exp, self.fullspace.sub(1))
        except NotImplementedError:
            print("Initial condition cannot be interpolated")
            electric = fdrk.project(electric_field_exp, self.fullspace.sub(0))
            magnetic = fdrk.project(magnetic_field_exp, self.fullspace.sub(1))

        if self.discretization=="hybrid":
            if self.formulation == "primal":
                exact_normaltrace = electric_field_exp
                exact_tangtrace = magnetic_field_exp                    

            else:
                exact_normaltrace = magnetic_field_exp
                exact_tangtrace = electric_field_exp 

            variable_normaltrace = self.project_NED_facet(exact_normaltrace, broken=True)

            try:
                variable_tangentialtrace = fdrk.interpolate(exact_tangtrace, self.space_global)
            except NotImplementedError:
                PETSc.Sys.Print("Tangential trace cannot be interpolated, project on the appropriate space")     
                variable_tangentialtrace = self.project_NED_facet(exact_tangtrace, broken=False)

            return (electric, magnetic, variable_normaltrace, variable_tangentialtrace)
        else:
            return (electric, magnetic)

    

    def essential_boundary_conditions(self, problem: Problem, time: fdrk.Constant):
        
        bc_dictionary = problem.get_boundary_conditions(time)

        if self.formulation=="primal":
            
            tuple_bc_data = bc_dictionary["magnetic"]
            
            if self.discretization=="hybrid":
                space_bc = self.space_global
            else:
                space_bc = self.fullspace.sub(1)
            
        else:

            tuple_bc_data = bc_dictionary["electric"]
            
            if self.discretization=="hybrid":
                space_bc = self.space_global
            else:
                space_bc = self.fullspace.sub(0)
        
        global_element = str(space_bc.ufl_element())
        assert f"N1curl{str(self.pol_degree)}" in global_element or "RTCE" in global_element

        list_id_bc = tuple_bc_data[0]
        value_bc = tuple_bc_data[1]

        dict_essential_bc = {"space": space_bc, "value": value_bc, "list_id": list_id_bc}

        return dict_essential_bc


    def natural_boundary_conditions(self, problem: Problem, time: fdrk.Constant):
        bc_dictionary = problem.get_boundary_conditions(time)

        if self.formulation=="primal":
            natural_bc = bc_dictionary["electric"][1]
            
        elif self.formulation=="dual":
            natural_bc = bc_dictionary["magnetic"][1]
        
        return natural_bc
    
    
    def dynamics(self, testfunctions, functions):

        if self.discretization=="hybrid":
            test_electric, test_magnetic, test_normaltrace, test_tangtrace = testfunctions
            electric_field, magnetic_field, normaltrace_field, tangtrace_field = functions
        else:
            test_electric, test_magnetic = testfunctions
            electric_field, magnetic_field = functions

        if self.problem.material_coefficients:
            coeff_electric, coeff_magnetic = self.problem.get_material_coefficients()     

            mass = fdrk.inner(test_electric, coeff_electric * electric_field) * fdrk.dx\
                    + fdrk.inner(test_magnetic, coeff_magnetic * magnetic_field) * fdrk.dx
          
        else:
            mass = fdrk.inner(test_electric, electric_field) * fdrk.dx\
                    + fdrk.inner(test_magnetic, magnetic_field) * fdrk.dx
        
        if self.formulation=="primal":
            interconnection = fdrk.dot(test_electric, fdrk.curl(magnetic_field)) * fdrk.dx \
            - fdrk.dot(fdrk.curl(test_magnetic), electric_field) * fdrk.dx
        else:
            interconnection = fdrk.dot(fdrk.curl(test_electric), magnetic_field) * fdrk.dx \
                - fdrk.dot(test_magnetic, fdrk.curl(electric_field)) * fdrk.dx
        
        dynamics = interconnection

        if self.discretization=="hybrid":
            control_global = fdrk.inner(fdrk.cross(test_normaltrace, self.normal_versor), fdrk.cross(tangtrace_field, self.normal_versor))
            control_global_adj = fdrk.inner(fdrk.cross(test_tangtrace, self.normal_versor), fdrk.cross(normaltrace_field, self.normal_versor))

            if self.formulation=="primal":
                
                control_local = fdrk.inner(fdrk.cross(test_magnetic, self.normal_versor), fdrk.cross(normaltrace_field, self.normal_versor))
                control_local_adj = fdrk.inner(fdrk.cross(test_normaltrace, self.normal_versor), fdrk.cross(magnetic_field, self.normal_versor))

                constr_local = facet_form(control_local, self.domain.extruded) - facet_form(control_local_adj, self.domain.extruded)
                constr_global = facet_form(control_global, self.domain.extruded) - facet_form(control_global_adj, self.domain.extruded)

            else:   
                control_local = -fdrk.inner(fdrk.cross(test_electric, self.normal_versor), fdrk.cross(normaltrace_field, self.normal_versor))
                control_local_adj = -fdrk.inner(fdrk.cross(test_normaltrace, self.normal_versor), fdrk.cross(electric_field, self.normal_versor))

                constr_local = facet_form(control_local, self.domain.extruded) - facet_form(control_local_adj, self.domain.extruded)
                constr_global = - facet_form(control_global, self.domain.extruded) + facet_form(control_global_adj, self.domain.extruded)

            dynamics += constr_local + constr_global
        
        return mass, dynamics
    
 
    def control(self, testfunctions, control):
        """
        Returns the forms for maxwell equations
        Parameters
            testfunctions (TestFunctions) : a mixed test function from the appropriate function space
            control (Function) : a control function from the appropriate function space
        """
        if self.discretization=="hybrid":
            test_control = testfunctions[-1]
        else:
            if self.formulation == "primal":
                test_control = testfunctions[1]
            else:
                test_control = testfunctions[0]

        if self.domain.extruded:
            if self.formulation == "primal":
                natural_control = + fdrk.dot(fdrk.cross(test_control, control), self.normal_versor) * fdrk.ds_v \
                                  + fdrk.dot(fdrk.cross(test_control, control), self.normal_versor) * fdrk.ds_t \
                                  + fdrk.dot(fdrk.cross(test_control, control), self.normal_versor) * fdrk.ds_b

            else: 
                natural_control = - fdrk.dot(fdrk.cross(test_control, control), self.normal_versor) * fdrk.ds_v \
                                  - fdrk.dot(fdrk.cross(test_control, control), self.normal_versor) * fdrk.ds_t \
                                  - fdrk.dot(fdrk.cross(test_control, control), self.normal_versor) * fdrk.ds_b
        else:
            if self.formulation == "primal":
                natural_control = + fdrk.dot(fdrk.cross(test_control, control), self.normal_versor) * fdrk.ds
            else: 
                natural_control = - fdrk.dot(fdrk.cross(test_control, control), self.normal_versor) * fdrk.ds

        return natural_control

    

    def project_NED_facet(self, variable_to_project, broken):

        if self.discretization!="hybrid":
            PETSc.Sys.Print("Formulation is not hybrid. Function not available")
            pass

        # project normal trace of field on the broken facet NED space
        if broken:
            trial_function = fdrk.TrialFunction(self.brokenfacet_NED_space)
            test_function = fdrk.TestFunction(self.brokenfacet_NED_space)
        else:
            trial_function = fdrk.TrialFunction(self.facet_NED_space)
            test_function = fdrk.TestFunction(self.facet_NED_space)
            projected_variable = fdrk.Function(self.facet_NED_space)

        a_integrand = fdrk.inner(fdrk.cross(test_function, self.normal_versor), \
                                 fdrk.cross(trial_function, self.normal_versor))
        
        if broken:
            l_integrand = fdrk.inner(fdrk.cross(test_function, self.normal_versor), \
                fdrk.cross(fdrk.cross(variable_to_project, self.normal_versor), self.normal_versor))
        else:
            l_integrand = fdrk.inner(fdrk.cross(test_function, self.normal_versor), \
                                     fdrk.cross(variable_to_project, self.normal_versor))

        a_operator = facet_form(a_integrand, self.domain.extruded)
        l_functional = facet_form(l_integrand, self.domain.extruded)
        
        if broken:
            A_matrix = fdrk.Tensor(a_operator)
            b_vector = fdrk.Tensor(l_functional)
            projected_variable = fdrk.assemble(A_matrix.inv * b_vector)
        else:
            A_mat = fdrk.assemble(a_operator)
            b_vec = fdrk.assemble(l_functional)
            fdrk.solve(A_mat, projected_variable, b_vec)

        return projected_variable


    def trace_norm_NED(self, variable):

        boundary_integrand = self.cell_diameter * fdrk.cross(variable, self.normal_versor) ** 2

        square_norm = facet_form(boundary_integrand, self.domain.extruded)

        return fdrk.sqrt(fdrk.assemble(square_norm))
    
        
    def __str__(self) -> str:
        return f"Maxwell Operators. discretization {self.discretization}, formulation {self.formulation}"

    