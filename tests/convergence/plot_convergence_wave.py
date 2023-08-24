import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.postprocessing import basic_plotting

bc_case = "mixed" 
discretization = "hybrid"
dim=2
quad=True
directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/Wave/{discretization}_discretization/" \
                  + f"{bc_case}_bc/dimension_{dim}/quad_mesh_{quad}/"

deg_vec  = [1,2,3]

h_list = []

norm = "Linf"
# norm = "L2"
# norm = "Tend"

# Errors

# Primal
error_L2_pressure_primal = []
error_L2_velocity_primal = []

error_Hdiv_velocity_primal = []

error_normal_primal = []
error_tangential_primal = []

# Dual
error_L2_pressure_dual = []
error_L2_velocity_dual = []

error_H1_pressure_dual = []
error_Hcurl_velocity_dual = []

error_normal_dual = []
error_tangential_dual = []

# Dual Field
error_pressure_df = []
error_velocity_df = []

# rates

# Primal
rate_L2_pressure_primal = []
rate_L2_velocity_primal = []

rate_Hdiv_velocity_primal = []

rate_normal_primal = []
rate_tangential_primal = []

# Dual
rate_L2_pressure_dual = []
rate_L2_velocity_dual = []

rate_H1_pressure_dual = []
rate_Hcurl_velocity_dual = []

rate_normal_dual = []
rate_tangential_dual = []

# Dual field
rate_pressure_df = []
rate_velocity_df = []

for count, deg in enumerate(deg_vec):
    file = f"convergence_r={deg}_{norm}.csv"

    df = pd.read_csv(directory_results+file)
    n_elements = df["N"].values

    h_elements = 1./n_elements

    errors_rate = df.columns[df.columns.str.contains('error')]
    columns_rate = df.columns[df.columns.str.contains('rate')]

    h_list.append(h_elements)

    # Primal
    error_L2_pressure_primal.append(df["error_L2_pressure_primal"].values)
    error_L2_velocity_primal.append(df["error_L2_velocity_primal"].values)

    error_Hdiv_velocity_primal.append(df["error_Hdiv_velocity_primal"].values)

    if "error_normal_primal" in df.columns and "error_tangential_primal" in df.columns:
        error_normal_primal.append(df["error_normal_primal"].values)
        error_tangential_primal.append(df["error_tangential_primal"].values)

    # Dual
    error_L2_pressure_dual.append(df["error_L2_pressure_dual"].values)
    error_L2_velocity_dual.append(df["error_L2_velocity_dual"].values)

    error_H1_pressure_dual.append(df["error_H1_pressure_dual"].values)
    error_Hcurl_velocity_dual.append(df["error_Hcurl_velocity_dual"].values)

    if "error_normal_dual" in df.columns and "error_tangential_dual" in df.columns:
        error_normal_dual.append(df["error_normal_dual"].values)
        error_tangential_dual.append(df["error_tangential_dual"].values)


    # Primal/Dual
    error_pressure_df.append(df["error_L2_pressure_df"].values)
    error_velocity_df.append(df["error_L2_velocity_df"].values)


    # rate_L2_pressure_primal.append(df["rate_L2_pressure_primal"].values)
    # rate_L2_velocity_primal.append(df["rate_L2_velocity_primal"].values)

    # rate_Hdiv_velocity_primal.append(df["rate_Hdiv_velocity_primal"].values)

    # rate_normal_primal.append(df["rate_normal_primal"].values)
    # rate_tangential_primal.append(df["rate_tangential_primal"].values)


    # # Print the columns
    # for column in columns_rate:
    #     print(column)
    #     for value in df[column]:
    #         print(value)

    columns_rate_H1 = df.columns[df.columns.str.contains('rate_H1')]

    for column in columns_rate_H1:
        print(column)
        for value in df[column]:
            print(value)

    columns_rate_Hdiv = df.columns[df.columns.str.contains('rate_Hdiv')]

    for column in columns_rate_Hdiv:
        print(column)
        for value in df[column]:
            print(value)


# Plot primal
basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_L2_pressure_primal, 
                                label="DG", 
                                ylabel=r'$\log||\widehat{p}^3_h - \widehat{p}^3_{\mathrm{ex}}||_{L^2}$',
                                title=r'Error $\widehat{p}^3_h$', 
                                save_path=f"{directory_results}error_L2_pressure_primal")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_L2_velocity_primal, 
                                label="RT", 
                                ylabel=r'$\log||\widehat{u}^2_h - \widehat{u}^2_{\mathrm{ex}}||_{L^2}$',
                                title=r'Error $\widehat{u}^2_h$', 
                                save_path=f"{directory_results}error_L2_velocity_primal")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_Hdiv_velocity_primal, 
                                label="RT", 
                                ylabel=r'$\log||\widehat{u}^2_h - \widehat{u}^2_{\mathrm{ex}}||_{H^{\mathrm{div}}}$',
                                title=r'Error $\widehat{u}^2_h$', 
                                save_path=f"{directory_results}error_Hdiv_velocity_primal")

if len(error_normal_primal) > 0 and len(error_tangential_primal)>0:
    basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_normal_primal, 
                                    label="RT",
                                    ylabel=r'$\log|||\widehat{p}^{2, \bm{n}}_h - P_h \widehat{p}^{2, \bm{n}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$',
                                    title=r'Error $\widehat{p}^{2, \bm{n}}_h$', 
                                    save_path=f"{directory_results}error_normal_primal")

    basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_tangential_primal, 
                                    label="RT", 
                                    ylabel=r'$\log|||\widehat{u}^{2, \bm{t}}_h - \widehat{u}^{2, \bm{t}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$',
                                    title=r'Error $\widehat{u}^{2, \bm{t}}_h$', 
                                    save_path=f"{directory_results}error_tangential_primal")
    

# # Plot dual
basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_L2_pressure_dual, 
                                label="CG", 
                                ylabel=r'$\log||{p}^0_h - {p}^0_{\mathrm{ex}}||_{L^2}$',
                                title=r'Error ${p}^0_h$', 
                                save_path=f"{directory_results}error_L2_pressure_dual")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_L2_velocity_dual, 
                                label="NED", 
                                ylabel=r'$\log||{u}^1_h -   {u}^1_{\mathrm{ex}}||_{L^2}$',
                                title=r'Error ${u}^1_h$', 
                                save_path=f"{directory_results}error_L2_velocity_dual")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_H1_pressure_dual, 
                                label="CG", 
                                ylabel=r'$\log||{p}^0_h - {p}^0_{\mathrm{ex}}||_{H^{1}}$',
                                title=r'Error ${p}^0_h$', 
                                save_path=f"{directory_results}error_H1_pressure_dual")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_Hcurl_velocity_dual, 
                                label="NED", 
                                ylabel=r'$\log||{u}^1_h - {u}^1_{\mathrm{ex}}||_{H^{\mathrm{curl}}}$',
                                title=r'Error ${u}^1_h$', 
                                save_path=f"{directory_results}error_Hcurl_velocity_dual")

if len(error_normal_dual) > 0 and  len(error_tangential_dual) > 0:
    basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_normal_dual, 
                                label="CG",
                                ylabel=r'$\log|||{u}^{0, \bm{n}}_h - P_h {u}^{0, \bm{n}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$',
                                title=r'Error ${u}^{0, \bm{n}}_h$', 
                                save_path=f"{directory_results}error_normal_dual")
    
    basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_tangential_dual, 
                                    label="CG", 
                                    ylabel=r'$\log|||{p}^{0, \bm{t}}_h - {p}^{0, \bm{t}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$',
                                    title=r'Error ${p}^{0, \bm{t}}_h$', 
                                    save_path=f"{directory_results}error_tangential_dual")
    
# # Plot primal/dual
basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_pressure_df, 
                                label="deg",
                                ylabel=r'$\log|| {p}^0_h - \widehat{p}^3_{h} ||_{L^2}$',
                                title=r'Error ${p}^0_h - \widehat{p}^3_{h}$', 
                                save_path=f"{directory_results}error_pressure_df")
                                
basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_velocity_df, 
                                label="deg", 
                                ylabel=r'$\log|| \widehat{u}^2_h - {u}^1_{h} ||_{L^2}$',
                                title=r'Error $\widehat{u}^2_h - {u}^1_{h}$', 
                                save_path=f"{directory_results}error_velocity_df")
    

plt.show()
