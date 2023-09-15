import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.postprocessing import basic_plotting

bc_case = "mixed" #input("Enter the boundary conditions (electric, magnetic, mixed):")
discretization = "mixed"
dim=3
quad=False
directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/" \
                   +f"Maxwell/{discretization}_discretization/" \
                   +f"{bc_case}_bc/dimension_{dim}/quad_mesh_{quad}/"

deg_vec  = [1,2]

h_list = []

# norm = "Linf"
# norm = "L2"
norm = "Tend"

# Errors

# Primal
error_L2_electric_primal = []
error_L2_magnetic_primal = []

error_Hdiv_electric_primal = []
error_Hcurl_magnetic_primal = []

error_normal_primal = []
error_tangential_primal = []

# Dual
error_L2_electric_dual = []
error_L2_magnetic_dual = []

error_Hcurl_electric_dual = []
error_Hdiv_magnetic_dual = []

error_normal_dual = []
error_tangential_dual = []

# Dual Field
error_electric_df = []
error_magnetic_df = []

# rates

# Primal
rate_L2_electric_primal = []
rate_L2_magnetic_primal = []

rate_Hdiv_electric_primal = []
rate_Hcurl_magnetic_primal = []

rate_normal_primal = []
rate_tangential_primal = []

# Dual
rate_L2_electric_dual = []
rate_L2_magnetic_dual = []

rate_Hcurl_electric_dual = []
rate_Hdiv_magnetic_dual = []

rate_normal_dual = []
rate_tangential_dual = []

# Dual field
rate_electric_df = []
rate_magnetic_df = []

for count, deg in enumerate(deg_vec):
    file = f"convergence_r={deg}_{norm}.csv"

    df = pd.read_csv(directory_results+file)
    n_elements = df["N"].values

    h_elements = 1./n_elements

    errors_rate = df.columns[df.columns.str.contains('error')]
    columns_rate = df.columns[df.columns.str.contains('rate')]

    h_list.append(h_elements)

    # Primal
    error_L2_electric_primal.append(df["error_L2_electric_primal"].values)
    error_L2_magnetic_primal.append(df["error_L2_magnetic_primal"].values)

    error_Hdiv_electric_primal.append(df["error_Hdiv_electric_primal"].values)
    error_Hcurl_magnetic_primal.append(df["error_Hcurl_magnetic_primal"].values)

    if "error_normal_primal" in df.columns and "error_tangential_primal" in df.columns:
        error_normal_primal.append(df["error_normal_primal"].values)
        error_tangential_primal.append(df["error_tangential_primal"].values)

    # Dual
    error_L2_electric_dual.append(df["error_L2_electric_dual"].values)
    error_L2_magnetic_dual.append(df["error_L2_magnetic_dual"].values)

    error_Hcurl_electric_dual.append(df["error_Hcurl_electric_dual"].values)
    error_Hdiv_magnetic_dual.append(df["error_Hdiv_magnetic_dual"].values)

    if "error_normal_dual" in df.columns and "error_tangential_dual" in df.columns:
        error_normal_dual.append(df["error_normal_dual"].values)
        error_tangential_dual.append(df["error_tangential_dual"].values)


    # Primal/Dual
    error_electric_df.append(df["error_L2_electric_df"].values)
    error_magnetic_df.append(df["error_L2_magnetic_df"].values)

    # # Print the columns
    # for column in errors_rate:
    #     print(column)
    #     for value in df[column]:
    #         print(value)

    # # Print the columns
    # for column in columns_rate:
    #     print(column)
    #     for value in df[column]:
    #         print(value)

    columns_rate_Hcurl = df.columns[df.columns.str.contains('rate_Hcurl')]

    for column in columns_rate_Hcurl:
        print(column)
        for value in df[column]:
            print(value)

# Plot primal
basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_L2_electric_primal, 
                                label="RT", 
                                ylabel=r'$\log||\widehat{E}^2_h - \widehat{E}^2_{\mathrm{ex}}||_{L^2}$',
                                title=r'Error $\widehat{E}^2_h$', 
                                save_path=f"{directory_results}error_L2_electric_primal")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_L2_magnetic_primal, 
                                label="NED", 
                                ylabel=r'$\log||\widehat{H}^1_h - \widehat{H}^1_{\mathrm{ex}}||_{L^2}$',
                                title=r'Error $\widehat{H}^1_h$', 
                                save_path=f"{directory_results}error_L2_magnetic_primal")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_Hdiv_electric_primal, 
                                label="RT", 
                                ylabel=r'$\log||\widehat{E}^2_h - \widehat{E}^2_{\mathrm{ex}}||_{H^{\mathrm{div}}}$',
                                title=r'Error $\widehat{E}^2_h$', 
                                save_path=f"{directory_results}error_Hdiv_electric_primal")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_Hcurl_magnetic_primal, 
                                label="NED", 
                                ylabel=r'$\log||\widehat{H}^1_h - \widehat{H}^1_{\mathrm{ex}}||_{H^{\mathrm{curl}}}$',
                                title=r'Error $\widehat{H}^1_h$', 
                                save_path=f"{directory_results}error_Hcurl_magnetic_primal")

if len(error_normal_primal) > 0 and len(error_tangential_primal)>0:
    basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_normal_primal, 
                                    label="NED",
                                    ylabel=r'$\log|||\widehat{E}^{1, \bm{n}}_h - P_h \widehat{E}^{1, \bm{n}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$',
                                    title=r'Error $\widehat{E}^{1, \bm{n}}_h$', 
                                    save_path=f"{directory_results}error_normal_primal")

    basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_tangential_primal, 
                                    label="NED", 
                                    ylabel=r'$\log|||\widehat{H}^{1, \bm{t}}_h - \widehat{H}^{1, \bm{t}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$',
                                    title=r'Error $\widehat{H}^{1, \bm{t}}_h$', 
                                    save_path=f"{directory_results}error_tangential_primal")
    

# # Plot dual
basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_L2_electric_dual, 
                                label="NED", 
                                ylabel=r'$\log||{E}^1_h - {E}^1_{\mathrm{ex}}||_{L^2}$',
                                title=r'Error ${E}^1_h$', 
                                save_path=f"{directory_results}error_L2_electric_dual")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_L2_magnetic_dual, 
                                label="RT", 
                                ylabel=r'$\log||{H}^2_h -   {H}^2_{\mathrm{ex}}||_{L^2}$',
                                title=r'Error ${H}^2_h$', 
                                save_path=f"{directory_results}error_L2_magnetic_dual")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_Hcurl_electric_dual, 
                                label="NED", 
                                ylabel=r'$\log||{E}^1_h - {E}^1_{\mathrm{ex}}||_{H^{\mathrm{curl}}}$',
                                title=r'Error ${E}^1_h$', 
                                save_path=f"{directory_results}error_Hcurl_electric_dual")

basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_Hdiv_magnetic_dual, 
                                label="RT", 
                                ylabel=r'$\log||{H}^2_h - {H}^2_{\mathrm{ex}}||_{H^{\mathrm{div}}}$',
                                title=r'Error ${H}^2_h$', 
                                save_path=f"{directory_results}error_Hdiv_magnetic_dual")

if len(error_normal_dual) > 0 and  len(error_tangential_dual) > 0:
    basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_normal_dual, 
                                label="NED",
                                ylabel=r'$\log|||{H}^{1, \bm{n}}_h - P_h {H}^{1, \bm{n}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$',
                                title=r'Error ${H}^{1, \bm{n}}_h$', 
                                save_path=f"{directory_results}error_normal_dual")
    
    basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_tangential_dual, 
                                    label="NED", 
                                    ylabel=r'$\log|||{E}^{1, \bm{t}}_h - {E}^{1, \bm{t}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$',
                                    title=r'Error ${E}^{1, \bm{t}}_h$', 
                                    save_path=f"{directory_results}error_tangential_dual")
    
# # Plot primal/dual
basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_electric_df, 
                                label="s",
                                ylabel=r'$\log|| {E}^1_h - \widehat{E}^2_{h} ||_{L^2}$',
                                title=r'Error ${E}^1_h - \widehat{E}^2_{h}$', 
                                save_path=f"{directory_results}error_electric_df")
                                
basic_plotting.plot_convergence(deg_vec, h_list=h_list, variable_list=error_magnetic_df, 
                                label="s", 
                                ylabel=r'$\log|| \widehat{H}^1_h - {H}^2_{h} ||_{L^2}$',
                                title=r'Error $\widehat{H}^1_h - {H}^2_{h}$', 
                                save_path=f"{directory_results}error_magnetic_df")
    

plt.show()
