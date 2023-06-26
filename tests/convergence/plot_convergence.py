import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.postprocessing import basic_plotting

bc_case = "mixed" #input("Enter the boundary conditions (electric, magnetic, mixed):")
directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/{bc_case}/"

deg_vec  = [1]

h_list = []

# Errors

# Primal
error_L2_electric_primal = []
error_L2_magnetic_primal = []
error_normal_primal = []
error_tangential_primal = []

# Dual
error_L2_electric_dual = []
error_L2_magnetic_dual = []
error_normal_dual = []
error_tangential_dual = []

# Dual Field
error_electric_df = []
error_magnetic_df = []

# Orders

# Primal
order_L2_electric_primal = []
order_L2_magnetic_primal = []
order_normal_primal = []
order_tangential_primal = []

# Dual
order_L2_electric_dual = []
order_L2_magnetic_dual = []
order_normal_dual = []
order_tangential_dual = []

# Dual field
order_electric_df = []
order_magnetic_df = []

for count, deg in enumerate(deg_vec):
    file = f"convergence_maxwell_r={deg}.csv"

    df = pd.read_csv(directory_results+file)
    n_elements = df["N"].values

    h_elements = 1./n_elements

    errors_rate = df.columns[df.columns.str.contains('error')]
    columns_rate = df.columns[df.columns.str.contains('rate')]

    h_list.append(h_elements)

    error_L2_electric_primal.append(df["error_L2_electric_primal"].values)

    # # Print the columns
    # for column in columns_rate:
    #     print(column)
    #     for value in df[column]:
    #         print(value)

basic_plotting.plot_convergence(h_list=h_list, variable_list=error_L2_electric_primal, 
                                label="RT", 
                                ylabel=r'$\log||\widehat{E}^2_h - \widehat{E}^2_{\mathrm{ex}}||_{L^2}$',
                                title=r'Error $\widehat{E}^2_h$', 
                                save_path=f"{directory_results}error_L2_electric_primal")
                                

plt.show()

# if save_plots:
#     plt.savefig(path_fig + "E_2" + geo_case + bc_case + ".pdf", format="pdf")

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_H1 = err_H1_dict[ii][:, 0]
#     plt.plot(np.log(h), np.log(errL2_H1), '-.+', label=r'NED$^1_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errL2_H1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||\widehat{H}^1_h - \widehat{H}^1_{\mathrm{ex}}||_{L^2}$')
# plt.title(r'Error $\widehat{H}^1_h$')

# plt.legend()

# if save_plots:
#     plt.savefig(path_fig + "H_1" + geo_case + bc_case + ".pdf", format="pdf")

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_E1nor = err_E1nor_dict[ii][:]
#     plt.plot(np.log(h), np.log(errL2_E1nor), '-.+', label=r'NED$^1_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**(ii)) + \
#              + 1.3*(np.log(errL2_E1nor)[0] - np.log(h**(ii))[0]), '-v', label=r'$h^' + str(ii) + '$')

# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log|||\widehat{E}^{1, \bm{n}}_h - P_h \widehat{E}^{1, \bm{n}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
# plt.title(r'Error $\widehat{E}^{1, \bm{n}}_h$')

# plt.legend()

# if save_plots:
#     plt.savefig(path_fig + "E_1nor" + geo_case + bc_case + ".pdf", format="pdf")


# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_H1tan = err_H1tan_dict[ii][:]
#     plt.plot(np.log(h), np.log(errL2_H1tan), '-.+', label=r'NED$^1_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.3*(np.log(errL2_H1tan)[-1] - 0.95*np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log|||\widehat{H}^{1, \bm{t}}_h - \widehat{H}^{1, \bm{t}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
# plt.title(r'Error $\widehat{H}^{1, \bm{t}}_h$')

# plt.legend()

# if save_plots:
#     plt.savefig(path_fig + "H_1tan" + geo_case + bc_case + ".pdf", format="pdf")


# # E1H2 system

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_E1 = err_E1_dict[ii][:, 0]
#     plt.plot(np.log(h), np.log(errL2_E1), '-.+', label=r'NED$^1_' + str(ii)+ '$')
#     plt.plot(np.log(h), np.log(h**(ii)) + \
#              + 1.1*(np.log(errL2_E1)[0] - np.log(h**(ii))[0]), '-v', label=r'$h^' + str(ii) + '$')

# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||E^1_h - E^1_{\mathrm{ex}}||_{L^2}$')
# plt.title(r'Error $E^1_h$')

# plt.legend()

# if save_plots:
#     plt.savefig(path_fig + "E_1" + geo_case + bc_case + ".pdf", format="pdf")


# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_H2 = err_H2_dict[ii][:, 0]
#     plt.plot(np.log(h), np.log(errL2_H2), '-.+', label=r'RT$^1_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.3*(np.log(errL2_H2)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||H^2_h - H^2_{\mathrm{ex}}||_{L^2}$')
# plt.title(r'Error $H^2_h$')

# plt.legend()

# if save_plots:
#     plt.savefig(path_fig + "H_2" + geo_case + bc_case + ".pdf", format="pdf")

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_H1nor = err_H1nor_dict[ii][:]
#     plt.plot(np.log(h), np.log(errL2_H1nor), '-.+', label=r'NED$^1_' + str(ii) + '$')

#     plt.plot(np.log(h), np.log(h**ii) + \
#         + 1.5*(np.log(errL2_H1nor)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log|||H^{1, \bm{n}}_h - P_h H^{1,\bm{n}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
# plt.title(r'Error $H^{1, \bm{n}}_h$')

# plt.legend()

# if save_plots:
#     plt.savefig(path_fig + "H_1nor" + geo_case + bc_case + ".pdf", format="pdf")


# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_E1tan = err_E1tan_dict[ii][:]
#     plt.plot(np.log(h), np.log(errL2_E1tan), '-.+', label=r'NED$^1_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.3*(np.log(errL2_E1tan)[0] - np.log(h**ii)[0]), '-v', label=r'$h^' + str(ii) + '$')

# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log|||E^{1, \bm{t}}_h - E^{1, \bm{t}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
# plt.title(r'Error $E^{1, \bm{t}}_h$')

# plt.legend()

# if save_plots:
#     plt.savefig(path_fig + "E_1tan" + geo_case + bc_case + ".pdf", format="pdf")


# # Dual Field
# #

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_E12 = err_E12_dict[ii]
#     plt.plot(np.log(h), np.log(errL2_E12), '-.+', label=r'$s=' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errL2_E12)[0] - np.log(h**ii)[0]), '-v', label=r'$h^' + str(ii) + '$')

# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||\widehat{E}^2_h - E^1_h||_{L^2}$')
# plt.title(r'Error between $\widehat{E}^2$ and $E^1$')

# plt.legend()

# if save_plots:
#     plt.savefig(path_fig + "E_12" + geo_case + bc_case + ".pdf", format="pdf")

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_H12 = err_H12_dict[ii]
#     plt.plot(np.log(h), np.log(errL2_H12), '-.+', label=r'$s=' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.25*(np.log(errL2_H12)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||H^2_h - \widehat{H}^1_h||_{L^2}$')
# plt.title(r'Error between $H^2_h$ and $\widehat{H}^1_h$')

# plt.legend()

# if save_plots:
#     plt.savefig(path_fig + "H_12" + geo_case + bc_case + ".pdf", format="pdf")


# plt.show()