import numpy as np
from compute_err import compute_err

pol_degree = int(input("Enter polynomial degree: "))
quad = False

n_elements_vec = [2,4,8,16]
h_vec = 1./np.array(n_elements_vec)

time_step = 1/1000
t_end = 1000*time_step

L2_error_pressure = {"Linf":[], "L2": [], "Tend":[]}
L2_error_velocity = {"Linf":[], "L2": [], "Tend":[]}
Hdiv_error_velocity = {"Linf":[], "L2": [], "Tend":[]}

L2_rate_pressure = {"Linf":[], "L2": [], "Tend":[]}
L2_rate_velocity = {"Linf":[], "L2": [], "Tend":[]}
Hdiv_rate_velocity = {"Linf":[], "L2": [], "Tend":[]}

for ii, n_elem in enumerate(n_elements_vec):

    dict_error_time = compute_err(n_elem, pol_degree, time_step, t_end, quad=quad)

    dict_error_Tend = dict_error_time["Tend"]
    dict_error_Linf = dict_error_time["Linf"]
    dict_error_L2 = dict_error_time["L2"]

    L2_error_pressure["Tend"].append(dict_error_Tend["L2_error_pressure"])
    L2_error_velocity["Tend"].append(dict_error_Tend["L2_error_velocity"])
    Hdiv_error_velocity["Tend"].append(dict_error_Tend["Hdiv_error_velocity"])

    L2_error_pressure["Linf"].append(dict_error_Linf["L2_error_pressure"])
    L2_error_velocity["Linf"].append(dict_error_Linf["L2_error_velocity"])
    Hdiv_error_velocity["Linf"].append(dict_error_Linf["Hdiv_error_velocity"])

    L2_error_pressure["L2"].append(dict_error_L2["L2_error_pressure"])
    L2_error_velocity["L2"].append(dict_error_L2["L2_error_velocity"])
    Hdiv_error_velocity["L2"].append(dict_error_L2["Hdiv_error_velocity"])

    if ii >=1:
        L2_rate_pressure["Tend"].append(np.log(L2_error_pressure["Tend"][ii]/L2_error_pressure["Tend"][ii-1])/np.log(h_vec[ii]/h_vec[ii-1]))
        L2_rate_velocity["Tend"].append(np.log(L2_error_velocity["Tend"][ii]/L2_error_velocity["Tend"][ii-1])/np.log(h_vec[ii]/h_vec[ii-1]))
        Hdiv_rate_velocity["Tend"].append(np.log(Hdiv_error_velocity["Tend"][ii]/Hdiv_error_velocity["Tend"][ii-1])/np.log(h_vec[ii]/h_vec[ii-1]))

        L2_rate_pressure["Linf"].append(np.log(L2_error_pressure["Linf"][ii]/L2_error_pressure["Linf"][ii-1])/np.log(h_vec[ii]/h_vec[ii-1]))
        L2_rate_velocity["Linf"].append(np.log(L2_error_velocity["Linf"][ii]/L2_error_velocity["Linf"][ii-1])/np.log(h_vec[ii]/h_vec[ii-1]))
        Hdiv_rate_velocity["Linf"].append(np.log(Hdiv_error_velocity["Linf"][ii]/Hdiv_error_velocity["Linf"][ii-1])/np.log(h_vec[ii]/h_vec[ii-1]))

        L2_rate_pressure["L2"].append(np.log(L2_error_pressure["L2"][ii]/L2_error_pressure["L2"][ii-1])/np.log(h_vec[ii]/h_vec[ii-1]))
        L2_rate_velocity["L2"].append(np.log(L2_error_velocity["L2"][ii]/L2_error_velocity["L2"][ii-1])/np.log(h_vec[ii]/h_vec[ii-1]))
        Hdiv_rate_velocity["L2"].append(np.log(Hdiv_error_velocity["L2"][ii]/Hdiv_error_velocity["L2"][ii-1])/np.log(h_vec[ii]/h_vec[ii-1]))


print("Convergence rate L2 primal pressure at Tend, Linf in time, L2 in time: ")
print(L2_rate_pressure["Tend"])
print(L2_rate_pressure["Linf"])
print(L2_rate_pressure["L2"])

print("Convergence rate L2 primal velocity at Tend, Linf in time, L2 in time: ") 
print(L2_rate_velocity["Tend"])
print(L2_rate_velocity["Linf"])
print(L2_rate_velocity["L2"])

print("Convergence rate Hdiv primal velocity at Tend, Linf in time, L2 in time: ")
print(Hdiv_rate_velocity["Tend"])
print(Hdiv_rate_velocity["Linf"])
print(Hdiv_rate_velocity["L2"])
