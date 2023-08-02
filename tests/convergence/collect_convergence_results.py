import numpy as np
import pandas as pd
from mpi4py import MPI
import os
from tests.convergence.compute_error import compute_error


def save_csv(dict_result, n_elem_vector, pol_degree, directory_results, norm):
     # get list of error dictionaries and store in pandas DataFrame
        df = pd.DataFrame(dict_result, index=n_elem_vector)
        df.index.name = 'N'

        # compute convergence rates
        df[df.columns.str.replace('error', 'rate')] = -df.apply(np.log2).diff()
        delta_logN = np.diff(np.array(df.index.map(np.log2)))

        columns_to_divide = df.columns[df.columns.str.contains('rate')]
        rows_to_divide = df.index > n_elem_vector[0]
        # Select the slice of columns to divide
        df.loc[rows_to_divide, columns_to_divide] = df.loc[rows_to_divide, columns_to_divide].div(delta_logN, axis=0)

        fileresults = f"convergence_r={pol_degree}_{norm}.csv"
        df.to_csv(directory_results + fileresults, na_rep='---')



def make_convergence_csv(pol_degree, bc_case, type_system, comm, time_step, t_end):
    rank = comm.Get_rank()

    if pol_degree==1:
        n_elem_vector = [1, 2, 4, 8, 16] 
    elif pol_degree==2:
        n_elem_vector = [1, 2, 4, 8]
    elif pol_degree==3:
        n_elem_vector = [1, 2, 4]

    if rank==0:
        list_dict_result_Linf = []
        list_dict_result_L2 = []
        list_dict_result_Tend = []

    for n_elem in n_elem_vector:
        dict_result_time = compute_error(n_elem, pol_degree, bc_type=bc_case, type_system=type_system, time_step=time_step, t_end=t_end)

        dict_result_Linf = dict_result_time["Linf"]
        dict_result_L2 = dict_result_time["L2"]
        dict_result_Tend = dict_result_time["Tend"]


        if rank==0:
            list_dict_result_Linf.append(dict_result_Linf)
            list_dict_result_L2.append(dict_result_L2)
            list_dict_result_Tend.append(dict_result_Tend)

    if rank==0:
        directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/{type_system}/{bc_case}/"
        if not os.path.exists(directory_results):
            os.makedirs(directory_results)

        save_csv(list_dict_result_Linf, n_elem_vector, pol_degree, directory_results, "Linf")
        save_csv(list_dict_result_L2, n_elem_vector, pol_degree, directory_results, "L2")
        save_csv(list_dict_result_Tend, n_elem_vector, pol_degree, directory_results, "Tend")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

pol_degree_test = None
if rank == 0:
    physics= "Maxwell" # input("Enter the physics (Wave or Maxwell): ")
    pol_degree_test = int(input("Enter the polynomial degree: "))
    bc_case= "mixed" # input("Enter the boundary conditions (electric, magnetic, mixed):")

pol_degree_test = comm.bcast(pol_degree_test, root=0)
bc_case = comm.bcast(bc_case, root=0)

time_step = 10**(-5)
t_end = time_step
make_convergence_csv(pol_degree_test, bc_case, physics, comm, time_step, t_end)

