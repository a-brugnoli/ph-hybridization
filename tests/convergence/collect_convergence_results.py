import numpy as np
import pandas as pd
from mpi4py import MPI
import os
from err_maxwell import compute_error

def make_convergence_csv(pol_degree, comm):
    rank = comm.Get_rank()

    if pol_degree==1:
        n_elem_vector = [1, 2, 4] 
    elif pol_degree==2:
        n_elem_vector = [1, 2, 4]
    elif pol_degree==3:
        n_elem_vector = [1, 2, 4]

    if rank==0:
        list_dict_result = []

    for n_elem in n_elem_vector:
        dict_result = compute_error(n_elem, pol_degree)

        if rank==0:
            list_dict_result.append(dict_result)
    
    if rank==0:
        directory_results_problem = os.path.dirname(os.path.abspath(__file__)) + '/'

        # get list of error dictionaries and store in pandas DataFrame
        df = pd.DataFrame(list_dict_result, index=n_elem_vector)
        df.index.name = 'N'

        # compute convergence rates
        df[df.columns.str.replace('error', 'rate')] = -df.apply(np.log2).diff()
        
        delta_logN = np.diff(np.array(df.index.map(np.log2)))

        columns_to_divide = df.columns[df.columns.str.contains('rate')]
        rows_to_divide = df.index > n_elem_vector[0]
        # Select the slice of columns to divide
        df.loc[rows_to_divide, columns_to_divide] = df.loc[rows_to_divide, columns_to_divide].div(delta_logN, axis=0)

        fileresults = f"convergence_maxwell_r={pol_degree}.csv"
        df.to_csv(directory_results_problem + fileresults, na_rep='---')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

pol_degree_test = None
if rank == 0:
    pol_degree_test = int(input("Enter the polynomial degree for manufactured Navier Stokes: "))

pol_degree_test = comm.bcast(pol_degree_test, root=0)

make_convergence_csv(pol_degree_test, comm)