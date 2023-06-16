import os
home_dir =os.environ['HOME']

directory_results = f"{home_dir}/StoreResults/Ph_Hybridization/"

# Check if the directory exists
if not os.path.exists(directory_results):
    # If it doesn't exist, create it
    os.makedirs(directory_results)

def directories_problem(str_problem):
    directory_results_problem = f"{directory_results}{str_problem}/"
    if not os.path.exists(directory_results_problem):
        os.makedirs(directory_results_problem)

    directory_results_problem_csv = f"{directory_results_problem}csv/"
    if not os.path.exists(directory_results_problem_csv):
        os.makedirs(directory_results_problem_csv)

    directory_results_problem_paraview = f"{directory_results_problem}Paraview/"
    if not os.path.exists(directory_results_problem_paraview):
        os.makedirs(directory_results_problem_paraview)

    directory_results_problem_matplotlib = f"{directory_results_problem}Matplotlib/"
    if not os.path.exists(directory_results_problem_matplotlib):
        os.makedirs(directory_results_problem_matplotlib)

    return directory_results_problem, directory_results_problem_matplotlib, directory_results_problem_paraview
