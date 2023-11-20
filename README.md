# Finite element hybridization of port-Hamiltonian systems
Firedrake implementation for the article 
* Brugnoli, Rashad, Zhang, Stramigioli [Finite element hybridization of port-Hamiltonian systems](https://arxiv.org/abs/2302.06239).

The numerical examples can be reproduced as follows:
* install the Zenodo-archived release of firedrake hosted at https://zenodo.org/records/10047121
  ```
  firedrake-install --doi 10.5281/zenodo.10047121
  ```
* actiivate the environment and run ```pip install tqdm pandas gmsh``` to install required librairies
* from the project root run ```source set_project.sh``` to set the ```PYTHONPATH```
* the different simulation are located into ```./run_results```
  * for the conservation properties of the scheme run
    ```
    $ python run_results/conservation/test_conservation.py --help
    usage: test_conservation.py [-h] [--model {Wave,Maxwell}] [--nel NEL [NEL ...]] [--degree DEGREE] [--ndim {2,3}]
                                [--dt DT] [--t_end T_END] [--quad] [--save_out]
    
    Full Parser for simulation options
    
    options:
      -h, --help            show this help message and exit
      --model {Wave,Maxwell}
                            The considered system (Wave or Maxwell)
      --nel NEL [NEL ...]   Array containing the number of elements of each side
      --degree DEGREE       An integer parameter for the polynomial degree
      --ndim {2,3}          Geometrical dimension of the problem
      --dt DT               Time step
      --t_end T_END         Final time of the simulation
      --quad                Boolean for quadrilateral or hexahedral mesh (true if specified, false otherwise)
      --save_out            Boolean to save possible output files (true if specified, false otherwise)
    ```
  * for the equivalence between mixed and hybrid formulation run (same usage as the conservation problem)
    ```
    $ python run_results/equivalence_mixedhybrid/test_equivalence.py --help
    ```
  * for the Fichera corner example run
    ```  
    python run_results/benchmarks/fichera_maxwell.py --help
    usage: fichera_maxwell.py [-h] [--quad] [--save_out]
    
    Basic Parser for simulation options
    
    options:
      -h, --help  show this help message and exit
      --quad      Boolean for quadrilateral or hexahedral mesh (true if specified, false otherwise)
      --save_out  Boolean to save possible output files (true if specified, false otherwise)
    ```
  * for the discontinuous coefficient problem for the wave equation run (same usage as Fichera corner)
    ```
    python run_results/benchmarks/discontinuous_coeff_wave.py --help
    ```
  * to run the convergence test run (this will take several hours on a laptop)
    ```
    python run_results/convergence/collect_convergence_results.py
    ```
  * to reproduce the convergence plot of the Maxwell equations run
    ```
    python run_results/convergence/plot_convergence_maxwell.py
    ```
  * to reproduce the convergence plot of the Wave equations run
    ```
    python run_results/convergence/plot_convergence_wave.py
    ```
    
