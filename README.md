# ph_hybridization
Firedrake implementation for the article [Finite element hybridization of port-Hamiltonian systems](https://arxiv.org/abs/2302.06239).

Additional librairies needed: \
tqdm \
pandas \

The results reported in the paper can be reproduced by running scripts in the folder tests. 


For the conservation properties of the scheme run \
tests/conservation/run_conservation.py 


For the equivalence between mixed and hybrid formulation run \
tests/equivalence_mixedhybrid/test_equivalence.py 


For the convergence rates run \
tests/convergence/collect_convergence_results.py 

Once the script is finished (this will take several hours) run 

tests/convergence/plot_convergence_wave.py \
for the wave equation

tests/convergence/plot_convergence_maxwell.py \
for the Maxwell equations. 


For the discontinuous coefficient problem for the wave equation run \
tests/benchmarks/discontinuous_coeff_wave.py 


For the Fichera corner problem for the Maxwell equation run \
tests/benchmarks/fichera_maxwell.py
