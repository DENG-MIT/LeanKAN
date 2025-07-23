**The code in this repository includes an implementation of LeanKAN. Our implementation uses the original, standard addition KAN implementation in the following repository as a starting point: https://github.com/vpuri3/KolmogorovArnold.jl**

## LeanKAN Implementation
Please see src/kdense_rm.jl. In addition to the standard definitions of nodes, grids, normalizers, etc., mult_flag needs to be defined for multiplication layers. mult_flag corresponds to n^mu in the Neural Networks manuscript, and dicates the number of input nodes used as multiplication nodes.

For a simple example, please see Mult_test_LeanKAN.jl. Further discussion of this case is available in Sec. 4.1 of the manuscript. 

For a PDE example leveraging KAN-ODEs ([CMAME paper](https://doi.org/10.1016/j.cma.2024.117397), [Github](https://github.com/DENG-MIT/KAN-ODEs)), please see Schrodinger_LeanKAN.jl. Further discussion of this case is available in Sec. 4.3 of the manuscript.

## MultKAN Implementation
Please see src/kdense_multKAN.jl. Here we implemented the original MultKAN framework of Liu et al. in Julia, with k=2, for comparison against LeanKAN. 

For a simple example, please see Mult_test_MultKAN.jl. Further discussion of this case is available in Sec. 4.1 of our manuscript.

**Training results for all three cases (LeanKAN test, MultKAN test, LeanKAN PDE) are plotted in results/figs**
