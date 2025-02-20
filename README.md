**The code in this repository includes an implementation of LeanKAN. Our implementation uses the original, standard addition KAN implementation in the following repository as a starting point: https://github.com/vpuri3/KolmogorovArnold.jl**

## LeanKAN Implementation
Please see the files in src_v2. In addition to the standard definitions of nodes, grids, normalizers, etc., mult_flag needs to be defined for multiplication layers. mult_flag corresponds to n^mu in the arxiv manuscript, and dicates the number of input nodes used as multiplication nodes.

For an example, please see Mult_test_v2.jl. Further discussion of this case is available in Sec. IIC of our manuscript.

## MultKAN Implementation
Please see the files in src_MultKAN. Here we implemented the original MultKAN framework of Liu et al. in Julia, with k=2, for comparison against NewMultKAN. 

For an example, please see Mult_test_MultKAN.jl. Further discussion of this case is available in Sec. IIB of our manuscript.

**Training results are plotted in results/**