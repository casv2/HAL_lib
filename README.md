
## Hyperactive Learning (HAL) Python interface for Atomic Cluster Expansion (ACE1.jl) 

This package builds ACE interatomic potentials using Hyperactive Learning (HAL) using the ACE.jl Julia software package.

### Installation instructions:

1. install julia 1.7.x and python 3.8 or 3.9 (with python ase, scikit-learn, matplotlib and numpy installed)
2. run julia command (make sure latest ACE1.jl is installed including committee potential support)

```using Pkg; Pkg.activate("."); pkg"registry add https://github.com/JuliaRegistries/General"; pkg"registry add https://github.com/JuliaMolSim/MolSim.git"; pkg"add ACE1, ASE, JuLIP and LinearAlgebra"```

3. install `julia` python package to set up Python -> Julia connection 

```python -m pip install julia```
```python -c "import julia; julia.install()"```

4. clone this repo and make sure that `HAL_lib` can be found in your `PYTHONPATH`

### User instructions:

Example scripts are provided in the `tutorials` folder. A HAL run requires the following main components:

- A Python ASE calculator to perform QM/DFT calculations
- ACE parameters for linear ACE basis in Julia (correlation order, polynomial degree)
- HAL parameters for sampling strategy (biasing strength, uncertainty tolerance, temperature, pressure)

Threading is advised to be enabled by setting environment `JULIA_NUM_THREADS` for Julia threading and `MKL_NUM_THREADS`/`OPENMP_NUM_THREADS`/`BLAS_NUM_THREADS` to enable numpy/scikit-learn threading

### References:

If using this code please reference

```
@misc{van2022hyperactive,
  doi = {10.48550/ARXIV.2210.04225},
  url = {https://arxiv.org/abs/2210.04225},
  author = {van der Oord, Cas and Sachs, Matthias and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  title = {Hyperactive Learning (HAL) for Data-Driven Interatomic Potentials},
  publisher = {arXiv},
  year = {2022},
}
