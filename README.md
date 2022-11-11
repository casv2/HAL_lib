
## Hyperactive Learning (HAL) Python interface for Atomic Cluster Expansion (ACE1.jl) 

This package uses the ACE.jl Julia software package to build linear ACE interatomic potentials (ACE1.jl).

### Installation instructions:

1. install julia 1.7.x and python 3.8 or 3.9 (with python ase, scikit-learn, matplotlib and numpy installed)
2. run julia command 

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

Threading is advised to be enabled by setting environment `JULIA_NUM_THREADS` for Julia threading and `MKL_NUM_THREADS`/`OPENMP_NUM_THREADS`/`BLAS_NUM_THREADS` to enable numpy/sklearn threading
