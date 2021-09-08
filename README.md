# BBOMol
Surrogate-based black-box optimization of molecular properties

## Installation

BBOMol depends on [EvoMol](https://doi.org/10.1186/s13321-020-00458-z) for evolutionary optimization of the surrogate 
function. Follow first the installation steps described on <a href='https://github.com/jules-leguy/evomol'>EvoMol 
repository</a>. Make sure to follow **Installation** and **DFT and Molecular Mechanics optimization** sections.

```shell script
$ git clone https://github.com/jules-leguy/BBOMol.git     # Cloning repository
$ cd BBOMol                                               # Moving into BBOMol directory
$ conda activate evomolenv                                # Activating anaconda environment
$ conda install scikit-learn=0.22.1                       # Installing additional scikit-learn dependency
$ pip install dscribe                                     # Installing additional DScribe dependency
$ conda install -c conda-forge notebook                   # Installing jupyter-notebook to access the reproduction notebooks
$ python -m pip install .                                 # Installing BBOMol
```

To use BBOMol, make sure to activate the *evomolenv* conda environment.

## Quickstart

Running a black-box optimization of the HOMO energy using an RBF-based kernel and the 
[MBTR](https://arxiv.org/abs/1704.06439) descriptor. The merit function that is optimized by the evolutionary algorithm
is the expected improvement of the surrogate function.

```python
from bbomol import run_optimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

run_optimization({
    "obj_function": "homo",
    "merit_optim_parameters": {
        "merit_type": "EI",
    },
    "surrogate_parameters": {
        "GPR_instance": GaussianProcessRegressor(1.0 * RBF(1.0) + WhiteKernel(1.0), normalize_y=True),
        "descriptor": {
            "type": "MBTR"
        }
    }
})



```

## Settings

A dictionary can be given to bbomol.run_optimization to describe the experiment to be performed. This dictionary can 
contain up to 6 entries, that are described in this section.

**Default values** are represented in bold.

### Objective function

The ```"obj_function"``` attribute describes the (costly) objective function to be optimized by the algorithm. It can
be defined according to the formalism of EvoMol. Any value accepted for this
attribute by EvoMol is also accepted here. This includes implemented functions (*e.g.* ```"homo"```), custom Python
functions evaluating a SMILES and functions combining several properties. See the relevant section in 
[EvoMol documentation](https://github.com/jules-leguy/EvoMol#objective-function).

### Surrogate parameters

The ```"surrogate_parameters"``` attribute describes the parameters of the Gaussian process regression model (Kriging) 
that is used as a surrogate of the objective function. This includes the setting of the molecular descriptor. It can
be set with a dictionary containing the following entries.
* ```"GPR_instance"``` : instance of sklearn.gaussian_process.GaussianProcessRegressor (**default :** 
```GaussianProcessRegressor(1.0*RBF(1.0)+WhiteKernel(1.0), normalize_y=True)```)
* ```"descriptor"```: a dictionary that defines the descriptor to be used to represent the solutions. The ```"type"``` 
attribute is used to select the descriptor, which can be configured using the following set of attributes.
  * ```"type"``` : name of the descriptor to be used.
    * **"MBTR"** : [many-body tensor representation](https://arxiv.org/abs/1704.06439), using 
  [DScribe](https://singroup.github.io/dscribe/latest/index.html) implementation.
    * "shingles" : boolean or integer vector of [shingles](https://doi.org/10.1186/s13321-018-0321-8).
    * "SOAP" : [smooth overlap of atomic positions](https://doi.org/10.1103/PhysRevB.87.184115), using 
[DScribe](https://singroup.github.io/dscribe/latest/index.html) implementation.

* *Parameters common to MBTR and SOAP*
  * ```"species"```: list of atomic symbols that can be represented (**["H", "C", "O", "N", "F"]**).
  * ```"MM_program"```: program used to perform MMFF94 molecular mechanics optimization. It can be either 
[RDKit](https://doi.org/10.1186/s13321-014-0037-3) (**"rdkit"**) or [OpenBabel](https://doi.org/10.1186/1758-2946-3-33)
("obabel").
* *Parameters specific to MBTR (see 
[DScribe documentation](https://singroup.github.io/dscribe/latest/tutorials/descriptors/mbtr.html))*
  * ```"atomic_numbers_n"```, ```"inverse_distances_n"```, ```"cosine_angles_n"```: number of bins to 
respectively encode the atomic numbers (**10**), the interatomic distances (**25**) and interatomic angles (**25**).
* *Parameters specific to the vector of shingles*
  * ```"lvl"``` : radius of the shingles (**1**).
  * ```"vect_size"```: size of the descriptor (**2000**).
  * ```"count"``` : if False, the descriptor is a boolean vector that represents whether the i<sup>th</sup> shingle is 
present in the molecule. If True, the descriptor is an integer vector that counts the number of occurrences of the 
i<sup>th</sup> shingle in the molecule (**True**).
* *Parameters specific to SOAP (see 
[DScribe documentation](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html))*
  * ```"rcut"``` : cutoff for local environments (**6.0** Å)
  * ```"nmax"```, ```"lmax"``` : resp. the number of radial basis functions (**8**) and the maximum degree of spherical
harmonics (**6**).
  * ```"average"``` : whether to average all local environments (**"inner"**, "outer") or to consider the environments
independently ("off").

### Merit optimization parameters

The ```"merit_optim_parameters"``` attribute is used to describe the merit function and the parameters of its 
evolutionary optimization. It can be set with a dictionary containing the following entries.

* ```"merit_type"``` : merit function. It can be either the expected improvement of the surrogate function (**"EI"**), or 
the surrogate function directly ("surrogate").
* ```"merit_EI_xi"``` : value of the [ξ parameter](https://www.csd.uwo.ca/~dlizotte/publications/lizotte_phd_thesis.pdf)
of the expected improvement (**0.01**). This parameter is only interpreted if ```"merit_type"``` is set to "EI".
* ```"evomol_parameters"``` : dictionary describing the parameters for the evolutionary optimization of the merit 
function, using the [EvoMol](https://doi.org/10.1186/s13321-020-00458-z) algorithm. See the relevant section in 
[EvoMol documentation](https://github.com/jules-leguy/EvoMol#search-space). The ```"action_space_parameters"```
and ```"optimization_parameters"``` attributes can be set here. They respectively define the number of optimization 
steps at each merit optimization phase, and the chemical space of the solutions that will be generated. The other
attributes are set automatically by BBOMol. **Default value** :
  ```
  {
      "optimization_parameters": {
          "max_steps": 5,
      },
      "action_space_parameters": {
          "max_heavy_atoms": 9,
          "atoms": "C,N,O,F"
      }
  }
    ```
* ```"init_pop_size"``` : number of solutions that are drawn from the dataset of known solutions to be inserted in the
initial population of the evolutionary algorithm, at each optimization phase and for each evolutionary optimization 
instance (**10**).
* ```"init_pop_strategy"``` : strategy to select the solutions from the dataset of known solutions to be inserted in the
initial population of the evolutionary optimization instances. Available strategies :
  * "methane" : always starting the evolutionary optimization from the methane molecule.
  * "best" : selecting the ```"init_pop_size"``` best solutions according to the objective function.
  * "random" : selecting randomly ```"init_pop_size"``` solutions with uniform probability.
  * **"random_weighted"** selecting randomly ```"init_pop_size"``` solutions with a probability that is proportional to 
their objective function value.
* ```"n_merit_optim_restarts"``` : number of merit function evolutionary optimization instances at each merit 
optimization phase (**10**).
* ```"n_best_retrieved"``` : number of (best) solutions that are retrieved from each evolutionary optimization restart
to be evaluated using the objective function and inserted in the dataset of known solutions (**1**).

### Black-box optimization parameters

The ```"bbo_optim_parameters"``` attribute is used to define the parameters of the black-box
optimization. It can be set with a dictionary containing the following entries.
* ```"max_obj_calls"``` : number of calls to the objective function before stopping the algorithm (**1000**).
* ```"score_assigned_to_failed_solutions"``` : the default behaviour is to ignore the solutions that fail either
the descriptors computation or the evaluation by the objective function (**None**). An alternative is to set the given
score as objective function value for these solutions.

### Input/Output parameters

The ```"io_parameters"``` attribute is used to define the input/output parameters. It can be set with a dictionary
containing the following entries.
* ```"smiles_list_init"```: list of SMILES in the initial dataset of solutions (**["C"]**).
* ```"results_path"``` : path of the directory that will be created by BBOMol and in which all results will be stored
for the experiment (**"BBOMol_optim/**).
* ```"save_surrogate_model"``` : whether to save the parameters of the latest surrogate model (**False**).
* ```"save_n_steps"```: period (number of steps) to write the progression of the optimization in ```"results_path"``` 
(**1**).
* ```"dft_working_dir"``` : path of the directory in which DFT calculations (if any) are performed and stored
(**"/tmp"**).
* ```"dft_cache_files"``` : list of paths of JSON files that store the results of previous DFT calculations (**[]**). 
This cache will be used to avoid performing DFT calculations for solutions whose OPT results are already known. Keys
must be SMILES, that are associated with a dictionary that maps the property ("homo", "lumo", ...) with its value in eV.

### Parallelization

The ```"parallelization"``` attribute is used to define the parallelization parameters. It can be set with a dictionary
containing the following entries.
* ```"n_jobs_merit_optim_restarts"``` : number of jobs to perform the restarts of the evolutionary optimization in 
parallel (**1**).
* ```"n_jobs_desc_comput"``` : number of jobs to compute the descriptors in parallel (**1**). This parameter is ignored
when computing the vector of shingles as it cannot be parallelized.
* ```"n_jobs_obj_comput"```: number of jobs to evaluate the selected solutions using the objective function in parallel
(**1**). In case of DFT evaluation, this is different to the parameter that sets the number of threads to perform DFT 
optimizations. The latter is set to 1 by default and cannot be accessed for now, except if using an
evomol.evaluation_dft.OPTEvaluationStrategy instance as objective function.