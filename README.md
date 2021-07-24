# BBOMol
Surrogate-based black-box optimization of molecular properties

## Installation

BBOMol depends on EvoMol for evolutionary optimization of the surrogate function. Follow first the installation steps 
described on <a href='https://github.com/jules-leguy/evomol'>EvoMol repository</a>. Make sure to follow **Installation** 
and **DFT and Molecular Mechanics optimization** sections.

```shell script
$ git clone https://github.com/jules-leguy/BBOMol.git     # Cloning repository
$ conda activate evomolenv                                # Activating anaconda environment
$ conda install scikit-learn=0.22.1                       # Installing additional scikit-learn dependency
$ python -m pip install .                                 # Installing BBOMol
```

To use BBOMol, make sure to activate the *evomolenv* conda environment.


