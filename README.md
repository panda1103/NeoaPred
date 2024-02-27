## NeoaPred - Neoantigen Prediction
This package contains deep learning models and related scripts to run NeoaPred.

## Software prerequisites
NeoaPred relies on external libraries/programs to handle PDB files and surface files,
to compute chemical/geometric features and coordinates, and to perform neural network calculations.
The following is the list of required libraries/programs, as well as the version on which it was tested (in parenthesis).

* python=3.6
```
conda create -n my_environment_name python=3.6
conda activate my_environment_name
```
* pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
* pymesh2 (Only python 3.6 is supported.)
```
conda install -c "conda-forge/label/cf202003" pymesh2
```
* biopython
```
conda install -c conda-forge biopython
```
* PeptideConstructor
```
pip install PeptideConstructor
```
* ml_collections
```
pip install ml_collections
```
* importlib-resources
```
pip install importlib-resources
#import importlib_resources as resources(3.6)
#from importlib import resources(3.8)
```
* openmm
```
conda install openmm
```
* dm-tree
```
conda install dm-tree
```
* modelcif
```
pip install modelcif
```
* einops
```
pip install einops
```
* pytorch_lightning
```
pip install pytorch_lightning
```
* pdbfixer
```
conda install -c conda-forge pdbfixer
```
NOTE:
Check the installed file "anaconda3/envs/my_environment_name/lib/python3.6/site-packages/pdbfixer/soft.xml", 
change "import simtk.openmm as mm" to "import openmm as mm" in line_4226, unless you install openmm from simtk.
* sklearn
```
pip install sklearn
```
* networkx
```
pip install networkx=2.5.1
```
* [reduce](http://kinemage.biochem.duke.edu/software/reduce.php) (3.23). To add protons to proteins.
* [MSMS](http://mgltools.scripps.edu/packages/MSMS/) (2.6.1). To compute the surface of proteins.
* PDB2PQR (2.1.1), multivalue, and [APBS](http://www.poissonboltzmann.org/) (1.5). These programs are necessary to compute electrostatics charges.
