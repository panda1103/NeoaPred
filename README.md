## NeoaPred - Neoantigen Prediction
This package contains deep learning models and related scripts to run NeoaPred.


## Installation

1.Clone NeoaPred to a local directory

```
git clone https://github.com/DeepImmune/NeoaPred.git
cd NeoaPred
```

2.Create conda environment and prepare the required software
* python=3.6
```
conda create -n my_environment_name python=3.6
conda activate my_environment_name
```

NeoaPred relies on external libraries/programs to handle PDB files and surface files,
to compute chemical/geometric features and coordinates, and to perform neural network calculations.
The following is the list of required libraries/programs.

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

3.After preinstalling dependencies, add the following environment variables to your path, changing the appropriate directories:

```
export APBS_BIN=/path_to_apbs/APBS-3.0.0.Linux/bin/apbs
export MULTIVALUE_BIN=/path_to_apbs/APBS-3.0.0.Linux/share/apbs/tools/bin/multivalue
export PDB2PQR_BIN=/path_to_anaconda3_envs_pymesh2/bin/pdb2pqr_cli
export MSMS_BIN=/path_to_msms/msms
export PDB2XYZRN=/path_to_msms/pdb_to_xyzrn
```

## Usage
### NeoaPred-PepConf
For peptide conformation prediction, you can:
```
python run_NeoaPred.py --input_file test_in.csv --output_dir test_out --mode PepConf
```
Input file: ```test_in.csv```('.csv' format)  
```
#Input files example: test_1.csv
ID,Allele,Pep
id_0,A2402,ELKFVTLVF
id_1,A2402,RYTRRKNRQ
id_2,A1101,SSKYITFTK

#Input files example: test_2.csv
ID,Allele,WT,Mut
ID_0,A2402,ELKFVTLVF,KLKFVTLVF
ID_1,A2402,RYTRRKNRQ,RYTRRKNRI
ID_2,A1101,SSKYITFTK,SSKYVTFTK
```

Output file:
The out results will be generated in ```test_out/Structure```:  
```*.relaxed_pep.pdb``` is the predicted conformation of peptide.  
```*.relaxed.pdb``` is the structure of pHLA complex.

### NeoaPred-PepFore
For peptide foreignness score prediction, you can:
```
python run_NeoaPred.py --input_file test_2.csv --output_dir test2_out --mode PepFore
```
Input file: ```test_2.csv``` (must contain two columns: 'WT' and 'Mut')  
Output file: 
```test2_out/Surface/FeatFilter/*_outer_surface.ply```(surface features of peptide)  
```test2_out/Foreignness/MhcPep_foreignness.csv```(foreignness score)

