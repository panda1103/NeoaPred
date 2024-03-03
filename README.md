
## NeoaPred: a deep-learning framework for deciphering tumor neoantigen based on surface and structural features of peptides  


## Table of Contents:
 - [Description](#Description)
      * [PepConf-Overview](#PepConf-Overview)
      * [PepFore-Overview](#PepFore-Overview)
 - [Installation](#Installation)
 - [Usage](#Usage)
      * [NeoaPred](#NeoaPred)
      * [NeoaPred-PepConf](#NeoaPred-PepConf)
      * [NeoaPred-PepFore](#NeoaPred-PepFore)
 - [PyMOL plugin](#PyMOL-plugin)
 - [HLA-I structure templates](#HLA-I-structure-templates)
 - [License](#License)



## Description
This package contains deep learning models and related scripts to run NeoaPred.  
NeoaPred includes two model: PepConf and PepFore. PepConf utilizes the sequence of peptide and HLA-I, as well as the structure of HLA-I to construct the conformation of peptide binding to HLA-I. PepConf has two peculiarities: 1) The model computes a two-dimensional matrix to describe the spatial distance between the peptide and HLA-I molecule; 2) The model uses a intermolecular loss to achieve the constraints of spatial distance between peptide and HLA-I molecule. PepFore integrates the differences in surface features, spatial structure, and atom groups between the mutant peptide and wild-type counterpart to predict a foreignness score. 

![NeoaPred workflow](https://github.com/DeepImmune/NeoaPred/blob/main/img/workflow.png)

### PepConf-Overview
![PepConf-Overview](https://github.com/DeepImmune/NeoaPred/blob/main/img/PepConf.png)

### PepFore-Overview
![PepFore-Overview](https://github.com/DeepImmune/NeoaPred/blob/main/img/PepFore.png)

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
* [Pymesh2](https://github.com/PyMesh/PyMesh) (0.1.14).  
To handle ply surface files, attributes, and to regularize meshes. Only python 3.6 is supported.
```
conda install -c "conda-forge/label/cf202003" pymesh2
```
* [BioPython](https://github.com/biopython/biopython) (1.78).  
To parse PDB files.
```
conda install -c conda-forge biopython
```
* [PeptideConstructor](https://github.com/CharlesHahn/PeptideConstructor) (0.2.1).  
Create an initial peptide PDB file.
```
pip install PeptideConstructor
```
* [ml_collections](https://github.com/google/ml_collections) (0.1.1).  
ML Collections is a library of Python collections designed for ML usecases.
```
pip install ml_collections
```
* [importlib-resources](https://github.com/python/importlib_resources) (5.4.0).  
Read resources from Python packages.
```
pip install importlib-resources
```
* [openmm](https://openmm.org) (7.6.0).  
Required by pdbfixer.
```
conda install openmm
```
* [pdbfixer](https://github.com/openmm/pdbfixer) (1.8.1).  
Fixing problems in predicted structure of peptides.
```
conda install -c conda-forge pdbfixer
```
NOTE:
Check the installed file "anaconda3/envs/my_environment_name/lib/python3.6/site-packages/pdbfixer/soft.xml", 
change "import simtk.openmm as mm" to "import openmm as mm" in line_4226, unless you install openmm from simtk.
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
### NeoaPred  
```
python run_NeoaPred.py --help
usage: run_NeoaPred.py [-h] --input_file INPUT_FILE [--output_dir OUTPUT_DIR]
                       [--mode MODE] [--trained_model_1 TRAINED_MODEL_1]
                       [--trained_model_2 TRAINED_MODEL_2]

optional arguments:
  -h, --help            show this help message and exit
  --input_file          Input file (*.csv)
  --output_dir          Output directory (default = ./)
  --mode                Prediction mode (default = PepFore)
                        PepConf: Predict the conformation of peptide binding to the HLA-I molecule.

                        PepFore: Predict the conformations of Mut and WT peptides,
                                 compute the features of peptides surface,
                                 and compute a foreignness score between Mut and WT.

  --trained_model_1     Pre-trained model for PepConf.
                        (default = NeoaPred/PepConf/trained_model/model_1.pth)
  --trained_model_2     Pre-trained model for PepFore.
                        (default = NeoaPred/PepFore/trained_model/model_2.pth)
```

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
![Peptide](https://github.com/DeepImmune/NeoaPred/blob/main/img/pep.png)
```*.relaxed.pdb``` is the structure of pHLA complex.
![Peptide-HLA](https://github.com/DeepImmune/NeoaPred/blob/main/img/pHLA.png)

### NeoaPred-PepFore
For peptide foreignness score prediction, you can:
```
python run_NeoaPred.py --input_file test_2.csv --output_dir test2_out --mode PepFore
```
Input file: ```test_2.csv``` (must contain two columns: 'WT' and 'Mut')  
```
#Input files example: test_2.csv
ID,Allele,WT,Mut
ID_0,A2402,ELKFVTLVF,KLKFVTLVF
ID_1,A2402,RYTRRKNRQ,RYTRRKNRI
ID_2,A1101,SSKYITFTK,SSKYVTFTK
```
Output files:   
```test2_out/Surface/Feat/*_si_ddc_dm.ply```(surface features of peptide)  
![Feature](https://github.com/DeepImmune/NeoaPred/blob/main/img/features.png)
```test2_out/Foreignness/MhcPep_foreignness.csv```(foreignness score)

## PyMOL plugin

A PyMOL plugin to visualize protein surfaces.  
This plugin was developed by MaSIF and had some modifications made by NeoaPred.  
Please see the MaSIF's tutorial on how to install and use it:
[MaSIF: https://github.com/LPDI-EPFL/masif]

## HLA-I structure templates
Structure templates of 200 HLA-I alleles are stored in ```NeoaPred/PepConf/data/MHC_template_PDB```.
The numbers of HLA-A, HLA-B, and HLA-C alleles are 66, 105, and 29.
To simplify the PepConf model and focus on the HLA-I binding groove domain, 
we only retained the residues of HLA-I from 1 to 180.

## License

NeoaPred is released under an [Apache v2.0 license](LICENSE).

