import pandas as pd
import numpy as np
import math
from Bio.PDB import *

AA_dict = { "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
            }

def residue_filter_h20(residue):
    residue = list(filter(lambda i:i.get_resname() in AA_dict ,residue))
    return residue

def chain_filter(chain):
    residue = residue_filter_h20(chain.get_list())
    return residue

def get_side_chain(aa_atom_list):
    temp = aa_atom_list.copy()
    for i in temp:
        if(i.name in ["N","C","O"]):
            aa_atom_list.remove(i)
    return(aa_atom_list)

def remove_hydrogen_atom(aa_atom_list):
    temp = aa_atom_list.copy()
    for i in temp:
        if(i.name.startswith('H')):
            aa_atom_list.remove(i)
    return(aa_atom_list)

def atom_dist(atom1, atom2):
    return atom1-atom2

def atom_dist_square_sigmoid(atom1, atom2):
    dist_sq = (atom1-atom2)**2
    value = 1/(math.e**(-dist_sq))
    return value

def atom_dist_deal(atom1, atom2, dist_threshold=5):
    aa_dist = atom1-atom2
    value = 1
    if aa_dist > dist_threshold :
        value = 0
    elif aa_dist < 1 :
        value = 1
    else:
        value = 1/aa_dist
    return value

def atoms_dist(aa1, aa2, side_chain=None, remove_hydro=True):
    aa1_atom_list = aa1.get_list()
    aa2_atom_list = aa2.get_list()
    if(bool(side_chain)):
        aa1_atom_list = get_side_chain(aa1_atom_list)
        aa2_atom_list = get_side_chain(aa2_atom_list)
    if(bool(remove_hydro)):
        aa1_atom_list = remove_hydrogen_atom(aa1_atom_list)
        aa2_atom_list = remove_hydrogen_atom(aa2_atom_list)
    aa1_atom_len = len(aa1_atom_list)
    aa2_atom_len = len(aa2_atom_list)
    matrix = np.zeros([aa1_atom_len, aa2_atom_len])
    aa1_atom_names = []
    aa2_atom_names = []
    for i in range(0, aa1_atom_len):
        aa1_atom_names.append(aa1_atom_list[i].get_name())
        for j in range(0, aa2_atom_len):
            if(i==0):
                aa2_atom_names.append(aa2_atom_list[j].get_name())
            matrix[i,j] = atom_dist(aa1_atom_list[i], aa2_atom_list[j])
    return matrix, aa1_atom_names, aa2_atom_names

def dist(pep1, pep2):
    pep1_len = len(pep1)
    pep2_len = len(pep2)
    pep1_atoms_name = []
    pep2_atoms_name = []
    for i in range(0, pep1_len):
        pos1 = i + 1
        aa_name1 = pep1[i].get_resname()
        for j in range(0, pep2_len):
            pos2 = j + 1
            aa_name2 = pep2[j].get_resname()
            if (j==0):
                aa_atoms_dist, _aa1_atom_names, _aa2_atom_names = atoms_dist(pep1[i], pep2[j], side_chain=False)
                aa1_atom_names = [str(pos1) + '_' + aa_name1 + '_' + n for n in _aa1_atom_names]
                aa2_atom_names = [str(pos2) + '_' + aa_name2 + '_' + n for n in _aa2_atom_names]
                pep1_atoms_name.extend(aa1_atom_names)
                pep2_atoms_name = aa2_atom_names
            else:
                tmp_atoms_dist, _aa1_atom_names, _aa2_atom_names = \
                        atoms_dist(pep1[i], pep2[j], side_chain=False, remove_hydro=True)
                aa_atoms_dist = np.concatenate([aa_atoms_dist, tmp_atoms_dist], axis=1)
                aa2_atom_names = [str(pos2) + '_' + aa_name2 + '_' + n for n in _aa2_atom_names]
                pep2_atoms_name.extend(aa2_atom_names)
        if (i==0):
            pep_atoms_dist = aa_atoms_dist
        else:
            pep_atoms_dist = np.concatenate([pep_atoms_dist, aa_atoms_dist], axis=0)
    return pep_atoms_dist, pep1_atoms_name, pep2_atoms_name


def getChain(pdb):
    parser = PDBParser(QUIET = True)
    data = parser.get_structure(str(pdb), pdb)
    model = next(data.get_models())
    chains = list(model.get_chains())
    chain = chain_filter(chains[0])
    return chain

def get_atom_and_dist(pdb1, pdb2, out1, out2, out3):
    pep1 = getChain(pdb1)
    pep2 = getChain(pdb2)
    matrix, pep1_atoms_name, pep2_atoms_name = dist(pep1, pep2)
    atom1 = {'atom':pep1_atoms_name}
    atom2 = {'atom':pep2_atoms_name}
    np.save(out1, atom1)
    np.save(out2, atom2)
    np.save(out3, matrix)

