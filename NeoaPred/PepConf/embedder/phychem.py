import numpy as np
import torch.nn as nn
import torch
import os 
#"ARNDCQEGHILKMFPSTWYVX-"
npy_path = os.path.dirname(os.path.abspath(__file__)) + "/PhyChem.dict.npy"
assert os.path.exists(npy_path), "\"PhyChem.dict.npy\" does not exists."
phychem = np.load(npy_path, allow_pickle = True).item()
phychem = np.array([phychem[i] for i in range(0,len(phychem))]).T
#Min-max normalization, range = [0,1]
phychem = np.array([(phychem[i] - phychem[i].min())/(phychem[i].max() - phychem[i].min())  for i in range(0,len(phychem))])
phychem = phychem.T
phychem = torch.FloatTensor(phychem)

class PhyChemEnbedding(nn.Module):

    def __init__(self, phychem_freeze):
        super().__init__()
        self.phychem = nn.Embedding.from_pretrained(phychem, freeze=phychem_freeze)

    def forward(self, x):
        return(self.phychem(x))
