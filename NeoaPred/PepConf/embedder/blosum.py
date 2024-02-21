import numpy as np
import torch.nn as nn
import torch
import os
#"ARNDCQEGHILKMFPSTWYVX-"
npy_path = os.path.dirname(os.path.abspath(__file__)) + "/BLOSUM50.dict.npy"
assert os.path.exists(npy_path), "\"BLOSUM50.dict.npy\" does not exists."
blosum50 = np.load(npy_path, allow_pickle = True).item()
blosum50 = np.array([blosum50[i] for i in range(0,len(blosum50))])
#Min-max normalization, range = [0,1]
blosum50 = (blosum50 - blosum50.min())/(blosum50.max() - blosum50.min())
blosum50 = torch.FloatTensor(blosum50)

class BlosumEmbedding(nn.Module):

    def __init__(self, blo_freeze):
        super().__init__()
        self.blosum = nn.Embedding.from_pretrained(blosum50, freeze=blo_freeze)
    
    def forward(self, x):
        return(self.blosum(x))
