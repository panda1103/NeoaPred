
# ------------------------------------------------------------------------
# Modified from BERT-pytorch (https://github.com/codertimo/BERT-pytorch)
# ------------------------------------------------------------------------

import torch.nn as nn
import torch

import sys
from .aminoacid import AminoacidEmbedding
from .position import PositionalEmbedding
from .blosum import BlosumEmbedding
from .phychem import PhyChemEnbedding


class Embedding(nn.Module):

    def __init__(self, aa_dim, pos_dim, blo_freeze, phychem_freeze, dropout):
        super().__init__()
        self.aminoacid = AminoacidEmbedding(aminoacid_size=22, dim=aa_dim) #dim=21
        self.position = PositionalEmbedding(max_len=200, dim=pos_dim) #dim=36
        self.blosum = BlosumEmbedding(blo_freeze) #dim=22
        self.phychem = PhyChemEnbedding(phychem_freeze) #dim=17
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequences):
        x1 = self.aminoacid(sequences)
        x2 = self.position(sequences)
        x3 = self.blosum(sequences)
        x4 = self.phychem(sequences)
        x = torch.cat((x1, x2, x3, x4), 2)
        return self.dropout(x)
