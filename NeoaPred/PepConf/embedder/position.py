import torch.nn as nn
import torch
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #print(pe)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: batch_size, seq_len
        '''
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = self.pe[:seq_len, :]
        x = x.repeat(batch_size,1,1)
        #x = self.dropout(x)
        return x

