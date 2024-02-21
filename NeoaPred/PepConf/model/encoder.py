import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from NeoaPred.PepConf.model.attention import AxialAttention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * nn.functional.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
        nn.init.constant_(self.net[-1].weight, 0.)
        nn.init.constant_(self.net[-1].bias, 0.)

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)

class PairwiseAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.1, global_column_attn = False):
        super().__init__()
        #self.outer_mean = OuterMean(dim)

        self.triangle_attention_outgoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.triangle_attention_ingoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True, global_query_attn = global_column_attn)

    def forward(self, x):
        x = self.triangle_attention_outgoing(x, edges = x) + x
        x = self.triangle_attention_ingoing(x, edges = x) + x
        return x


class EncoderBlock(nn.Module):
    """
        EncoderBlock is made up of self-attention and feedforward network.
    """

    def __init__(self, embed_dim, num_heads, dim_head, dropout=0.1):
        super().__init__()
        self.pair_attn = PairwiseAttentionBlock(embed_dim, num_heads, dim_head, dropout)
        self.ff = FeedForward(dim = embed_dim, dropout = dropout)

    def forward(self, x):
        x = self.pair_attn(x)
        x = x + self.ff(x)
        return x

class Encoder(nn.Module):
    """
        Encoder is a stack of N EncoderBlock.
    """

    def __init__(self, num_layers, embed_dim, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([EncoderBlock(embed_dim, **kwargs) for _ in range(num_layers)])

    def forward(self, x):
        chunks = 8
        x = checkpoint_sequential(self.layers, chunks, x)
        x = self.norm(x)
        return x
