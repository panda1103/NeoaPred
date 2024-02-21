import torch
from torch import nn, einsum
from math import sqrt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Attention(nn.Module):
    def __init__( self, dim, heads = 8, dim_head = 64, dropout = 0.1, gating = True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        nn.init.constant_(self.to_out.weight, 0.)
        nn.init.constant_(self.to_out.bias, 0.)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias = None):
        device, h = x.device, self.heads


        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # query / key similarities

        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)

        dots = dots + attn_bias

        # attention

        dots = dots - dots.max(dim = -1, keepdims = True).values
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating

        gates = self.gating(x)
        out = out * gates.sigmoid()

        # combine to out

        out = self.to_out(out)
        return out


class AxialAttention(nn.Module):
    def __init__(self, dim, heads, row_attn = True, col_attn = True, accept_edges = False, global_query_attn = False, **kwargs):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn

        self.norm = nn.LayerNorm(dim)

        self.attn = Attention(dim = dim, heads = heads, **kwargs)

        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None

    def forward(self, x, edges = None):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'

        b, h, w, d = x.shape

        x = self.norm(x)

        # axial attention

        if self.col_attn:
            axial_dim = w
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'

        elif self.row_attn:
            axial_dim = h
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'

        x = rearrange(x, input_fold_eq)

        attn_bias = self.edges_to_attn_bias(edges)
        attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = axial_dim)

        out = self.attn(x, attn_bias = attn_bias)
        out = rearrange(out, output_fold_eq, h = h, w = w)

        return out

