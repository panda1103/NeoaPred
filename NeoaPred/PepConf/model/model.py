import torch
from torch import nn, einsum
from einops import repeat
from einops import rearrange
import sys
from NeoaPred.PepConf.utils.feats import atom14_to_atom37
from NeoaPred.PepConf.embedder.embedder import Embedding
from NeoaPred.PepConf.model.encoder import Encoder
from NeoaPred.PepConf.model.heads import AuxiliaryHeads, DistogramHead
from NeoaPred.PepConf.model.decoder import Decoder

class MPSP(nn.Module):
    """
    MPSP (Mhc-Peptide Complex Structure Prediction)
    """
    def __init__(self, config, name, device):
        super(MPSP, self).__init__()
        self.config = config.model_config(name=name)
        self.device = device
        self.max_rel_dist = self.config.model["max_rel_dist"]
        self.rel_pos_dim = self.config.model["rel_pos_dim"]
        self.pos_emb = nn.Embedding(self.max_rel_dist * 2 + 1, self.rel_pos_dim)
        self.embedder = Embedding(**self.config.model.embedder)
        self.encoder = Encoder(**self.config.model.encoder)
        self.decoder = Decoder(**self.config.model.decoder)
        self.aux_heads = AuxiliaryHeads(self.config.model["heads"])
        self.distogram = DistogramHead(**self.config.model["heads"]["distogram"])

    def forward(self, mhc_feats, pep_feats, feats):
        p_outputs = {}
        mp_outputs = {}
        mhc_bb = mhc_feats["backbone_rigid_tensor"].to(self.device)
        mhc_len = self.config.model.decoder["mhc_len"]
        pep_len = self.config.model.decoder["pep_len"]
        x_aatype = feats["aatype"].to(self.device)
        p_aatype = pep_feats["aatype"].to(self.device)
        #emb
        x_emb = self.embedder(x_aatype)
        x_pair = rearrange(x_emb, 'b i d -> b i () d') + rearrange(x_emb, 'b j d-> b () j d')
        
        mp_index = torch.arange(mhc_len + pep_len).to(self.device)
        rel_pos = rearrange(mp_index, 'i -> () i ()') - rearrange(mp_index, 'j -> () () j')
        rel_pos = rel_pos.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        rel_pos_emb = self.pos_emb(rel_pos)
        
        x_pair = x_pair + rel_pos_emb
        #encoder
        x_en = self.encoder(x_pair)
        #
        x_pair = x_pair.cpu()
        p_outputs["single"] = x_emb[:,mhc_len:,:]
        p_outputs["pair"] = x_en[:,mhc_len:,mhc_len:,:]

        #decoder and generate structure information
        p_sm = self.decoder(x_emb, x_en, p_aatype, mhc_bb=mhc_bb, inplace_safe=True, _offload_inference=True)
        #
        del x_aatype, p_aatype, mhc_bb
        x_pair = x_pair.to(self.device)
        #get final structure information
        p_outputs.update({"sm":p_sm})
        p_outputs["final_atom_positions"] = atom14_to_atom37(p_outputs["sm"]["positions"][-1], pep_feats)
        p_outputs["final_atom_mask"] = pep_feats["atom37_atom_exists"].to(self.device)
        p_outputs["final_affine_tensor"] = p_outputs["sm"]["frames"][-1].to(self.device)
        p_outputs.update(self.aux_heads(p_outputs))
        
        #get mhc-pep distogram
        all_distogram_logits = self.distogram(x_pair)
        mp_dist_1, mp_dist_2 = torch.split(all_distogram_logits, (mhc_len, pep_len), dim=1)
        _, mp_dist_1 = torch.split(mp_dist_1, (mhc_len, pep_len), dim=2)
        mp_dist_2, _ = torch.split(mp_dist_2, (mhc_len, pep_len), dim=2)
        mp_dist = torch.cat((mp_dist_1, mp_dist_2.transpose(1,2)), dim=3)
        mp_outputs["distogram_logits"] = mp_dist
        return p_outputs, mp_outputs
