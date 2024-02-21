# Copyright 2023 AIDuhl Laboratory
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import reduce
import importlib
import math
import sys
import copy
from operator import mul
from einops import rearrange
from einops import repeat

import torch
import torch.nn as nn
from typing import Optional, Tuple, Sequence

from NeoaPred.PepConf.model.invariant_point_attention import IPABlock
from NeoaPred.PepConf.utils.primitives import Linear, LayerNorm
from NeoaPred.PepConf.utils.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from NeoaPred.PepConf.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from NeoaPred.PepConf.utils.rigid_utils import Rotation, Rigid
from NeoaPred.PepConf.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)

class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s

class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, single_dim):
        """
        Args:
            single_dim:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.single_dim = single_dim

        self.linear = Linear(self.single_dim, 6, init="final")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        update = self.linear(s)

        return update 


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class Decoder(nn.Module):
    def __init__(
        self,
        single_dim,
        pair_dim,
        ipa_dim,
        resnet_dim,
        no_heads_ipa,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        **kwargs,
    ):
        """
        #################################################################
        #  Decoder is modified from openfold's StructureModule.
        #  The MHC rigids in IPA block is initialized from transformed
        #  true structure data.
        #################################################################
        Args:
            single_dim:
                Single representation channel dimension
            pair_dim:
                Pair representation channel dimension
            ipa_dim:
                IPA hidden channel dimension
            resnet_dim:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alphafold2: Algorithm. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
        """
        super(Decoder, self).__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.ipa_dim = ipa_dim
        self.resnet_dim = resnet_dim
        self.no_heads_ipa = no_heads_ipa
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon

        self.layer_norm_s = LayerNorm(self.single_dim)
        self.layer_norm_z = LayerNorm(self.pair_dim)

        self.linear_in = Linear(self.single_dim, self.single_dim)
        self.ipa = IPABlock(dim=self.single_dim, heads=self.no_heads_ipa)
        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.single_dim)

        self.transition = StructureModuleTransition(
            self.single_dim,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.bb_update = BackboneUpdate(self.single_dim)

        self.angle_resnet = AngleResnet(
            self.single_dim,
            self.resnet_dim,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def forward(
        self,
        single_repr,
        pair_repr,
        aatype,
        mhc_bb,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            single_repr:
                [*, N_res, C_s] single representation
            pair_repr:
                [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mhc_bb:
                [*, mhc_crop_size, 4, 4] MHC backbone rigid
        Returns:
            A dictionary of outputs
        """
        s = single_repr
        z = pair_repr
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        if(_offload_inference):
            #assert(sys.getrefcount(pair_repr) == 8)
            assert(sys.getrefcount(pair_repr) == 6)
            pair_repr = pair_repr.cpu()

        s_initial = s
        s = self.linear_in(s)

        mhc_rigids = copy.deepcopy(Rigid.from_tensor_4x4(mhc_bb))
        mhc_crop_size = mhc_rigids.shape[1]
        pep_crop_size = aatype.shape[1]
        pep_rigids = Rigid.identity(
            torch.tensor([s.shape[0],pep_crop_size]),
            s.dtype, 
            s.device, 
            requires_grad = self.training, 
            fmt="rot_mat",
        )
        outputs = []
        for i in range(self.no_blocks):
            mhc_rigids._rots.get_rot_mats().requires_grad = True
            mhc_rigids.get_trans().requires_grad = True
            mhc_pep_rots = torch.cat((mhc_rigids._rots.get_rot_mats(), pep_rigids._rots.get_rot_mats()), dim=1)
            mhc_pep_trans = torch.cat((mhc_rigids.get_trans(), pep_rigids.get_trans()), dim=1)
            s = s + self.ipa(
                single_repr = s, 
                pairwise_repr = z, 
                rotations = mhc_pep_rots, 
                translations = mhc_pep_trans, 
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)
            
            pep_s = s[:,mhc_crop_size:,:]
            pep_rigids = pep_rigids.compose_q_update_vec(self.bb_update(pep_s))
            backb_to_global = Rigid(
                Rotation(
                    rot_mats=pep_rigids.get_rots().get_rot_mats(), 
                    quats=None
                ),
                pep_rigids.get_trans(),
            )
            backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)
            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s[:,mhc_crop_size:,:], s_initial[:,mhc_crop_size:,:])
            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype,
            )
            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )
            scaled_rigids = pep_rigids.scale_translation(self.trans_scale_factor)
            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": pep_s,
            }

            outputs.append(preds)

            pep_rigids = pep_rigids.stop_rot_gradient()

        del z
        
        if(_offload_inference):
            pair_repr = (pair_repr.to(s.device))

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = pep_s

        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
