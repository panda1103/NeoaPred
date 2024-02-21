# Copyright 2022 AlQuraishi Laboratory
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
import contextlib
from functools import partialmethod
import numpy as np
import torch

from NeoaPred.PepConf.utils.tensor_utils import tensor_tree_map


def pad_feature_dict_seq(feature_dict, seqlen):
    """ Pads the sequence length of a feature dict. Used for tracing. """
    # The real sequence length can't be longer than the desired one
    true_n = feature_dict["aatype"].shape[-2]
    assert(true_n <= seqlen)
    
    new_feature_dict = {}
    
    feat_seq_dims = {
        "aatype": -2,
        "between_segment_residues": -1,
        "residue_index": -1,
        "seq_length": -1,
        "deletion_matrix_int": -1,
        "msa": -1,
        "num_alignments": -1,
        "template_aatype": -2,
        "template_all_atom_mask": -2,
        "template_all_atom_positions": -3,
    }

    for k,v in feature_dict.items():
        if(k not in feat_seq_dims):
            new_feature_dict[k] = v
            continue

        seq_dim = feat_seq_dims[k]
        padded_shape = list(v.shape)
        padded_shape[seq_dim] = seqlen
        new_value = np.zeros(padded_shape, dtype=v.dtype)
        new_value[tuple(slice(0, s) for s in v.shape)] = v
        new_feature_dict[k] = new_value
    
    new_feature_dict["seq_length"][0] = seqlen

    return new_feature_dict
