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

import torch
import torch.nn as nn
from NeoaPred.PepConf.utils.primitives import Linear, LayerNorm
from NeoaPred.PepConf.utils.loss import (
    compute_plddt,
    compute_tm,
    compute_predicted_aligned_error,
)


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config["experimentally_resolved"],
        )

        if config.tm.enabled:
            self.tm = TMScoreHead(
                **config.tm,
            )

        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        experimentally_resolved_logits = self.experimentally_resolved(
            outputs["single"]
        )
        aux_out[
            "experimentally_resolved_logits"
        ] = experimentally_resolved_logits

        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits
            aux_out["predicted_tm_score"] = compute_tm(
                tm_logits, **self.config.tm
            )
            aux_out.update(
                compute_predicted_aligned_error(
                    tm_logits,
                    **self.config.tm,
                )
            )

        return aux_out


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, no_bins, c_in, c_hidden):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, pair_dim, no_bins, **kwargs):
        """
        Args:
            pair_dim:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.pair_dim = pair_dim
        self.no_bins = no_bins

        self.linear = Linear(self.pair_dim, self.no_bins, init="final")

    def _forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits
    
    def forward(self, z): 
        return self._forward(z)


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """

    def __init__(self, pair_dim, no_bins, **kwargs):
        """
        Args:
            pair_dim:
                Input channel dimension
            no_bins:
                Number of bins
        """
        super(TMScoreHead, self).__init__()

        self.pair_dim = pair_dim
        self.no_bins = no_bins

        self.linear = Linear(self.pair_dim, self.no_bins, init="final")

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, no_bins] prediction
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        return logits

class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, single_dim, out_dim, **kwargs):
        """
        Args:
            single_dim:
                Input channel dimension
            out_dim:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.single_dim = single_dim
        self.out_dim = out_dim

        self.linear = Linear(self.single_dim, self.out_dim, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits
