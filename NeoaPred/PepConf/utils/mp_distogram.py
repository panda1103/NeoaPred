import numpy as np
import torch
import torch.nn as nn

def relative_entropy(true, pred):
    true = true.to(torch.float64)
    pred = pred.to(torch.float64)
    kld = torch.sum(true * torch.nn.functional.log_softmax(true, dim=-1), dim = -1) - \
          torch.sum(true * torch.nn.functional.log_softmax(pred, dim=-1) , dim = -1)
    return torch.mean(kld)


def mhc_pep_distogram(
    m_pseudo_beta,
    p_pseudo_beta,
    m_pseudo_beta_mask,
    p_pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
    )
    boundaries = boundaries ** 2
    end_idx = torch.sum(p_pseudo_beta_mask, dim=1)
    p_new = torch.zeros(p_pseudo_beta.shape[0], 6, p_pseudo_beta.shape[-1])
    p_new_mask = torch.ones(p_pseudo_beta.shape[0], 6)
    for i in range(p_pseudo_beta.shape[0]):
        select_6_pos = torch.tensor([0, 1, 2, end_idx[i]-3, end_idx[i]-2, end_idx[i]-1]).to(torch.int64)
        p_new[i] = p_pseudo_beta[i][select_6_pos,:]

    dists = torch.sum(
        (m_pseudo_beta[..., None, :] - p_new[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    #pdb.set_trace()
    true_bins = torch.sum(dists > boundaries, dim=-1)
    pk = torch.nn.functional.one_hot(true_bins, no_bins)
    return pk

def mhc_pep_all_distogram(
    m_pseudo_beta,
    p_pseudo_beta,
    m_pseudo_beta_mask,
    p_pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
    )
    boundaries = boundaries ** 2
    dists = torch.sum(
        (m_pseudo_beta[..., None, :] - p_pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)
    pk = torch.nn.functional.one_hot(true_bins, no_bins)
    return pk
