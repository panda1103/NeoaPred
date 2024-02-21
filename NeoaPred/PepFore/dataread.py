import os
import random
import numpy as np
import pandas as pd
import math
import torch
import torch.utils.data as Data

class DataSet(Data.Dataset):
    def __init__(self, df, all_data):
        super().__init__()
        self.df = df
        self.all_data = all_data
        self.ids, \
        self.rho_wt, self.theta_wt, self.feat_wt, self.mask_wt, \
        self.rho_mut, self.theta_mut, self.feat_mut, self.mask_mut, \
        self.patch_dist, self.patch_mask, self.patch_dist_mp, \
        self.dist, self.dist_mask, self.atom_wt, self.atom_mut = self.getDataset()

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        return self.ids[idx], \
        self.rho_wt[idx], self.theta_wt[idx], self.feat_wt[idx], self.mask_wt[idx], \
        self.rho_mut[idx], self.theta_mut[idx], self.feat_mut[idx], self.mask_mut[idx], \
        self.patch_dist[idx], self.patch_mask[idx], self.patch_dist_mp[idx], \
        self.dist[idx], self.dist_mask[idx], self.atom_wt[idx], self.atom_mut[idx]
 
    def getDataset(self):

        IDs, rho_wt, theta_wt, feat_wt, mask_wt, \
             rho_mut, theta_mut, feat_mut, mask_mut, \
             patch_dist, patch_mask, patch_dist_mp, \
             dist, dist_mask, atom_wt, atom_mut = self.all_data

        select_IDs = self.df["ID"].values
        selected_idx = [np.where(i == IDs)[0][0] for i in select_IDs]
        
        IDs = IDs[selected_idx]
        #wt
        rho_wt = rho_wt[selected_idx]
        theta_wt = theta_wt[selected_idx]
        feat_wt = feat_wt[selected_idx]
        mask_wt = mask_wt[selected_idx]

        #mut
        rho_mut = rho_mut[selected_idx]
        theta_mut = theta_mut[selected_idx]
        feat_mut = feat_mut[selected_idx]
        mask_mut = mask_mut[selected_idx]

        #patch
        patch_dist = patch_dist[selected_idx]
        patch_mask = patch_mask[selected_idx]
        patch_dist_mp = patch_dist_mp[selected_idx]
        
        #dist
        dist = dist[selected_idx]
        dist_mask = dist_mask[selected_idx]

        #atom
        atom_wt = atom_wt[selected_idx]
        atom_mut = atom_mut[selected_idx]

        return IDs, rho_wt, theta_wt, feat_wt, mask_wt, \
                    rho_mut, theta_mut, feat_mut, mask_mut, \
                    patch_dist, patch_mask, patch_dist_mp, \
                    dist, dist_mask, atom_wt, atom_mut

def preload(in_dir):
    path = in_dir + '/'

    IDs = torch.load(path+'IDs.pth')
    
    rho_wt = torch.load(path+'rho_wt.pth')
    theta_wt = torch.load(path+'theta_wt.pth')
    feat_wt = torch.load(path+'feat_wt.pth')
    mask_wt = torch.load(path+'mask_wt.pth')

    rho_mut = torch.load(path+'rho_mut.pth')
    theta_mut = torch.load(path+'theta_mut.pth')
    feat_mut = torch.load(path+'feat_mut.pth')
    mask_mut = torch.load(path+'mask_mut.pth')
    
    patch_dist = torch.load(path+'patch_dist.pth')
    patch_mask = torch.load(path+'patch_mask.pth')
    patch_dist_mp = torch.load(path+'patch_dist_mp.pth')

    dist = torch.load(path+'dist.pth')
    dist_mask = torch.load(path+'dist_mask.pth')
    atom_wt = torch.load(path+'atom_wt.pth')
    atom_mut = torch.load(path+'atom_mut.pth')

    return IDs, rho_wt, theta_wt, feat_wt, mask_wt, \
                rho_mut, theta_mut, feat_mut, mask_mut, \
                patch_dist, patch_mask, patch_dist_mp, \
                dist, dist_mask, atom_wt, atom_mut

