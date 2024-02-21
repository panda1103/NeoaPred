import torch
import torch.nn as nn
from einops import repeat
from einops import rearrange
import numpy as np
import math
import copy

class LinearLayer(nn.Linear):
    def __init__(self, in_size, out_size):
        super(LinearLayer, self).__init__(in_size, out_size)
        with torch.no_grad():
            self.bias.fill_(0)
            nn.init.xavier_uniform_(self.weight, gain=1)

class CosineSimilarityGate(nn.Module):
 
    def __init__(self, dim, out_dim):
        super(CosineSimilarityGate, self).__init__()
        self.gating = nn.Linear(dim, out_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)
 
    def forward(self, tensor_1, tensor_2):
        norm1 = tensor_1.norm(dim=-1, keepdim=True)
        norm2 = tensor_2.norm(dim=-1, keepdim=True)
        normalized_tensor_1 = tensor_1 / norm1
        normalized_tensor_2 = tensor_2 / norm2
        result = normalized_tensor_1 * normalized_tensor_2
        result = self.gating(result)
        result = torch.squeeze(result, dim=-1)
        return result

class QuaLinearDifference(nn.Module):
 
    def __init__(self, input_dim, activation=None):
        super(QuaLinearDifference, self).__init__()
        self.weight_vector = nn.Parameter(torch.Tensor(input_dim*4))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation
        self.bn = nn.BatchNorm1d(input_dim)
        self.reset_parameters()
 
    def reset_parameters(self):
        std = math.sqrt(6 / (self.weight_vector.size(0) + 1))
        self.weight_vector.data.uniform_(-std, std)
        self.bias.data.fill_(0)
 
    def forward(self, tensor_1, tensor_2, eps=1e-8):
        tensor_1 = torch.reshape(tensor_1, [tensor_1.shape[0], tensor_1.shape[2], tensor_1.shape[1]])
        tensor_2 = torch.reshape(tensor_2, [tensor_2.shape[0], tensor_2.shape[2], tensor_2.shape[1]])
        tensor_1 = self.bn(tensor_1)
        tensor_2 = self.bn(tensor_2)
        tensor_1 = torch.reshape(tensor_1, [tensor_1.shape[0], tensor_1.shape[2], tensor_1.shape[1]])
        tensor_2 = torch.reshape(tensor_2, [tensor_2.shape[0], tensor_2.shape[2], tensor_2.shape[1]])
        z_sub = torch.abs(tensor_1 - tensor_2)
        tensor_1[tensor_1==0] = eps
        tensor_2[tensor_2==0] = eps
        z_mul = 1/(tensor_1 * tensor_2)
        combined_tensors = torch.cat([tensor_1, tensor_2, z_sub, z_mul], dim=-1)
        result = torch.matmul(combined_tensors, self.weight_vector) + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result

class Predict(nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.device = device
        self.n_rotations = 16
        self.max_rho = 6
        self.n_rhos = 4
        self.n_thetas = 16
        self.n_feat = 5
        self.n_patchs = 256
        self.sigma_rho_init = self.max_rho/8
        self.sigma_theta_init = 1.0
        self.desc_out_nums = 16

        initial_coords = self.compute_initial_coordinates()
        mu_rho_initial = np.expand_dims(initial_coords[:, 0], 0).astype("float32")
        mu_theta_initial =  np.expand_dims(initial_coords[:, 1], 0).astype("float32")

        self.mu_rho = []
        self.mu_theta = []
        self.sigma_rho = []
        self.sigma_theta = []
        for i in range(self.n_feat):
            self.mu_rho.append(mu_rho_initial)
            self.mu_theta.append(mu_theta_initial)
            self.sigma_rho.append(
                    np.ones_like(mu_rho_initial) * self.sigma_rho_init
            )
            self.sigma_theta.append(
                    np.ones_like(mu_theta_initial) * self.sigma_theta_init
            )
        self.mu_rho = torch.tensor(np.array(self.mu_rho)).to(device)
        self.mu_theta = torch.tensor(np.array(self.mu_theta)).to(device)
        self.sigma_rho = torch.tensor(np.array(self.sigma_rho)).to(device)
        self.sigma_theta = torch.tensor(np.array(self.sigma_theta)).to(device)
        self.mu_rho_wt = copy.deepcopy(self.mu_rho)
        self.mu_theta_wt = copy.deepcopy(self.mu_theta)
        self.sigma_rho_wt = copy.deepcopy(self.sigma_rho)
        self.sigma_theta_wt = copy.deepcopy(self.sigma_theta)
        self.mu_rho_mut = copy.deepcopy(self.mu_rho)
        self.mu_theta_mut = copy.deepcopy(self.mu_theta)
        self.sigma_rho_mut = copy.deepcopy(self.sigma_rho)
        self.sigma_theta_mut = copy.deepcopy(self.sigma_theta)
        
        linearlayer = LinearLayer(self.n_rhos*self.n_thetas, self.n_rhos*self.n_thetas)
        self.linear_layers = nn.ModuleList(
            [copy.deepcopy(linearlayer) for _ in range(self.n_feat)])

        self.DescLN = nn.Sequential(
                                    nn.Linear(self.n_rhos*self.n_thetas, self.n_rhos*self.n_thetas),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(self.n_rhos*self.n_thetas, self.desc_out_nums),
                                    nn.ReLU(),
                               )

        self.Diff = QuaLinearDifference(self.desc_out_nums*self.n_feat, nn.Sigmoid())
        self.DiffLN = nn.Sequential(
                                    nn.Linear(self.n_patchs, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.BatchNorm1d(64),
                                    nn.ReLU(),
                                 )
        self.AtomEmb = nn.Embedding(38, 64)
        self.pos_len = 128
        self.max_rel_dist = 64
        self.PosEmb = nn.Embedding(self.max_rel_dist * 2 + 1, 8)

        self.Dist = nn.Sequential(
                                    nn.Conv2d(9, 32, kernel_size=(5,5), stride=2, padding=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),
                                    nn.Conv2d(32, 1, kernel_size=(3,3), stride=1, padding=1),
                                    nn.Sigmoid(),
                                    nn.BatchNorm2d(1),
                                    nn.Flatten(),
                                    nn.Linear(64*64, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 16*16),
                                    nn.BatchNorm1d(16*16),
                                    nn.ReLU(),
                                    nn.Linear(16*16, 32),
                                    nn.ReLU(),
                               )
        self.Atom = nn.Sequential(
                                    nn.Conv2d(128, 32, kernel_size=(5,5), stride=2, padding=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),
                                    nn.Conv2d(32, 1, kernel_size=(3,3), stride=1, padding=1),
                                    nn.Sigmoid(),
                                    nn.BatchNorm2d(1),
                                    nn.Flatten(),
                                    nn.Linear(64*64, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 16*16),
                                    nn.BatchNorm1d(16*16),
                                    nn.ReLU(),
                                    nn.Linear(16*16, 32),
                                    nn.ReLU(),
                               )

        self.Classify = nn.Sequential(
                                    nn.Linear(64+32+32, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Linear(128, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.BatchNorm1d(64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1),
                                    nn.Sigmoid(),
                                )


    def inference(
        self,
        rho_coords,
        theta_coords,
        input_feat,
        mask,
        layer,
        mu_rho,
        sigma_rho,
        mu_theta,
        sigma_theta,
        eps=1e-8,
        mean_gauss_activation=True,
    ):
        n_samples = rho_coords.shape[0]
        n_patchs = rho_coords.shape[1]
        n_vertices = rho_coords.shape[2]

        all_conv_feat = []
        for k in range(self.n_rotations):
            rho_coords_ = torch.reshape(rho_coords, [-1, 1])  # batch_size*n_patch*n_vertices
            thetas_coords_ = torch.reshape(theta_coords, [-1, 1])  # batch_size*n_patch*n_vertices

            thetas_coords_ += k * 2 * np.pi / self.n_rotations
            thetas_coords_ = torch.fmod(thetas_coords_, 2 * np.pi)
            rho_coords_ = torch.exp(
                -torch.square(rho_coords_ - mu_rho) / (torch.square(sigma_rho) + eps)
            )
            thetas_coords_ = torch.exp(
                -torch.square(thetas_coords_ - mu_theta) / (torch.square(sigma_theta) + eps)
            )
            gauss_activations = torch.mul(
                rho_coords_, thetas_coords_
            )  # batch_size*n_patch*n_vertices, n_gauss
            gauss_activations = torch.reshape(
                gauss_activations, [n_samples, n_patchs, n_vertices, -1]
            )  # batch_size, n_patchs, n_vertices, n_gauss
            mask = torch.reshape(mask, [n_samples, n_patchs, n_vertices, 1])
            gauss_activations = torch.mul(gauss_activations, mask)
            if (
                mean_gauss_activation
            ):  # computes mean weights for the different gaussians
                gauss_activations /= (
                    torch.sum(gauss_activations, dim=2, keepdim=True) + eps
                )  # batch_size, n_patchs, n_vertices, n_gauss
            input_feat_ = torch.unsqueeze(
                input_feat, 3
            )  # batch_size, n_patchs, n_vertices, 1
            gauss_desc = torch.mul(
                gauss_activations, input_feat_
            )  # batch_size, n_patchs, n_vertices, n_gauss,
            gauss_desc = torch.sum(gauss_desc, dim=2)  # batch_size, n_patchs, n_gauss,
            gauss_desc = torch.reshape(
                gauss_desc, [n_samples, n_patchs, self.n_thetas * self.n_rhos]
            )  # batch_size, n_patchs, n_thetas*n_rhos
            conv_feat = layer(gauss_desc)  # batch_size, n_patchs, n_thetas*n_rhos
            all_conv_feat.append(conv_feat)
        all_conv_feat = torch.stack(all_conv_feat)
        conv_feat = torch.max(all_conv_feat, dim=0)[0]
        conv_feat = torch.relu(conv_feat)
        return conv_feat

    def compute_initial_coordinates(self):
        range_rho = [0.0, self.max_rho]
        range_theta = [0, 2 * np.pi]

        grid_rho = np.linspace(range_rho[0], range_rho[1], num=self.n_rhos + 1)
        grid_rho = grid_rho[1:]
        grid_theta = np.linspace(range_theta[0], range_theta[1], num=self.n_thetas + 1)
        grid_theta = grid_theta[:-1]

        grid_rho_, grid_theta_ = np.meshgrid(grid_rho, grid_theta, sparse=False)
        grid_rho_ = (
            grid_rho_.T
        )  # the traspose here is needed to have the same behaviour as Matlab code
        grid_theta_ = (
            grid_theta_.T
        )  # the traspose here is needed to have the same behaviour as Matlab code
        grid_rho_ = grid_rho_.flatten()
        grid_theta_ = grid_theta_.flatten()

        coords = np.concatenate((grid_rho_[None, :], grid_theta_[None, :]), axis=0)
        coords = coords.T  # every row contains the coordinates of a grid intersection
        return coords


    def forward(self, rho_wt, theta_wt, feat_wt, mask_wt,
                      rho_mut, theta_mut, feat_mut, mask_mut,
                      patch_dist, patch_mask, patch_dist_mp, 
                      dist, dist_mask, atom_wt, atom_mut, eps=1e-8):

        #patch feat
        global_desc_wt = []
        global_desc_mut = []
        for i in range(feat_wt.shape[-1]):
            infer_feat_wt = feat_wt[...,i]
            infer_feat_mut = feat_mut[...,i]
            layer = self.linear_layers[i]
            desc_wt = self.inference(
                                    rho_wt,
                                    theta_wt,
                                    infer_feat_wt,
                                    mask_wt,
                                    layer,
                                    self.mu_rho_wt[i],
                                    self.sigma_rho_wt[i],
                                    self.mu_theta_wt[i],
                                    self.sigma_theta_wt[i],
                                    )
            global_desc_wt.append(desc_wt)
            desc_mut = self.inference(
                                    rho_mut,
                                    theta_mut,
                                    infer_feat_mut,
                                    mask_mut,
                                    layer,
                                    self.mu_rho_mut[i],
                                    self.sigma_rho_mut[i],
                                    self.mu_theta_mut[i],
                                    self.sigma_theta_mut[i],
                                    )
            global_desc_mut.append(desc_mut)
        
        global_desc_wt = torch.stack(global_desc_wt, axis=1)
        global_desc_mut = torch.stack(global_desc_mut, axis=1)
        global_desc_wt = self.DescLN(global_desc_wt)
        global_desc_mut = self.DescLN(global_desc_mut)
       
        global_desc_wt = torch.reshape(global_desc_wt, [global_desc_wt.shape[0], self.n_patchs, -1])
        global_desc_mut = torch.reshape(global_desc_mut, [global_desc_mut.shape[0], self.n_patchs, -1])
        
        diff = self.Diff(global_desc_wt, global_desc_mut)
        
        patch_dist_mp = torch.mean(patch_dist_mp, dim=-1) * patch_mask

        diff = diff * patch_dist_mp

        diff = self.DiffLN(diff)
        
        #dist
        pos_index = torch.arange(self.pos_len).to(self.device)
        rel_pos = rearrange(pos_index, 'i -> () i ()') - rearrange(pos_index, 'j -> () () j')
        rel_pos = rel_pos.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        rel_pos_emb = self.PosEmb(rel_pos)
        rel_pos_emb = torch.reshape(rel_pos_emb, [rel_pos_emb.shape[0], rel_pos_emb.shape[-1], rel_pos_emb.shape[1], rel_pos_emb.shape[2]])
        rel_pos_emb = repeat(rel_pos_emb, 'b h i j -> (b x) h i j', x = dist.shape[0])
        dist = torch.reshape(dist, [dist.shape[0], 1, dist.shape[-2], dist.shape[-1]])

        dist = torch.cat([dist, rel_pos_emb], dim=1)

        dist = self.Dist(dist)

        #atom
        atom_wt = self.AtomEmb(atom_wt)
        atom_mut = self.AtomEmb(atom_mut)
        atom_wt = atom_wt.transpose(1, 2)
        atom_mut = atom_mut.transpose(1, 2)

        dif = torch.abs(atom_wt.unsqueeze(3) - atom_mut.unsqueeze(2))
        mul = atom_wt.unsqueeze(3) * atom_mut.unsqueeze(2)

        atom = torch.cat([dif, mul], 1)

        atom = self.Atom(atom)
        
        #classify
        x = torch.cat([diff, dist, atom], -1)
        res = self.Classify(x)
        res = res.view(1,-1)[0]
        return res
