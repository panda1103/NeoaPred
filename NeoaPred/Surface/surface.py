#Compute molecular surfaces features

import os
import numpy as np
import pandas as pd
import multiprocessing
import torch
import pymesh
import math
import torch.utils.data as Data
from scipy.spatial import cKDTree
from NeoaPred.masif_tools.default_config.masif_opts import masif_opts
from NeoaPred.masif_tools.input_output.extractPDB import extractPDB
from NeoaPred.masif_tools.input_output.save_ply import mesh_feat_compute
from NeoaPred.masif_tools.input_output.patch_feat_compute import save_select_ply, ss_patch_feat_compute
from NeoaPred.masif_tools.computeDsit import get_atom_and_dist

atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i+1 for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.


def padding(matrix, padding_patch_size):
    patch_size = matrix.shape[0]
    if(padding_patch_size > patch_size):
        padd_len = padding_patch_size - patch_size
        shape = list(matrix.shape)
        shape[0] = padd_len
        padd = np.zeros(shape)
        matrix = np.concatenate((matrix, padd), axis=0)
    else:
        matrix = matrix[:padding_patch_size,]
    return matrix

def filter_pdb_header(in_dir, out_dir, df):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filter_list = []
    for idx, row in df.iterrows():
        raw_con = in_dir+"/"+row["ID"]+"_relaxed.pdb"
        f_con = out_dir+"/"+row["ID"]+"_complex.pdb"
        cmd = 'cat %s | grep -v ^PARENT > %s' % (raw_con, f_con)
        os.system(cmd)
        filter_list.append([row["ID"], f_con])
    return filter_list

def mesh_deal(ID, in_file, out_pep_pdb, out_mhc_pdb, out_pep_prefix):
    #deal pep
    extractPDB(in_file, out_pep_pdb, "B")
    mesh_feat_compute(
                        molecule = 'peptide',
                        chain_pdb_filename = out_pep_pdb,
                        out_filename_prefix = out_pep_prefix,
                     )
    #deal mhc
    extractPDB(in_file, out_mhc_pdb, "A")
    print ('Mesh feat compute done.')

def mesh_deal_mulproc(params):
    return mesh_deal(params[0], params[1], params[2], params[3], params[4])

def pdb2ply(out_dir, filter_list, threads):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    params = []
    pep_ply_list = []
    for ID, in_file in filter_list:
        out_pep_pdb = out_dir + "/" + ID + "_pep.pdb"
        out_mhc_pdb = out_dir + "/" + ID + "_mhc.pdb"
        out_pep_ply = out_dir + "/" + ID + "_pep.ply"
        out_pep_prefix = out_dir + "/" + ID + "_pep"
        params.append([ID, in_file, out_pep_pdb, out_mhc_pdb, out_pep_prefix])
        pep_ply_list.append([ID, out_pep_pdb, out_mhc_pdb, out_pep_ply])
    with multiprocessing.Pool(processes=threads) as pool:
        pool.map(mesh_deal_mulproc, params)
    return pep_ply_list

def for_feat_comp(out_prefix, wt_pep_ply, mut_pep_ply, wt_mhc_pdb, mut_mhc_pdb, wt_si_ddc_dm_ply, mut_si_ddc_dm_ply):
    opts = masif_opts["ppi_search"]
    ss_patch_feat_compute(out_prefix, wt_pep_ply, mut_pep_ply, wt_mhc_pdb, mut_mhc_pdb, wt_si_ddc_dm_ply, mut_si_ddc_dm_ply, opts)

def for_feat_comp_mulproc(params):
    for_feat_comp(params[0], params[1], params[2], params[3], params[4], params[5], params[6])

def ply2ImFeat(in_dir, out_dir, pdb_ply_list2, threads):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    params = []
    for ID, _, wt_pep_ply, _, mut_pep_ply, wt_mhc_pdb, _, mut_mhc_pdb, _ in pdb_ply_list2:
        out_prefix = out_dir + "/" + ID + "_"
        wt_si_ddc_dm_ply = out_dir + "/" + ID + '_wt_si_ddc_dm.ply'
        mut_si_ddc_dm_ply = out_dir + "/" + ID + '_mut_si_ddc_dm.ply'
        params.append([out_prefix, wt_pep_ply, mut_pep_ply, wt_mhc_pdb, mut_mhc_pdb, wt_si_ddc_dm_ply, mut_si_ddc_dm_ply])
    with multiprocessing.Pool(processes=threads) as pool:
        pool.map(for_feat_comp_mulproc, params)

def pdb2ImAtom(out_dir, pdb_ply_list2):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for ID, _, _, _, _, _, wt_pep_pdb, _, mut_pep_pdb in pdb_ply_list2:
        select_wt_atom = out_dir + "/" + ID + "_wt_atom.npy"
        select_mut_atom = out_dir + "/" + ID + "_mut_atom.npy"
        out_dist = out_dir + "/" + ID + "_dist.npy"
        get_atom_and_dist(
                        wt_pep_pdb,
                        mut_pep_pdb,
                        select_wt_atom,
                        select_mut_atom,
                        out_dist,
                       )

def for_feat_filter(feat_dir, out_featfilt_dir, ID, padding_patch_size):
    dm_threshold = 4.5
    ply_wt = feat_dir + "/" + ID + "_wt_si_ddc_dm.ply"
    ply_mut = feat_dir + "/" + ID + "_mut_si_ddc_dm.ply"
    new_ply_wt = out_featfilt_dir + '/' + ID + '_wt_outer_surface.ply'
    new_ply_mut = out_featfilt_dir + '/' + ID + '_mut_outer_surface.ply'

    mesh1 = pymesh.load_mesh(ply_wt)
    mesh2 = pymesh.load_mesh(ply_mut)

    v1 = mesh1.vertices
    v2 = mesh2.vertices

    #select dm
    select_idx_wt = np.where(mesh1.get_attribute('vertex_dm') > dm_threshold)[0]
    select_idx_mut = np.where(mesh2.get_attribute('vertex_dm') > dm_threshold)[0]

    #load feat
    wt_rho_wrt_center = np.load(feat_dir + "/" + ID + '_wt_rho_wrt_center.npy')
    wt_theta_wrt_center = np.load(feat_dir + "/" + ID + '_wt_theta_wrt_center.npy')
    wt_input_feat = np.load(feat_dir + "/" + ID + '_wt_input_feat.npy')
    wt_mask = np.load(feat_dir + "/" + ID + '_wt_mask.npy')

    mut_rho_wrt_center = np.load(feat_dir + "/" + ID + '_mut_rho_wrt_center.npy')
    mut_theta_wrt_center = np.load(feat_dir + "/" + ID + '_mut_theta_wrt_center.npy')
    mut_input_feat = np.load(feat_dir + "/" + ID + '_mut_input_feat.npy')
    mut_mask = np.load(feat_dir + "/" + ID + '_mut_mask.npy')

    kdt = cKDTree(v1)
    d1, r1= kdt.query(v2[select_idx_mut])

    kdt = cKDTree(v2)
    d2, r2= kdt.query(v1[select_idx_wt])

    k1 = np.concatenate([r1, select_idx_wt])
    k2 = np.concatenate([select_idx_mut, r2])
    d = np.concatenate([d1, d2])

    #remove duplicate vertices
    assert (len(k1) == len(k2))
    check_dict = {}
    tmp_k1 = np.array([], dtype='int32')
    tmp_k2 = np.array([], dtype='int32')
    tmp_d = np.array([], dtype='float')
    for i in range(len(k1)):
        key = str(k1[i]) + '_' + str(k2[i])
        if not (key in check_dict.keys()):
            tmp_k1 = np.append(tmp_k1, k1[i])
            tmp_k2 = np.append(tmp_k2, k2[i])
            tmp_d = np.append(tmp_d, d[i])
            check_dict[key] = key
    k1 = tmp_k1
    k2 = tmp_k2
    d = tmp_d

    save_select_ply(mesh1, k1, new_ply_wt)
    save_select_ply(mesh2, k2, new_ply_mut)

    d_mp_wt = mesh1.get_attribute('vertex_dm')[k1]
    d_mp_mut = mesh2.get_attribute('vertex_dm')[k2]
    d_mp = np.stack([d_mp_wt, d_mp_mut]).T

    wt_rho_wrt_center = padding(wt_rho_wrt_center[k1], padding_patch_size)
    wt_theta_wrt_center = padding(wt_theta_wrt_center[k1], padding_patch_size)
    wt_input_feat = padding(wt_input_feat[k1], padding_patch_size)
    wt_mask = padding(wt_mask[k1], padding_patch_size)

    mut_rho_wrt_center = padding(mut_rho_wrt_center[k2], padding_patch_size)
    mut_theta_wrt_center = padding(mut_theta_wrt_center[k2], padding_patch_size)
    mut_input_feat = padding(mut_input_feat[k2], padding_patch_size)
    mut_mask = padding(mut_mask[k2], padding_patch_size)

    patch_mask = np.ones(len(k1))
    patch_mask = padding(patch_mask, padding_patch_size)
    d = padding(d, padding_patch_size)
    d_mp = padding(d_mp, padding_patch_size)

    np.save(out_featfilt_dir + '/' + ID + '_wt_rho_wrt_center.npy', wt_rho_wrt_center)
    np.save(out_featfilt_dir + '/' + ID + '_wt_theta_wrt_center.npy', wt_theta_wrt_center)
    np.save(out_featfilt_dir + '/' + ID + '_wt_input_feat.npy', wt_input_feat)
    np.save(out_featfilt_dir + '/' + ID + '_wt_mask.npy', wt_mask)

    np.save(out_featfilt_dir + '/' + ID + '_mut_rho_wrt_center.npy', mut_rho_wrt_center)
    np.save(out_featfilt_dir + '/' + ID + '_mut_theta_wrt_center.npy', mut_theta_wrt_center)
    np.save(out_featfilt_dir + '/' + ID + '_mut_input_feat.npy', mut_input_feat)
    np.save(out_featfilt_dir + '/' + ID + '_mut_mask.npy', mut_mask)

    np.save(out_featfilt_dir + '/' + ID + '_patch_mask.npy', patch_mask)
    np.save(out_featfilt_dir + '/' + ID + '_patch_dist.npy', d)
    np.save(out_featfilt_dir + '/' + ID + '_mhc_patch_dist.npy', d_mp)


def for_feat_filter_mulproc(params):
    for_feat_filter(params[0], params[1], params[2], params[3])

def for_feat_f(feat_dir, out_featfilt_dir, df, padding_patch_size, threads):
    if not os.path.exists(out_featfilt_dir):
        os.makedirs(out_featfilt_dir)
    params = []
    for idx, row in df.iterrows():
        ID = row["SampleID"]
        params.append([feat_dir, out_featfilt_dir, ID, padding_patch_size])
    with multiprocessing.Pool(processes=threads) as pool:
        pool.map(for_feat_filter_mulproc, params)

def for_dist_pad(atom_dist_dir, df, padding_size):
    for idx, row in df.iterrows():
        ID = row["SampleID"]
        dist = np.load(atom_dist_dir + '/' + ID + '_dist.npy')
        dist_mask = np.ones_like(dist)
        wt_atom_name = np.load(atom_dist_dir + '/' + ID + '_wt_atom.npy', allow_pickle=True).item()['atom']
        mut_atom_name = np.load(atom_dist_dir + '/' + ID + '_mut_atom.npy', allow_pickle=True).item()['atom']

        dist = padding(dist, padding_size)
        dist = padding(dist.T, padding_size).T
        dist_mask = padding(dist_mask, padding_size)
        dist_mask = padding(dist_mask.T, padding_size).T

        wt_atom_name = np.array([atom_order[i.split("_",2)[2]] for i in wt_atom_name])
        wt_atom_name = padding(wt_atom_name, padding_size)
        mut_atom_name = np.array([atom_order[i.split("_",2)[2]] for i in mut_atom_name])
        mut_atom_name = padding(mut_atom_name, padding_size)

        np.save(atom_dist_dir + '/' + ID +'_dist_padding.npy', dist)
        np.save(atom_dist_dir + '/' + ID +'_dist_mask_padding.npy', dist_mask)
        np.save(atom_dist_dir + '/' + ID +'_wt_atom_padding.npy', wt_atom_name)
        np.save(atom_dist_dir + '/' + ID +'_mut_atom_padding.npy', mut_atom_name)

def merge_for_cache(in_feat_dir, atom_dist_dir, cache_dir, df, neigh_num):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    #
    IDs = df["SampleID"].values
   
    #wt
    rho_wt =[]
    theta_wt = []
    feat_wt = []
    mask_wt = []

    #mut
    rho_mut =[]
    theta_mut = []
    feat_mut = []
    mask_mut = []

    #patch
    patch_dist = []
    patch_mask = []
    patch_dist_mp = []
    
    #dist
    dist = []
    dist_mask = []
    atom_wt = []
    atom_mut = []

    for ID in IDs:
        rho_wt.append(np.load(in_feat_dir + '/' + ID + '_wt_rho_wrt_center.npy')[:,0:neigh_num])
        theta_wt.append(np.load(in_feat_dir + '/' + ID + '_wt_theta_wrt_center.npy')[:,0:neigh_num])
        feat_wt.append(np.load(in_feat_dir + '/' + ID + '_wt_input_feat.npy')[:,0:neigh_num])
        mask_wt.append(np.load(in_feat_dir + '/' + ID + '_wt_mask.npy')[:,0:neigh_num])

        rho_mut.append(np.load(in_feat_dir + '/' + ID + '_mut_rho_wrt_center.npy')[:,0:neigh_num])
        theta_mut.append(np.load(in_feat_dir + '/' + ID + '_mut_theta_wrt_center.npy')[:,0:neigh_num])
        feat_mut.append(np.load(in_feat_dir + '/' + ID + '_mut_input_feat.npy')[:,0:neigh_num])
        mask_mut.append(np.load(in_feat_dir + '/' + ID + '_mut_mask.npy')[:,0:neigh_num])
        
        patch_dist.append(np.load(in_feat_dir + '/' + ID + '_patch_dist.npy'))
        patch_mask.append(np.load(in_feat_dir + '/' + ID + '_patch_mask.npy'))
        patch_dist_mp.append(np.load(in_feat_dir + '/' + ID + '_mhc_patch_dist.npy'))

        dist.append(np.load(atom_dist_dir + '/' + ID + '_dist_padding.npy'))
        dist_mask.append(np.load(atom_dist_dir + '/' + ID + '_dist_mask_padding.npy'))
        atom_wt.append(np.load(atom_dist_dir + '/' + ID + '_wt_atom_padding.npy'))
        atom_mut.append(np.load(atom_dist_dir + '/' + ID + '_mut_atom_padding.npy'))


    torch.save(IDs, cache_dir + '/'  + 'IDs.pth')
    torch.save(torch.FloatTensor(rho_wt), cache_dir + '/'  + 'rho_wt.pth')
    torch.save(torch.FloatTensor(theta_wt), cache_dir + '/'  + 'theta_wt.pth')
    torch.save(torch.FloatTensor(feat_wt), cache_dir + '/'  + 'feat_wt.pth')
    torch.save(torch.FloatTensor(mask_wt), cache_dir + '/'  + 'mask_wt.pth')
    torch.save(torch.FloatTensor(rho_mut), cache_dir + '/'  + 'rho_mut.pth')
    torch.save(torch.FloatTensor(theta_mut), cache_dir + '/'  + 'theta_mut.pth')
    torch.save(torch.FloatTensor(feat_mut), cache_dir + '/'  + 'feat_mut.pth')
    torch.save(torch.FloatTensor(mask_mut), cache_dir + '/'  + 'mask_mut.pth')
    torch.save(torch.FloatTensor(patch_dist), cache_dir + '/'  + 'patch_dist.pth')
    torch.save(torch.FloatTensor(patch_mask), cache_dir + '/'  + 'patch_mask.pth')
    torch.save(torch.FloatTensor(patch_dist_mp), cache_dir + '/'  + 'patch_dist_mp.pth')
    torch.save(torch.FloatTensor(dist), cache_dir + '/'  + 'dist.pth')
    torch.save(torch.FloatTensor(dist_mask), cache_dir + '/'  + 'dist_mask.pth')
    torch.save(torch.LongTensor(atom_wt), cache_dir + '/' + 'atom_wt.pth')
    torch.save(torch.LongTensor(atom_mut), cache_dir + '/' + 'atom_mut.pth')

def compute_surface(
                        input_dir,
                        input_file,
                        output_dir,
                        output_file,
                        threads = 4,
                        for_patch_pad_size = 256,
                        dist_pad_size = 128,
                        for_neigh_num = 10,
                      ):
    df = pd.read_csv(input_file, header=0)
    out_pdb_dir = output_dir + "/Complex_PDB"
    out_ply_dir = output_dir + "/PLY"
    out_for_feat_dir = output_dir + "/Feat"
    out_for_atom_dist_dir = output_dir + "/AtomDist"
    out_for_featfilt_dir = output_dir + "/FeatFilter"
    out_for_cache_dir = output_dir + "/Cache"
    
    #filter
    filter_list = filter_pdb_header(input_dir, out_pdb_dir, df)

    #pdb2ply
    try:
        pep_ply_list = pdb2ply(out_ply_dir, filter_list, threads)
    except:
        print("Error in surface.pdb2ply")
    
    #pdb_ply_list2
    df_pdb_ply = pd.DataFrame(pep_ply_list, columns = ["ID", "pep_pdb", "mhc_pdb", "pep_ply"])
    
    df_wt_ply = df_pdb_ply[df_pdb_ply['ID'].str.contains('_wt')].copy()
    df_wt_ply['SampleID'] = df_wt_ply['ID'].replace(['_wt'],[''], regex=True, inplace=False)
    df_wt_ply = df_wt_ply[['SampleID', 'ID', 'pep_ply']]
    df_wt_ply.columns = ['SampleID', 'ID_wt', 'wt_pep_ply']
    df_mut_ply = df_pdb_ply[df_pdb_ply['ID'].str.contains('_mut')].copy()
    df_mut_ply['SampleID'] = df_mut_ply['ID'].replace(['_mut'],[''], regex=True, inplace=False)
    df_mut_ply = df_mut_ply[['SampleID', 'ID', 'pep_ply']]
    df_mut_ply.columns = ['SampleID', 'ID_mut', 'mut_pep_ply']
    
    df_ply = pd.merge(df_wt_ply, df_mut_ply, on='SampleID')
    df_ply.to_csv(output_file, index=False, header=True)

    df_wt_pdb = df_pdb_ply[df_pdb_ply['ID'].str.contains('_wt')].copy()
    df_wt_pdb['SampleID'] = df_wt_pdb['ID'].replace(['_wt'],[''], regex=True, inplace=False)
    df_wt_pdb = df_wt_pdb[['SampleID', 'ID', 'mhc_pdb', 'pep_pdb']]
    df_wt_pdb.columns = ['SampleID', 'ID_wt', 'wt_mhc_pdb', 'wt_pep_pdb']
    df_mut_pdb = df_pdb_ply[df_pdb_ply['ID'].str.contains('_mut')].copy()
    df_mut_pdb['SampleID'] = df_mut_pdb['ID'].replace(['_mut'],[''], regex=True, inplace=False)
    df_mut_pdb = df_mut_pdb[['SampleID', 'ID', 'mhc_pdb', 'pep_pdb']]
    df_mut_pdb.columns = ['SampleID', 'ID_mut', 'mut_mhc_pdb', 'mut_pep_pdb']
    
    df_pdb = pd.merge(df_wt_pdb, df_mut_pdb, on='SampleID')
    pdb_list = df_pdb.values.tolist()

    df_pdb_ply2 = pd.merge(df_ply, df_pdb, on=['SampleID', 'ID_wt', 'ID_mut'])
    pdb_ply_list2 = df_pdb_ply2.values.tolist()
 
    #ply2ImFeat
    try:
        ply2ImFeat(out_ply_dir, out_for_feat_dir, pdb_ply_list2, threads)
    except:
        print("Error in surface.ply2ImFeat")
    
    #atom and dist
    pdb2ImAtom(out_for_atom_dist_dir, pdb_ply_list2)
    
    #foreignness feat filter
    for_feat_f(out_for_feat_dir, out_for_featfilt_dir, df_pdb_ply2, for_patch_pad_size, threads)
    
    #foreignness dist padding
    for_dist_pad(out_for_atom_dist_dir, df_pdb_ply2, dist_pad_size)
    
    #merge cache
    merge_for_cache(out_for_featfilt_dir, out_for_atom_dist_dir, out_for_cache_dir, df_pdb_ply2, for_neigh_num)
