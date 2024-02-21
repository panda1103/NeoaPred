import pymesh
import numpy as np
import pandas as pd
import os
import time
from NeoaPred.masif_tools.input_output.save_ply import save_ply
from NeoaPred.masif_tools.input_output.point2area import get_area, point2area_distance
from NeoaPred.masif_tools.read_data_from_surface import read_data_from_surface, compute_shape_complementarity, compute_shape_similarity

def save_select_ply(mesh, idx, out_ply):
    mesh = mesh
    select_idx = idx
    select_idx_dict = {}
    for i in range(mesh.faces.shape[0]):
        face = mesh.faces[i]
        test_face = all([x in select_idx for x in face])
        if(test_face):
            for j in face:
                select_idx_dict[j] = ''
    select_idx = np.array(list(select_idx_dict.keys()))
    faces = []
    for i in range(mesh.faces.shape[0]):
        face = mesh.faces[i]
        test_face = all([x in select_idx for x in face])
        if(test_face):
            select_face = [np.where(select_idx==x)[0][0] for x in face]
            faces.append(select_face)
    if(select_idx.shape[0] > 0):
        vertices = mesh.vertices[select_idx]
        faces = np.array(faces)
        nx = mesh.get_attribute("vertex_nx")
        ny = mesh.get_attribute("vertex_ny")
        nz = mesh.get_attribute("vertex_nz")
        normals = np.stack([nx,ny,nz], axis=1)[select_idx]
        si = mesh.get_attribute('vertex_si')[select_idx]
        ddc = mesh.get_attribute('vertex_ddc')[select_idx]
        charges = mesh.get_attribute('vertex_charge')[select_idx]
        hbonds = mesh.get_attribute('vertex_hbond')[select_idx]
        hphobs = mesh.get_attribute('vertex_hphob')[select_idx]
        ifaces = mesh.get_attribute('vertex_iface')[select_idx]
        dm = mesh.get_attribute('vertex_dm')[select_idx]
        save_ply(out_ply, vertices, faces, normals=normals, charges=charges, normalize_charges=False, hbond=hbonds, hphob=hphobs, iface=ifaces, si=si, ddc=ddc, dm=dm)


def save_ply_si_ddc_dm(mesh, new_ply, feats, pdb):
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = mesh.vertices
    #faces = np.array(faces)
    nx = mesh.get_attribute("vertex_nx")
    ny = mesh.get_attribute("vertex_ny")
    nz = mesh.get_attribute("vertex_nz")
    normals = np.stack([nx,ny,nz], axis=1)
    charges = mesh.get_attribute('vertex_charge')
    hbonds = mesh.get_attribute('vertex_hbond')
    hphobs = mesh.get_attribute('vertex_hphob')
    ifaces = mesh.get_attribute('vertex_iface')
    si = feats[:,0:6,0].mean(axis=1)
    ddc = feats[:,0:6,1].mean(axis=1)
    dm = []
    aa_positions = [143, 72, 159]
    point3_coords = get_area(pdb, aa_positions)
    for v in vertices:
        d = point2area_distance(point3_coords[0], point3_coords[1],  point3_coords[2], v)
        dm.append(d)
    dm = np.array(dm)
    save_ply(new_ply, vertices, faces, normals=normals, charges=charges, normalize_charges=False, hbond=hbonds, hphob=hphobs, iface=ifaces, si=si, ddc=ddc, dm=dm)

def save_patch_feat_sc(out_filename_prefix, pid, sc_labels, rho, theta, input_feat, mask, neigh_indices, iface_labels, verts):
    np.save(out_filename_prefix+pid+'_sc_labels', sc_labels)
    np.save(out_filename_prefix+pid+'_rho_wrt_center', rho)
    np.save(out_filename_prefix+pid+'_theta_wrt_center', theta)
    np.save(out_filename_prefix+pid+'_input_feat', input_feat)
    np.save(out_filename_prefix+pid+'_mask', mask)
    np.save(out_filename_prefix+pid+'_list_indices', neigh_indices)
    np.save(out_filename_prefix+pid+'_iface_labels', iface_labels)
    np.save(out_filename_prefix+pid+'_X.npy', verts[:,0])
    np.save(out_filename_prefix+pid+'_Y.npy', verts[:,1])
    np.save(out_filename_prefix+pid+'_Z.npy', verts[:,2])

def sc_patch_feat_compute(out_prefix, in_pep, in_mhc, params):
    #compute patch feat
    start_time = time.time()
    
    mesh1, input_feat1, rho1, theta1, mask1, neigh_indices1, iface_labels1, verts1 = read_data_from_surface(in_pep, params)
    mesh2, input_feat2, rho2, theta2, mask2, neigh_indices2, iface_labels2, verts2 = read_data_from_surface(in_mhc, params)
    #compute shape complementarity
    p1_sc_labels, p2_sc_labels = compute_shape_complementarity(in_pep, in_mhc, neigh_indices1, neigh_indices2, rho1, rho2, mask1, mask2, params)
    print("compute shape complementarity patch feat. time:{:2.2f}s".format(time.time()-start_time))

    #save
    save_patch_feat_sc(out_prefix, 'pep', p1_sc_labels, rho1, theta1, input_feat1, mask1, neigh_indices1, iface_labels1, verts1)
    save_patch_feat_sc(out_prefix, 'mhc', p2_sc_labels, rho2, theta2, input_feat2, mask2, neigh_indices2, iface_labels2, verts2)

def save_patch_feat_ss(out_filename_prefix, pid, ss_labels, rho, theta, input_feat, mask, neigh_indices, verts):
    np.save(out_filename_prefix+pid+'_ss_labels', ss_labels)
    np.save(out_filename_prefix+pid+'_rho_wrt_center', rho)
    np.save(out_filename_prefix+pid+'_theta_wrt_center', theta)
    np.save(out_filename_prefix+pid+'_input_feat', input_feat)
    np.save(out_filename_prefix+pid+'_mask', mask)
    np.save(out_filename_prefix+pid+'_list_indices', neigh_indices)
    np.save(out_filename_prefix+pid+'_X.npy', verts[:,0])
    np.save(out_filename_prefix+pid+'_Y.npy', verts[:,1])
    np.save(out_filename_prefix+pid+'_Z.npy', verts[:,2])

def ss_patch_feat_compute(out_prefix, in_wt, in_mut, mhc_pdb_wt, mhc_pdb_mut, out_wt, out_mut, params):
    #compute patch feat
    start_time = time.time()
    mesh1, input_feat1, rho1, theta1, mask1, neigh_indices1, iface_labels1, verts1 = read_data_from_surface(in_wt, params)
    mesh2, input_feat2, rho2, theta2, mask2, neigh_indices2, iface_labels2, verts2 = read_data_from_surface(in_mut, params)
    #compute shape similarity
    mut_ss_labels, wt_ss_labels = compute_shape_similarity(in_mut, in_wt, neigh_indices2, neigh_indices1, rho2, rho1, mask2, mask1, params)
    save_patch_feat_ss(out_prefix, 'wt', wt_ss_labels, rho1, theta1, input_feat1, mask1, neigh_indices1, verts1)
    save_patch_feat_ss(out_prefix, 'mut', mut_ss_labels, rho2, theta2, input_feat2, mask2, neigh_indices2, verts2)
    save_ply_si_ddc_dm(mesh1, out_wt, input_feat1, mhc_pdb_wt)
    save_ply_si_ddc_dm(mesh2, out_mut, input_feat2, mhc_pdb_mut)

def mesh_filter_iface(ply, new_ply, filter_value):
    mesh = pymesh.load_mesh(ply)
    select_idx = np.where(mesh.get_attribute('vertex_iface')!=filter_value)[0]
    #check select faces, some select_idx lost its faces
    select_idx_dict = {}
    for i in range(mesh.faces.shape[0]):
        face = mesh.faces[i]
        test_face = all([x in select_idx for x in face])
        if(test_face):
            for j in face:
                select_idx_dict[j] = ''
    select_idx = np.array(list(select_idx_dict.keys()))
    faces = []
    for i in range(mesh.faces.shape[0]):
        face = mesh.faces[i]
        test_face = all([x in select_idx for x in face])
        if(test_face):
            select_face = [np.where(select_idx==x)[0][0] for x in face]
            faces.append(select_face)
    vertices = mesh.vertices[select_idx]
    faces = np.array(faces)
    nx = mesh.get_attribute("vertex_nx")
    ny = mesh.get_attribute("vertex_ny")
    nz = mesh.get_attribute("vertex_nz")
    normals = np.stack([nx,ny,nz], axis=1)[select_idx]
    charges = mesh.get_attribute('vertex_charge')[select_idx]
    hbonds = mesh.get_attribute('vertex_hbond')[select_idx]
    hphobs = mesh.get_attribute('vertex_hphob')[select_idx]
    ifaces = mesh.get_attribute('vertex_iface')[select_idx]
    save_ply(new_ply, vertices, faces, normals=normals, charges=charges, normalize_charges=False, hbond=hbonds, hphob=hphobs, iface=ifaces)
    return select_idx
