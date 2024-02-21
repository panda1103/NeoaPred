import os
import time
import sys
import pymesh
import numpy as np
from sklearn.neighbors import KDTree
from NeoaPred.masif_tools.triangulation.fixmesh import fix_mesh
from NeoaPred.masif_tools.triangulation.computeMSMS import computeMSMS
from NeoaPred.masif_tools.triangulation.computeHydrophobicity import computeHydrophobicity
from NeoaPred.masif_tools.triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from NeoaPred.masif_tools.triangulation.computeAPBS import computeAPBS
from NeoaPred.masif_tools.triangulation.compute_normal import compute_normal

"""
read_ply.py: Save a ply file to disk using pymesh and load the attributes used by MaSIF. 
Created by Pablo Gainza - LPDI STI EPFL 2019
Modified by Dawei Jiang - AIDuli Lab 2023
Released under an Apache License 2.0
"""

def compute_iface(molecule, regular_mesh, complex_pdb_filename, mesh_res=1.0):
    # Compute the surface of the entire complex and from that compute the interface.
    v3, f3, _, _, _ = computeMSMS(complex_pdb_filename, protonate=True)
    # Regularize the mesh
    mesh = pymesh.form_mesh(v3, f3)
    # Regularize the full mesh. If you ignore it, this can speed up things by a lot.
    start_time = time.time()
    full_regular_mesh = fix_mesh(mesh, mesh_res)
    print("Regularize the full mesh. time:{:2.2f}s".format(time.time()-start_time))
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    #d = np.square(d)
    #Masif: Square d, because this is how it was in the pyflann version.
    #Square d is used in masif code, which is different from the method in masif paper.
    assert(len(d) == len(regular_mesh.vertices))
    iface = np.zeros(len(regular_mesh.vertices))
    if(molecule == 'peptide'):
        iface_v = np.where(d >= 2.0)[0]
        iface[iface_v] = 1.0
        iface_v = np.where((d >= 1.0) & (d < 2.0))[0]
        iface[iface_v] = 0.5
        iface_v = np.where(d < 1.0)[0]
        iface[iface_v] = 0.0
    elif(molecule == 'mhc'):
        iface_v = np.where(d >= 2.0)[0]
        iface[iface_v] = 1.0
    else:
        print("Compute iface error. Unknown molecule name " + molecule )
        os._exit("molecule = peptide or mhc.")
    return iface

def mesh_feat_compute(molecule, chain_pdb_filename, out_filename_prefix, mesh_res=1.0):
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(chain_pdb_filename, protonate=True)
    # Fix the mesh.
    mesh = pymesh.form_mesh(vertices1, faces1)
    # Regularize the full mesh. If you ignore it, this can speed up things by a lot.
    start_time = time.time()
    regular_mesh = fix_mesh(mesh, mesh_res)
    print("Regularize the full mesh. time:{:2.2f}s".format(time.time()-start_time))
    # Compute the normals
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    # Compute charge
    vertex_hbond = computeCharges(out_filename_prefix, vertices1, names1)
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, feature_interpolation=True)
    # Compute hydrophobicity
    vertex_hphobicity = computeHydrophobicity(names1)
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, feature_interpolation=True)
    # Compute APBS
    vertex_charges = computeAPBS(regular_mesh.vertices, chain_pdb_filename, out_filename_prefix)
    # Convert to ply and save.
    save_ply(out_filename_prefix+".ply", regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal, 
             charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)


def save_ply(
    filename,
    vertices,
    faces=[],
    normals=None,
    charges=None,
    vertex_cb=None,
    hbond=None,
    hphob=None,
    iface=None,
    si=None,
    ddc=None,
    dm=None,
    normalize_charges=False,
):
    """ Save vertices, mesh in ply format.
        vertices: coordinates of vertices
        faces: mesh
    """
    mesh = pymesh.form_mesh(vertices, faces)
    if normals is not None:
        n1 = normals[:, 0]
        n2 = normals[:, 1]
        n3 = normals[:, 2]
        mesh.add_attribute("vertex_nx")
        mesh.set_attribute("vertex_nx", n1)
        mesh.add_attribute("vertex_ny")
        mesh.set_attribute("vertex_ny", n2)
        mesh.add_attribute("vertex_nz")
        mesh.set_attribute("vertex_nz", n3)
    if charges is not None:
        mesh.add_attribute("charge")
        if normalize_charges:
            charges = charges / 10
        mesh.set_attribute("charge", charges)
    if hbond is not None:
        mesh.add_attribute("hbond")
        mesh.set_attribute("hbond", hbond)
    if vertex_cb is not None:
        mesh.add_attribute("vertex_cb")
        mesh.set_attribute("vertex_cb", vertex_cb)
    if hphob is not None:
        mesh.add_attribute("vertex_hphob")
        mesh.set_attribute("vertex_hphob", hphob)
    if iface is not None:
        mesh.add_attribute("vertex_iface")
        mesh.set_attribute("vertex_iface", iface)
    if si is not None:
        mesh.add_attribute("si")
        mesh.set_attribute("si", si)
    if ddc is not None:
        mesh.add_attribute("ddc")
        mesh.set_attribute("ddc", ddc)
    if dm is not None:
        mesh.add_attribute("dm")
        mesh.set_attribute("dm", dm)

    pymesh.save_mesh(
        filename, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True
    )

