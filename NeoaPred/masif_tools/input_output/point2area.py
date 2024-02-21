import numpy as np
from Bio.PDB import *

def get_area(pdb, aa_positions):
    parser = PDBParser(PERMISSIVE = True, QUIET = True, get_header=True)
    data = parser.get_structure(str(pdb), pdb)
    model = next(data.get_models())
    chains = list(model.get_chains())
    hla = chains[0].get_list()
    coords = []
    for pos in aa_positions:
        aa = hla [pos - 1]
        aa_atom = [atom for atom in aa.get_atoms() if atom.name == 'CA' ][0]
        coords.append(aa_atom.get_coord())
    return np.array(coords)

def define_area(point1, point2, point3):
    """
    法向量    ：n={A,B,C}
    空间上某点：p={x0,y0,z0}
    点法式方程：A(x-x0)+B(y-y0)+C(z-z0)=Ax+By+Cz-(Ax0+By0+Cz0)
    :param point1:
    :param point2:
    :param point3:
    :return: Ax, By, Cz, D (Ax + By + Cz + D = 0)
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    AB = np.asmatrix(point2 - point1)
    AC = np.asmatrix(point3 - point1)
    N = np.cross(AB, AC)  # 向量叉乘，求法向量
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
    return Ax, By, Cz, D
 
 
def point2area_distance(point1, point2, point3, point4):
    """
    :param point1:
    :param point2:
    :param point3:
    :param point4:
    :return: distance of point4 to area (point1, point2, point3)
    """
    Ax, By, Cz, D = define_area(point1, point2, point3)
    mod_d = Ax * point4[0] + By * point4[1] + Cz * point4[2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    #d = abs(mod_d) / mod_area
    d = mod_d / mod_area
    return d
