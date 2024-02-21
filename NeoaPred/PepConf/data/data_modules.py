import numpy as np
import pandas as pd
import os
import sys

import torch
import torch.utils.data as Data
from typing import Mapping, Dict, Sequence
from NeoaPred.PepConf.config import model_config
from NeoaPred.PepConf.data import mmcif_parsing, feature_pipeline, residue_constants, protein

FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]

def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
    features: Sequence[str],
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.
    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.
    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    tensor_dict = {
        k: torch.tensor(v) for k, v in np_example.items() if k in features
    }
    return tensor_dict


def make_sequence_features(
    sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features

def make_mmcif_features(
    mmcif_object: mmcif_parsing.MmcifObject, chain_id: str
) -> FeatureDict:
    input_sequence = mmcif_object.chain_to_seqres[chain_id]
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)
    mmcif_feats = {}
    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )
    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, chain_id=chain_id
    )
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask
    mmcif_feats["resolution"] = np.array(
        [mmcif_object.header["resolution"]], dtype=np.float32
    )
    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=np.object_
    )
    mmcif_feats["is_distillation"] = np.array(0., dtype=np.float32)

    return mmcif_feats

def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]]
        for i in range(len(aatype))
    ])

def make_pdb_features(
    protein_object: protein.Protein, chain_id: str, file_id
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    description = "_".join([file_id, chain_id])
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(0.).astype(np.float32)

    return pdb_feats

def parse_mmcif(path, file_id, chain_id):
    with open(path, 'r') as f:
        mmcif_string = f.read()
    mmcif_object = mmcif_parsing.parse(
        file_id=file_id, mmcif_string=mmcif_string
    )
    mmcif_object = mmcif_object.mmcif_object
    return mmcif_object

def parse_pdb(path, file_id, chain_id):
    with open(path, 'r') as f:
        pdb_str = f.read()
    protein_object = protein.from_pdb_string(pdb_str, chain_id)
    return protein_object

def cif_to_feat(path, ID, chain_id, feature_pipeline, mode, crop_size):
    mmcif = parse_mmcif(path=path, file_id=ID, chain_id=chain_id)
    data = make_mmcif_features(mmcif, chain_id=chain_id)
    config = model_config(name=None)
    feature_pipeline = feature_pipeline.FeaturePipeline(config.data)
    feats = feature_pipeline.process_features(data, mode=mode, crop_size=crop_size)
    return feats

def pdb_to_feat(path, ID, chain_id, feature_pipeline, mode='train', crop_size=16):
    pdb = parse_pdb(path=path, file_id=ID, chain_id=chain_id)
    data = make_pdb_features(pdb, chain_id=chain_id, file_id=ID)
    config = model_config(name=None)
    feature_pipeline = feature_pipeline.FeaturePipeline(config.data)
    feats = feature_pipeline.process_features(data, mode=mode, crop_size=crop_size)
    return feats

def cat_mhc_pep(mhc_feat, pep_feat):
    feat = {}
    m_len = mhc_feat["aatype"].shape[0]
    p_len = pep_feat["aatype"].shape[0]
    for k in pep_feat.keys():
        if(pep_feat[k].dim()==0):
            if(k=="seq_length"):
                feat[k] = mhc_feat[k] + pep_feat[k]
            else:
                feat[k] = pep_feat[k]
        elif((mhc_feat[k].shape[0]==m_len)&(pep_feat[k].shape[0]==p_len)):
            feat[k] = torch.cat((mhc_feat[k], pep_feat[k]), dim=0)
        else:
            feat[k] = pep_feat[k]
    return feat

class DataSet(Data.Dataset):
    def __init__(self,
                 dataList,
                 mode,
                 mhc_crop_size = 180,
                 pep_crop_size = 16,
                 inDir = None,
    ):
        super(DataSet).__init__()
        self.dataList = dataList
        if(inDir):
            self.inDir = inDir
        else:
            current_path = os.path.abspath(os.path.dirname(__file__))
            self.inDir = current_path + "/MHC_pep_pdb/"
        self.mode = mode
        self.mhc_crop_size = mhc_crop_size
        self.pep_crop_size = pep_crop_size
        self.feature_pipeline = feature_pipeline
        self.ids, self.mhcs, self.peps, self.mhc_peps = self.getDataset()

    def __len__(self):
        return len(self.ids)

    def getDataset(self):
        ids = []
        mhcs = []
        peps = []
        mhc_peps = []
        df = pd.read_table(self.dataList, header=0)
        ID_list = df["ID"].values
        for ID in ID_list:
            path_pep_init = self.inDir + "/pep_" + str(ID) + ".pdb"
            path_mhc_init = self.inDir + "/mhc_" + str(ID) + ".pdb"
            if os.path.exists(path_pep_init):
                ids.append(ID)
                pep_feat = pdb_to_feat(path=path_pep_init, ID=ID, chain_id="A", feature_pipeline=self.feature_pipeline, mode=self.mode, crop_size=self.pep_crop_size)
                peps.append(pep_feat)
                mhc_feat = pdb_to_feat(path=path_mhc_init, ID=ID, chain_id="A", feature_pipeline=self.feature_pipeline, mode=self.mode, crop_size=self.mhc_crop_size)
                mhcs.append(mhc_feat)
                feat = cat_mhc_pep(mhc_feat, pep_feat)
                mhc_peps.append(feat)
        return ids, mhcs, peps, mhc_peps 

    def __getitem__(self, idx):
        return self.ids[idx], self.mhcs[idx], self.peps[idx], self.mhc_peps[idx]

class DataSet_Relax(Data.Dataset):
    def __init__(self,
                 dataList,
                 mode,
                 mhc_crop_size = 180,
                 pep_crop_size = 16,
                 inDir = None,
    ):
        super(DataSet).__init__()
        self.dataList = dataList
        if(inDir):
            self.inDir = inDir
        else:
            current_path = os.path.abspath(os.path.dirname(__file__))
            self.inDir = current_path + "../all_eval_out/"
        self.mode = mode
        self.mhc_crop_size = mhc_crop_size
        self.pep_crop_size = pep_crop_size
        self.feature_pipeline = feature_pipeline
        self.ids, self.mhcs, self.peps, self.mhc_peps = self.getDataset()

    def __len__(self):
        return len(self.ids)

    def getDataset(self):
        ids = []
        mhcs = []
        peps = []
        mhc_peps = []
        df = pd.read_table(self.dataList, header=0)
        ID_list = df["ID"].values
        for ID in ID_list:
            path_init = self.inDir + "/" + str(ID) + "_relaxed.pdb"
            if os.path.exists(path_init):
                ids.append(ID)
                pep_feat = pdb_to_feat(path=path_init, ID=ID, chain_id="B", feature_pipeline=self.feature_pipeline, mode=self.mode, crop_size=self.pep_crop_size)
                peps.append(pep_feat)
                mhc_feat = pdb_to_feat(path=path_init, ID=ID, chain_id="A", feature_pipeline=self.feature_pipeline, mode=self.mode, crop_size=self.mhc_crop_size)
                mhcs.append(mhc_feat)
                feat = cat_mhc_pep(mhc_feat, pep_feat)
                mhc_peps.append(feat)
        return ids, mhcs, peps, mhc_peps 

    def __getitem__(self, idx):
        return self.ids[idx], self.mhcs[idx], self.peps[idx], self.mhc_peps[idx]

class DataSet_eval(Data.Dataset):
    def __init__(self,
                 dataList,
                 mode,
                 mhc_crop_size = 180,
                 pep_crop_size = 16,
    ):
        super(DataSet).__init__()
        self.dataList = dataList
        current_path = os.path.abspath(os.path.dirname(__file__))
        self.inDir = current_path + "/MHC_pep_pdb/"
        self.mode = mode
        self.mhc_crop_size = mhc_crop_size
        self.pep_crop_size = pep_crop_size
        self.feature_pipeline = feature_pipeline
        self.ids, self.mhcs, self.peps, self.mhc_peps, self.mhc_files = self.getDataset()

    def __len__(self):
        return len(self.ids)

    def getDataset(self):
        ids = []
        mhcs = []
        peps = []
        mhc_peps = []
        mhc_files = []
        df = pd.read_table(self.dataList, header=0)
        ID_list = df["ID"].values
        for ID in ID_list:
            path_pep_init = self.inDir + "/pep_" + str(ID) + ".pdb"
            path_mhc_init = self.inDir + "/mhc_" + str(ID) + ".pdb"
            if os.path.exists(path_pep_init):
                ids.append(ID)
                pep_feat = pdb_to_feat(path=path_pep_init, ID=ID, chain_id="A", feature_pipeline=self.feature_pipeline, mode=self.mode, crop_size=self.pep_crop_size)
                peps.append(pep_feat)
                mhc_feat = pdb_to_feat(path=path_mhc_init, ID=ID, chain_id="A", feature_pipeline=self.feature_pipeline, mode=self.mode, crop_size=self.mhc_crop_size)
                mhcs.append(mhc_feat)
                feat = cat_mhc_pep(mhc_feat, pep_feat)
                mhc_peps.append(feat)
                mhc_files.append(path_mhc_init)
        return ids, mhcs, peps, mhc_peps, mhc_files

    def __getitem__(self, idx):
        return self.ids[idx], self.mhcs[idx], self.peps[idx], self.mhc_peps[idx], self.mhc_files[idx]

class DataSet_Predict(Data.Dataset):
    def __init__(self,
                 dataList,
                 mode,
                 MHC_inDir=None,
                 Pep_inDir=None,
                 mhc_crop_size = 180,
                 pep_crop_size = 16,
    ):
        super(DataSet).__init__()
        self.dataList = dataList
        self.mode = mode
        current_path = os.path.abspath(os.path.dirname(__file__))
        if (not MHC_inDir):
            MHC_inDir = current_path + "/MHC_template_PDB/"
        if (not Pep_inDir):
            Pep_inDir = current_path + "/Peptide/"
        self.MHC_inDir = MHC_inDir
        self.Pep_inDir = Pep_inDir
        self.mhc_crop_size = mhc_crop_size
        self.pep_crop_size = pep_crop_size
        self.feature_pipeline = feature_pipeline
        self.ids, self.mhcs, self.peps, self.mhc_peps, self.mhc_files = self.getDataset()

    def __len__(self):
        return len(self.ids)

    def getDataset(self):
        ids = []
        mhcs = []
        peps = []
        mhc_peps = []
        mhc_files = []
        df = pd.read_csv(self.dataList, header=0)
        ID_list = df["ID"].values
        pep_list = df["Peptide_Seq"].values
        hla_list = df["Allele"].values
        for idx in range(len(ID_list)):
            ID = ID_list[idx]
            fa_seq = pep_list[idx]
            path_mhc_init = self.MHC_inDir + "/HLA_" + hla_list[idx] + ".pdb"
            path_pep_init = self.Pep_inDir + "/" + ID + ".pdb"
            if os.path.exists(path_mhc_init):
                ids.append(ID)
                pep_feat = pdb_to_feat(path=path_pep_init, ID=ID, chain_id="A", feature_pipeline=self.feature_pipeline, mode=self.mode, crop_size=self.pep_crop_size)
                peps.append(pep_feat)
                mhc_feat = pdb_to_feat(path=path_mhc_init, ID=hla_list[idx], chain_id="A", feature_pipeline=self.feature_pipeline, mode=self.mode, crop_size=self.mhc_crop_size)
                mhcs.append(mhc_feat)
                feat = cat_mhc_pep(mhc_feat, pep_feat)
                mhc_peps.append(feat)
                mhc_files.append(path_mhc_init)
        return ids, mhcs, peps, mhc_peps, mhc_files

    def __getitem__(self, idx):
        return self.ids[idx], self.mhcs[idx], self.peps[idx], self.mhc_peps[idx], self.mhc_files[idx]
