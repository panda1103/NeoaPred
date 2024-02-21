#Predict Peptide conformation

import numpy as np
import pandas as pd
import os
import torch
import torch.utils.data as Data
import sys
import NeoaPred.PepConf.config as config
from .data.data_modules import DataSet_Predict, parse_pdb
from .model.model import MPSP
from .utils.loss import Loss, MP_Loss, compute_plddt
from .config import model_config
from .data.protein import from_prediction, to_pdb, Protein
from .utils.tensor_utils import tensor_tree_map
from .utils.script_utils import relax_protein

def cat_mhc_pep_pdb(mhc, pep):
    mhc_pep = Protein(
        aatype = np.concatenate((mhc.aatype, pep.aatype), axis=0),
        atom_positions = np.concatenate((mhc.atom_positions, pep.atom_positions), axis=0),
        atom_mask = np.concatenate((mhc.atom_mask, pep.atom_mask), axis=0),
        residue_index = np.concatenate((mhc.residue_index, pep.residue_index), axis=0),
        b_factors = np.concatenate((mhc.b_factors, pep.b_factors), axis=0),
        chain_index = np.concatenate((mhc.chain_index, (pep.chain_index+1)), axis=0),
    )
    return mhc_pep

def save_pdb(config, device, outputs, feats, ids, out_path, mhc_files):
    if("sm" in outputs.keys()):
        del outputs["sm"]
    if("violation" in outputs.keys()):
        del outputs["violation"]
    for i in range(len(ids)):
        ID = ids[i]
        mhc_file = mhc_files[i]
        chain_id = "A"
        s_feats = tensor_tree_map(lambda x: np.array(x[i,...].cpu()), feats)
        s_outputs = tensor_tree_map(lambda x: x[i,...].detach().numpy(), outputs)
        pep_unrelaxed_protein = from_prediction(features=s_feats, result=s_outputs)
        pep_unrelaxed_protein_out = out_path + "/"+ str(ID) + ".pdb"
        with open(pep_unrelaxed_protein_out, 'w') as fp:
            fp.write(to_pdb(pep_unrelaxed_protein))
        pep_protein = parse_pdb(pep_unrelaxed_protein_out, ID, chain_id=chain_id)
        mhc_protein = parse_pdb(mhc_file, "MHC", chain_id=chain_id)
        unrelaxed_protein = cat_mhc_pep_pdb(mhc_protein, pep_protein)
        
        try:
            relax_protein(config, str(device), unrelaxed_protein, out_path, ID, False)
            relax_protein_complex = out_path + "/" + str(ID) + "_relaxed.pdb"
            pep_protein = parse_pdb(relax_protein_complex, ID, chain_id="B")
            relax_protein_pep = out_path + "/" + str(ID) + "_relaxed_pep.pdb"
            with open(relax_protein_pep, 'w') as fp:
                fp.write(to_pdb(pep_protein))
        except:
            print("Relax Error: "+ID)

def evaluate(model, device, batch_size, eval_data, config, out_file, out_path, MHC_inDir, Pep_inDir):
    model = model.to(device)
    model = model.eval()
    eval_data = DataSet_Predict(eval_data, mode="train", MHC_inDir=MHC_inDir, Pep_inDir=Pep_inDir)
    eval_loader = Data.DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    id_list = []
    pLDDTs_list = []
    for batch_i, (ids, mhc_feats, pep_feats, feats, mhc_files) in enumerate(eval_loader):
        id_list.extend(ids)
        p_outputs, mp_outputs = model(mhc_feats, pep_feats, feats)
        #plddt
        for i in range(p_outputs['lddt_logits'].shape[0]):
            pred_lddt_logits = p_outputs['lddt_logits'][i]
            #out_lddt = out_path + "/lddt_"+ str(ids[i]) + ".pt"
            #torch.save(pred_lddt_logits, out_lddt)
            seq_mask = pep_feats['seq_mask'][i]
            plddt = compute_plddt(pred_lddt_logits).detach().numpy()
            plddt = [plddt[i] for i in range(len(seq_mask)) if(seq_mask[i])]
            plddt = np.mean(plddt)
            pLDDTs_list.append(plddt)
        #pdb
        save_pdb(config, device, p_outputs, pep_feats, ids, out_path, mhc_files)
    df = pd.DataFrame({'ID':id_list, 'PLDDT':pLDDTs_list})
    df.to_csv(out_file, header=True, index=False)

def structure_predicter(
                input_dir,
                input_file, 
                output_dir,
                output_file,
                pre_train_model,
                MHC_inDir,
                device = "cpu",
                batch_size = 1,
             ):
    name="initial"
    device = torch.device(device)
    Model = MPSP(config, name, device)
    Model.load_state_dict(torch.load(pre_train_model, map_location=device))
    try:
        evaluate(Model, device, batch_size, input_file, config.model_config(name), output_file, output_dir, MHC_inDir, input_dir)
    except:
        print("Error in structure_predicter.evaluate")
