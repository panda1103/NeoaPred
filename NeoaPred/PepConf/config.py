import copy
import importlib
import ml_collections as mlc


def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf

def model_config(name):
    c = copy.deepcopy(config)
    if name == "initial":
        c.loss.experimentally_resolved.weight=0.0
        c.loss.violation.weight=0.0
    elif name == "finetuning":
        c.loss.experimentally_resolved.weight=0.01
        c.loss.violation.weight=1.0
    else:
        pass
    return c


single_dim = mlc.FieldReference(96, field_type=int)
pair_dim = mlc.FieldReference(96, field_type=int)
mhc_crop_size = mlc.FieldReference(180, field_type=int)
pep_crop_size = mlc.FieldReference(16, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
tm_enabled = mlc.FieldReference(False, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
NUM_RES = "num residues placeholder"

config = mlc.ConfigDict(
    {
        "data": {
            "common": {
                "feat": {
                    "aatype": [NUM_RES],
                    "all_atom_mask": [NUM_RES, None],
                    "all_atom_positions": [NUM_RES, None, None],
                    "alt_chi_angles": [NUM_RES, None],
                    "atom14_alt_gt_exists": [NUM_RES, None],
                    "atom14_alt_gt_positions": [NUM_RES, None, None],
                    "atom14_atom_exists": [NUM_RES, None],
                    "atom14_atom_is_ambiguous": [NUM_RES, None],
                    "atom14_gt_exists": [NUM_RES, None],
                    "atom14_gt_positions": [NUM_RES, None, None],
                    "atom37_atom_exists": [NUM_RES, None],
                    "backbone_rigid_mask": [NUM_RES],
                    "backbone_rigid_tensor": [NUM_RES, None, None],
                    "chi_angles_sin_cos": [NUM_RES, None, None],
                    "chi_mask": [NUM_RES, None],
                    "pseudo_beta": [NUM_RES, None],
                    "pseudo_beta_mask": [NUM_RES],
                    "residue_index": [NUM_RES],
                    "residx_atom14_to_atom37": [NUM_RES, None],
                    "residx_atom37_to_atom14": [NUM_RES, None],
                    "resolution": [],
                    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
                    "rigidgroups_group_exists": [NUM_RES, None],
                    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
                    "rigidgroups_gt_exists": [NUM_RES, None],
                    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
                    "seq_length": [],
                    "seq_mask": [NUM_RES],
                },
            },
            "supervised": {
                "supervised_features": [
                    "aatype",
                    "residue_index",
                    "seq_length",
                    "all_atom_mask",
                    "all_atom_positions",
                    "resolution",
                ],
            },
            "predict": {
                "fixed_size": True,
                "crop": True,
                "crop_size": None,
                "supervised": False,
            },
            "train": {
                "fixed_size": True,
                "crop": True,
                "crop_size": None,
                "supervised": True,
            },
            "data_module": {
                "use_small_bfd": False,
                "data_loaders": {
                    "batch_size": 1,
                    "num_workers": 16,
                    "pin_memory": True,
                },
            },
        },
        "model": {
            "max_rel_dist": 32,
            "rel_pos_dim": 96,
            "embedder": {
                "aa_dim": 21,
                "pos_dim": 36,
                "blo_freeze": True,
                "phychem_freeze": True,
                "dropout": 0.1,
            },
            "encoder": {
                "num_layers": 8,
                "embed_dim": 96,
                "num_heads": 2,
                "dim_head": 64,
                "dropout": 0.1
            },
            "decoder": {
                "single_dim": single_dim,
                "pair_dim": pair_dim,
                "mhc_len": mhc_crop_size,
                "pep_len": pep_crop_size,
                "ipa_dim": 32,
                "resnet_dim": 96,
                "no_heads_ipa": 4,
                "dropout_rate": 0.1,
                "no_blocks": 3,
                "no_transition_layers": 1,
                "no_resnet_blocks": 1,
                "no_angles": 7,
                "trans_scale_factor": 10,
                "epsilon": 1e-12,
            },
            "heads": {
                "lddt": {
                    "no_bins": 50,
                    "c_in": single_dim,
                    "c_hidden": 96,
                },
                "distogram": {
                    "pair_dim": pair_dim,
                    "no_bins": aux_distogram_bins,
                },
                "tm": {
                    "pair_dim": pair_dim,
                    "no_bins": aux_distogram_bins,
                    "enabled": tm_enabled,
                },
                "experimentally_resolved": {
                    "single_dim": single_dim,
                    "out_dim": 37,
                },
            },
        },
        "relax": {
            "max_iterations": 0,  # no max
            "tolerance": 2.39,
            "stiffness": 10.0,
            "max_outer_iterations": 20,
            "exclude_residues": [],
        },
        "loss": {
            "distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": aux_distogram_bins,
                "eps": eps,  # 1e-6,
                "weight": 0.3,
            },
            "mp_distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": aux_distogram_bins,
                "eps": eps,  # 1e-6,
                "weight": 0.5,
            },
            "experimentally_resolved": {
                "eps": eps,  # 1e-8,
                "min_resolution": 0.0,
                "max_resolution": 3.0,
                "weight": 0.01,
            },
            "fape": {
                "backbone": {
                    "clamp_distance": 10.0,
                    "loss_unit_distance": 10.0,
                    "weight": 0.5,
                },
                "sidechain": {
                    "clamp_distance": 10.0,
                    "length_scale": 10.0,
                    "weight": 0.5,
                },
                "eps": 1e-4,
                "weight": 1.0,
            },
            "mp_fape": {
                "backbone": {
                    "clamp_distance": 10.0,
                    "loss_unit_distance": 10.0,
                    "weight": 0.5,
                },
                "sidechain": {
                    "clamp_distance": 10.0,
                    "length_scale": 10.0,
                    "weight": 0.5,
                },
                "eps": 1e-4,
                "weight": 9.5,
            },
            "plddt_loss": {
                "min_resolution": 0.0,
                "max_resolution": 3.0,
                "cutoff": 15.0,
                "no_bins": 50,
                "eps": eps,  # 1e-10,
                "weight": 0.01,
            },
            "supervised_chi": {
                "chi_weight": 0.5,
                "angle_norm_weight": 0.01,
                "eps": eps,  # 1e-6,
                "weight": 1.0,
            },
            "violation": {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "eps": eps,  # 1e-6,
                "weight": 1.0,
            },
            "tm": {
                "max_bin": 31,
                "no_bins": 64,
                "min_resolution": 0.0,
                "max_resolution": 3.0,
                "eps": eps,  # 1e-8,
                "weight": 0.,
                "enabled": tm_enabled,
            },
            "eps": eps,
        },
    }
)
