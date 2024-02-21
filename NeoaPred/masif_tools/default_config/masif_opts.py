import tempfile

masif_opts = {}

masif_opts["tmp_dir"] = tempfile.gettempdir()
# Surface features
masif_opts["use_hbond"] = True
masif_opts["use_hphob"] = True
masif_opts["use_apbs"] = True
masif_opts["compute_iface"] = True
# Mesh resolution. Everything gets very slow if it is lower than 1.0
masif_opts["mesh_res"] = 1.0
masif_opts["feature_interpolation"] = True


# Coords params
masif_opts["radius"] = 6.0

# Neural network patch application specific parameters.
masif_opts["ppi_search"] = {}
masif_opts["ppi_search"]["max_shape_size"] = 80
masif_opts["ppi_search"]["max_distance"] = 6.0  # Radius for the neural network.
masif_opts["ppi_search"]["feat_mask"] = [1.0] * 5
masif_opts["ppi_search"]["max_sc_filt"] = 1.0
masif_opts["ppi_search"]["min_sc_filt"] = 0.5
masif_opts["ppi_search"]["pos_surf_accept_probability"] = 1.0
masif_opts["ppi_search"]["pos_interface_cutoff"] = 1.0 #1.0 to 0.5
masif_opts["ppi_search"]["range_val_samples"] = 0.9  # 0.9 to 1.0
# Parameters for shape complementarity calculations.
masif_opts["ppi_search"]["sc_radius"] = 6.0
masif_opts["ppi_search"]["sc_interaction_cutoff"] = 1.5
masif_opts["ppi_search"]["sc_iface_filter"] = 0.0
masif_opts["ppi_search"]["sc_w"] = 0.25

masif_opts["ppi_search"]["ss_radius"] = 6.0
masif_opts["ppi_search"]["ss_iface_filter"] = 1.0
masif_opts["ppi_search"]["ss_w"] = 0.25

masif_opts["ppi_search"]["si_filter_value"] = 0.5
masif_opts["ppi_search"]["ddc_filter_value"] = 0.35

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
masif_opts["atom_order"] = {atom_type: i for i, atom_type in enumerate(atom_types)}
