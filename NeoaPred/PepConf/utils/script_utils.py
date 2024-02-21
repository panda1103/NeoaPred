import json
import logging
import os
import re
import time
import numpy
import torch
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict
)

from NeoaPred.PepConf.data import residue_constants, protein
from NeoaPred.PepConf.np.relax import relax

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

def update_timings(timing_dict, output_file=os.path.join(os.getcwd(), "timings.json")):
    """
    Write dictionary of one or more run step times to a file
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                timings = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Overwriting non-standard JSON in {output_file}.")
                timings = {}
    else:
        timings = {}
    timings.update(timing_dict)
    with open(output_file, "w") as f:
        json.dump(timings, f)
    return output_file


def relax_protein(config, model_device, unrelaxed_protein, output_directory, output_name, cif_output):
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"),
        **config.relax,
    )

    t = time.perf_counter()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if "cuda" in model_device:
        device_no = model_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
    # the struct_str will contain either a PDB-format or a ModelCIF format string
    struct_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein, cif_output=cif_output)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    relaxation_time = time.perf_counter() - t

    logger.info(f"Relaxation time: {relaxation_time}")
    update_timings({"relaxation": relaxation_time}, os.path.join(output_directory, "timings.json"))

    # Save the relaxed PDB.
    suffix = "_relaxed.pdb"
    if cif_output:
        suffix = "_relaxed.cif"
    relaxed_output_path = os.path.join(
        output_directory, f'{output_name}{suffix}'
    )
    with open(relaxed_output_path, 'w') as fp:
        fp.write(struct_str)
    fp.close()
    '''
    If delete this line "fp.close()", 
    OSError "Too many open files" may occurred 
    when iterate the module by Dataloader.
    '''
    logger.info(f"Relaxed output written to {relaxed_output_path}...")
