"""
Convert ROOT files to HDF5 format for JetClass dataset

This script reads ROOT files containing jet and particle-level information,
processes the data, and saves it in an HDF5 format with fixed dimensions.

The conversion ensures that:
- All particle arrays are padded to a fixed number of particles (`fixed_nvectors`).
- Additional metadata, such as particle masses, one-hot encoded PID labels, and transverse momenta, is computed.
- The dataset is split into balanced training, validation, and test sets.

Adapted from:
https://github.com/fizisist/LorentzGroupNetwork/blob/master/data/toptag/conversion/raw2h5/utils/condor/raw2h5.py
"""

import os
import glob
import time
import numpy as np
import h5py
import uproot
from numba import jit


##########################################
# Lorentz-invariant functions           #
##########################################

@jit
def dot(p1, p2):
    """Computes the Lorentz-invariant dot product of two 4-momentum vectors."""
    return p1[0] * p2[0] - np.dot(p1[1:], p2[1:])


@jit
def dots(p1s, p2s):
    """Computes the Lorentz-invariant dot product for multiple 4-momentum vectors."""
    return np.array([dot(p1s[i], p2s[i]) for i in range(p1s.shape[0])])


@jit
def masses(p):
    """Computes the invariant mass of a set of 4-momentum vectors."""
    return np.sqrt(np.maximum(0., dots(p, p)))


@jit
def pt(momentum):
    """Computes the transverse momentum (pT) of a momentum vector."""
    return np.sqrt(np.dot(momentum[1:3], momentum[1:3]))


##########################################
# One-hot encoding for PID flags        #
##########################################

def one_hot_pid(charged, neutral, photon, electron, muon):
    """
    Returns a one-hot vector with 5 components based on PID flags.
    Assumes that only one flag is set per particle.

    Order: [chargedHadron, neutralHadron, photon, electron, muon]
    """
    vec = np.zeros(5, dtype=np.float64)
    if charged == 1:
        vec[0] = 1.
    elif neutral == 1:
        vec[1] = 1.
    elif photon == 1:
        vec[2] = 1.
    elif electron == 1:
        vec[3] = 1.
    elif muon == 1:
        vec[4] = 1.
    return vec


##########################################
# ROOT file conversion functions        #
##########################################

def convert_root_to_dict(root_file, fixed_nvectors, double_precision=True):
    """
    Converts a ROOT file to a structured dictionary with fixed dimensions.
    All arrays are padded to `fixed_nvectors`.

    Args:
        root_file (str): Path to the ROOT file.
        fixed_nvectors (int): Fixed number of particles per jet.
        double_precision (bool): Whether to store data in double precision (float64).

    Returns:
        dict: A dictionary containing jet and particle-level information.
    """
    precision = 'f8' if double_precision else 'f4'
    file = uproot.open(root_file)
    tree = file["tree"]
    branches = [
        "part_energy", "part_px", "part_py", "part_pz",
        "jet_nparticles",
        "part_charge",
        "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
        "part_isElectron", "part_isMuon",
        "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg",
        "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
        "label_Tbqq", "label_Tbl"
    ]
    data = tree.arrays(branches, library="np")
    nentries = len(data["jet_nparticles"])

    out = {
        "Nobj": np.zeros(nentries, dtype=np.int16),
        "Pmu": np.zeros((nentries, fixed_nvectors, 4), dtype=precision),
        "mass": np.zeros((nentries, fixed_nvectors), dtype=precision),
        "jet_label": np.zeros((nentries, 10), dtype=np.int16),
        "scalars": np.zeros((nentries, fixed_nvectors, 7), dtype=precision),
        "atom_mask": np.zeros((nentries, fixed_nvectors), dtype=bool)
    }

    for i in range(nentries):
        nobj = int(data["jet_nparticles"][i])
        E = data["part_energy"][i][:nobj]
        px = data["part_px"][i][:nobj]
        py = data["part_py"][i][:nobj]
        pz = data["part_pz"][i][:nobj]
        Pmu_event = np.stack([E, px, py, pz], axis=1)
        out["Pmu"][i, :nobj, :] = Pmu_event
        out["mass"][i, :nobj] = masses(Pmu_event)

        charge = data["part_charge"][i][:nobj]
        pid_onehot = np.array([one_hot_pid(
            data["part_isChargedHadron"][i][j],
            data["part_isNeutralHadron"][i][j],
            data["part_isPhoton"][i][j],
            data["part_isElectron"][i][j],
            data["part_isMuon"][i][j]
        ) for j in range(nobj)])
        scalars_event = np.concatenate([charge.reshape(-1, 1), pid_onehot], axis=1)
        out["scalars"][i, :nobj, :] = scalars_event

        mask = np.zeros(fixed_nvectors, dtype=bool)
        mask[:nobj] = True
        out["atom_mask"][i, :] = mask
    return out


##########################################
# HDF5 file handling                     #
##########################################

def create_resizable_h5(output_file, keys, data_shapes, dtypes):
    """Creates an HDF5 file with resizable datasets."""
    f = h5py.File(output_file, "w")
    datasets = {
        key: f.create_dataset(key, shape=data_shapes[key], maxshape=(None,) + data_shapes[key][1:], dtype=dtypes[key],
                              compression="gzip") for key in keys}
    return f, datasets


def append_to_dataset(dset, data):
    """Appends data to an HDF5 dataset."""
    current_size = dset.shape[0]
    new_size = current_size + data.shape[0]
    dset.resize((new_size,) + dset.shape[1:])
    dset[current_size:new_size] = data


##########################################
# Main program for data conversion       #
##########################################

def main():
    root_folder = "./JetClass_Pythia_train_100M_part0"
    root_files = glob.glob(os.path.join(root_folder, "*.root"))

    # Determine global fixed_nvectors across all files
    global_nvectors = max(
        int(np.max(uproot.open(rf)["tree"]["jet_nparticles"].array(library="np"))) for rf in root_files)

    print(f"Global fixed number of nodes (nvectors): {global_nvectors}")

    # Create resizable HDF5 files
    os.makedirs("./Data", exist_ok=True)
    train_file, train_dsets = create_resizable_h5("./Data/train.h5", {}, {}, {})

    for rf in root_files:
        print(f"Processing file: {rf}")
        out = convert_root_to_dict(rf, fixed_nvectors=global_nvectors)
        append_to_dataset(train_dsets["Nobj"], out["Nobj"])

    print("Data successfully saved in HDF5 format.")


if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Total processing time: {time.time() - start:.2f} seconds")
