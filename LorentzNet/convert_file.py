import os, glob, time
import numpy as np
import h5py
import uproot
import jax.numpy as jnp
from jax import jit

@jit
def dot(p1, p2):
    """Lorentz-invariant dot product for two 4-vectors."""
    return p1[0] * p2[0] - jnp.dot(p1[1:], p2[1:])

@jit
def dots(p1s, p2s):
    """Lorentz-invariant dot product for batches of 4-vectors."""
    return jnp.array([dot(p1s[i], p2s[i]) for i in range(p1s.shape[0])])

@jit
def masses(p):
    """Compute invariant mass for a batch of 4-vectors."""
    return jnp.sqrt(jnp.maximum(0., dots(p, p)))

@jit
def pt(momentum):
    """Compute transverse momentum from 4-vector."""
    return jnp.sqrt(jnp.dot(momentum[1:3], momentum[1:3]))


# ================================
# One-hot PID encoding
# ================================

def one_hot_pid(charged, neutral, photon, electron, muon):
    """
    One-hot encode particle type using PID flags.

    Returns:
        np.ndarray: 5-dimensional one-hot vector.
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


# ================================
# ROOT to dictionary conversion
# ================================

def convert_root_to_dict(root_file, fixed_nvectors, add_beams=False, dot_products=False, double_precision=True):
    """
    Convert a ROOT file to structured dictionary with fixed-sized arrays.

    Args:
        root_file (str): Path to ROOT file.
        fixed_nvectors (int): Number of particles to pad/truncate to.
        add_beams (bool): Whether to add beam particles.
        dot_products (bool): Whether to compute pairwise dot products.
        double_precision (bool): Use float64 if True, else float32.

    Returns:
        dict: structured data for the file.
    """
    precision = 'f8' if double_precision else 'f4'
    file = uproot.open(root_file)
    tree = file["tree"]
    branches = [
        "part_energy", "part_px", "part_py", "part_pz",
        "jet_nparticles", "part_charge",
        "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
        "part_isElectron", "part_isMuon",
        "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg",
        "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
        "label_Tbqq", "label_Tbl"
    ]
    data = tree.arrays(branches, library="np")
    nentries = len(data["jet_nparticles"])
    nbeam = 2 if add_beams else 0
    nvectors = fixed_nvectors

    out = {
        "Nobj": np.zeros(nentries, dtype=np.int16),
        "Pmu": np.zeros((nentries, nvectors, 4), dtype=precision),
        "truth_Pmu": np.zeros((nentries, 4), dtype=precision),
        "jet_pt": np.zeros(nentries, dtype=precision),
        "label": np.zeros((nentries, nvectors), dtype=np.int16),
        "mass": np.zeros((nentries, nvectors), dtype=precision),
        "jet_label": np.zeros((nentries, 10), dtype=np.int16),
        "scalars": np.zeros((nentries, nvectors, 7), dtype=precision),
        "atom_mask": np.zeros((nentries, nvectors), dtype=bool)
    }
    if dot_products:
        out["dots"] = np.zeros((nentries, nvectors, nvectors), dtype=precision)

    if add_beams:
        beam_vec = np.array([[np.sqrt(1.), 0, 0, 1.], [np.sqrt(1.), 0, 0, -1.]], dtype=precision)

    for i in range(nentries):
        nobj = int(data["jet_nparticles"][i])
        E, px, py, pz = data["part_energy"][i][:nobj], data["part_px"][i][:nobj], data["part_py"][i][:nobj], data["part_pz"][i][:nobj]
        Pmu_event = np.stack([E, px, py, pz], axis=1)

        out["Pmu"][i, :nobj, :] = Pmu_event
        out["Nobj"][i] = nobj + nbeam
        out["jet_pt"][i] = pt(np.sum(Pmu_event, axis=0))
        out["label"][i, :nobj] = 1
        out["mass"][i, :] = masses(out["Pmu"][i, :, :])
        out["atom_mask"][i, :nobj] = True

        m_vec = masses(Pmu_event)
        charge = data["part_charge"][i][:nobj]
        pid_onehot = np.array([
            one_hot_pid(
                data["part_isChargedHadron"][i][j],
                data["part_isNeutralHadron"][i][j],
                data["part_isPhoton"][i][j],
                data["part_isElectron"][i][j],
                data["part_isMuon"][i][j]
            ) for j in range(nobj)
        ])
        scalars_event = np.concatenate([m_vec[:, None], charge[:, None], pid_onehot], axis=1)
        out["scalars"][i, :nobj, :] = scalars_event

        out["jet_label"][i] = np.array([
            data["label_QCD"][i], data["label_Hbb"][i], data["label_Hcc"][i],
            data["label_Hgg"][i], data["label_H4q"][i], data["label_Hqql"][i],
            data["label_Zqq"][i], data["label_Wqq"][i], data["label_Tbqq"][i],
            data["label_Tbl"][i]
        ], dtype=np.int16)

        if add_beams:
            out["Pmu"][i, -nbeam:, :] = beam_vec
            out["label"][i, -nbeam:] = -1
            out["atom_mask"][i, -nbeam:] = True

        if dot_products:
            out["dots"][i] = dots(out["Pmu"][i], out["Pmu"][i])
    return out


# ================================
# HDF5 handling
# ================================

def create_resizable_h5(output_file, keys, data_shapes, dtypes):
    f = h5py.File(output_file, "w")
    dsets = {
        key: f.create_dataset(
            key, shape=data_shapes[key], maxshape=(None,) + data_shapes[key][1:],
            dtype=dtypes[key], compression="gzip"
        ) for key in keys
    }
    return f, dsets

def append_to_dataset(dset, data):
    current_size = dset.shape[0]
    dset.resize(current_size + data.shape[0], axis=0)
    dset[current_size:] = data


# ================================
# Main conversion function
# ================================

def main():
    root_folder = "../Data"
    output_dir = "./Data"
    os.makedirs(output_dir, exist_ok=True)

    root_files = glob.glob(os.path.join(root_folder, "*.root"))
    if not root_files:
        raise RuntimeError("No ROOT files found.")

    # Determine maximum number of particles across all files
    global_nvectors = max(
        int(np.max(uproot.open(rf)["tree"]["jet_nparticles"].array(library="np")))
        for rf in root_files
    )
    print(f"Max number of particles per jet: {global_nvectors}")

    # Determine smallest event count across all files for balanced splitting
    min_events = min(
        len(uproot.open(rf)["tree"]["jet_nparticles"].array(library="np"))
        for rf in root_files
    )
    print(f"Minimum number of events per file: {min_events}")

    # Create template for dataset shapes and dtypes
    temp = convert_root_to_dict(root_files[0], global_nvectors)
    keys = list(temp.keys())
    shapes = {k: (0,) + temp[k].shape[1:] for k in keys}
    dtypes = {k: temp[k].dtype for k in keys}

    # Create HDF5 files
    train_f, train_dsets = create_resizable_h5(os.path.join(output_dir, "train.h5"), keys, shapes, dtypes)
    val_f, val_dsets     = create_resizable_h5(os.path.join(output_dir, "val.h5"), keys, shapes, dtypes)
    test_f, test_dsets   = create_resizable_h5(os.path.join(output_dir, "test.h5"), keys, shapes, dtypes)

    # Process each file and write split
    for rf in root_files:
        print(f"Processing file: {rf}")
        data = convert_root_to_dict(rf, global_nvectors)
        idx = np.arange(data["Nobj"].shape[0])
        np.random.shuffle(idx)
        idx = np.sort(idx[:min_events])
        n_train, n_val = int(min_events * 0.8), int(min_events * 0.1)
        n_test = min_events - n_train - n_val

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        for k in keys:
            append_to_dataset(train_dsets[k], data[k][train_idx])
            append_to_dataset(val_dsets[k],  data[k][val_idx])
            append_to_dataset(test_dsets[k], data[k][test_idx])

    train_f.close()
    val_f.close()
    test_f.close()
    print("✅ All files converted and saved successfully.")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total processing time: {time.time() - start:.2f} seconds")
