import os
import glob
import logging
import numpy as np
import torch
import h5py
import uproot
from tqdm import tqdm

SOURCE_DIR = "../Data"
DEST_DIR = "./DataCNN"
os.makedirs(DEST_DIR, exist_ok=True)
ROOT_FILES = glob.glob(os.path.join(SOURCE_DIR, "*.root"))

GRID_SIZE = 32
ETA_RANGE = (-0.5, 0.5)
PHI_RANGE = (-0.5, 0.5)
LABEL_COLS = [
    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
    'label_Tbqq', 'label_Tbl'
]
EPS = 1e-6

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def build_jet_image(eta, phi, features, grid_size=GRID_SIZE, eta_range=ETA_RANGE, phi_range=PHI_RANGE):
    image = np.zeros((features.shape[1], grid_size, grid_size), dtype=np.float32)
    eta_bins = np.linspace(*eta_range, grid_size + 1)
    phi_bins = np.linspace(*phi_range, grid_size + 1)
    eta_idx = np.digitize(eta, eta_bins) - 1
    phi_idx = np.digitize(phi, phi_bins) - 1
    mask = (eta_idx >= 0) & (eta_idx < grid_size) & (phi_idx >= 0) & (phi_idx < grid_size)
    for i in np.where(mask)[0]:
        for c in range(features.shape[1]):
            image[c, eta_idx[i], phi_idx[i]] += features[i, c]
    return image

def transform_dataframe(df, eps=1e-6):
    images, labels = [], []

    for i in tqdm(range(len(df)), desc="Jets"):
        px = df["part_px"].iloc[i].to_numpy()
        py = df["part_py"].iloc[i].to_numpy()
        pz = df["part_pz"].iloc[i].to_numpy()
        E  = df["part_energy"].iloc[i].to_numpy()

        eta = df["part_deta"].iloc[i].to_numpy()
        phi = df["part_dphi"].iloc[i].to_numpy()

        charge             = df["part_charge"].iloc[i].to_numpy()
        isElectron         = df["part_isElectron"].iloc[i].to_numpy()
        isMuon             = df["part_isMuon"].iloc[i].to_numpy()
        isChargedHadron    = df["part_isChargedHadron"].iloc[i].to_numpy()
        isNeutralHadron    = df["part_isNeutralHadron"].iloc[i].to_numpy()
        isPhoton           = df["part_isPhoton"].iloc[i].to_numpy()

        pt = np.sqrt(px**2 + py**2 + eps)
        log_pt = np.log(pt + eps)
        log_E = np.log(E + eps)
        deltaR = np.sqrt(eta**2 + phi**2 + eps)

        sum_pt = np.sum(pt)
        sum_E  = np.sum(E)
        log_ptrel = log_pt - np.log(sum_pt + eps)
        log_Erel  = log_E  - np.log(sum_E  + eps)

        # === Sett sammen feature-kanaler ===
        features = np.stack([
            log_pt, log_E, eta, phi, deltaR,
            log_ptrel, log_Erel, charge,
            isElectron, isMuon,
            isChargedHadron, isNeutralHadron, isPhoton
        ], axis=1)

        image = build_jet_image(eta, phi, features)
        label = np.array([df[col].iloc[i] for col in LABEL_COLS], dtype=np.int32)

        images.append(image)
        labels.append(label)

    return np.stack(images), np.stack(labels)

def create_h5_file(path, image_shape, label_shape):
    f = h5py.File(path, "w")
    image_dset = f.create_dataset("image", shape=(0,) + image_shape, maxshape=(None,) + image_shape,
                                  chunks=True, compression="gzip", compression_opts=4)
    label_dset = f.create_dataset("label", shape=(0,) + label_shape, maxshape=(None,) + label_shape,
                                  chunks=True, compression="gzip", compression_opts=4)
    return f, image_dset, label_dset

def append(dset, arr):
    cur = dset.shape[0]
    new = cur + arr.shape[0]
    dset.resize((new,) + dset.shape[1:])  # korrekt rank
    dset[cur:new] = arr


# === Bestem felles split-størrelse over alle filer ===
train_counts, test_counts, val_counts = [], [], []

for file in ROOT_FILES:
    try:
        df = uproot.open(file)["tree"].arrays(library="pd")
        n = len(df)
        n_train = int(n * 0.8)
        n_test  = int(n * 0.1)
        n_val   = n - n_train - n_test
        train_counts.append(n_train)
        test_counts.append(n_test)
        val_counts.append(n_val)
    except Exception as e:
        logging.error(f"Failed to read {file}: {e}")

if not train_counts:
    raise RuntimeError("No usable ROOT files found.")

common_train = min(train_counts)
common_test = min(test_counts)
common_val = min(val_counts)
n_total = common_train + common_test + common_val

sample_df = uproot.open(ROOT_FILES[0])["tree"].arrays(library="pd").iloc[:1]
sample_img, sample_lbl = transform_dataframe(sample_df)
img_shape = sample_img.shape[1:]
lbl_shape = sample_lbl.shape[1:]

train_f, train_img, train_lbl = create_h5_file(os.path.join(DEST_DIR, "train.h5"), img_shape, lbl_shape)
test_f,  test_img,  test_lbl  = create_h5_file(os.path.join(DEST_DIR, "test.h5"),  img_shape, lbl_shape)
val_f,   val_img,   val_lbl   = create_h5_file(os.path.join(DEST_DIR, "val.h5"),   img_shape, lbl_shape)

for file in tqdm(ROOT_FILES, desc="Processing files"):
    try:
        df = uproot.open(file)["tree"].arrays(library="pd")
        logging.info(f"✅ Ferdig: {os.path.basename(file)} ({n_total} jets)")
        if len(df) < n_total:
            logging.warning(f"Skipping {file}: too few jets.")
            continue

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = df.iloc[:common_train]
        test_df  = df.iloc[common_train:common_train + common_test]
        val_df   = df.iloc[common_train + common_test:n_total]

        for sub_df, img_dset, lbl_dset in [
            (train_df, train_img, train_lbl),
            (test_df,  test_img,  test_lbl),
            (val_df,   val_img,   val_lbl)
        ]:
            images, labels = transform_dataframe(sub_df)
            append(img_dset, images)
            append(lbl_dset, labels)

        logging.info(f"✅ Processed {os.path.basename(file)} with {n_total} jets.")
    except Exception as e:
        logging.error(f"❌ Failed to process {file}: {e}")

train_f.close()
test_f.close()
val_f.close()
logging.info("✅ Jet image conversion complete.")
