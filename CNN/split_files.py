import os
import math
import h5py
import numpy as np

def append_dataset(dset, arr):
    old_size = dset.shape[0]
    new_size = old_size + arr.shape[0]
    dset.resize((new_size,) + dset.shape[1:])
    dset[old_size:new_size] = arr

def split_h5_file(input_h5, output_dir, base_name="train_part",
                  chunk_size=1_000_000, sub_chunk_size=100_000):

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(input_h5, "r") as f_in:
        images = f_in["image"]  # shape (N, C, H, W)
        labels = f_in["label"]  # shape (N, L)
        N = images.shape[0]
        num_parts = math.ceil(N / chunk_size)

        C = images.shape[1]
        H = images.shape[2]
        W = images.shape[3]
        L = labels.shape[1]

        for part_idx in range(num_parts):
            start = part_idx * chunk_size
            end = min((part_idx + 1) * chunk_size, N)
            n_this_chunk = end - start

            out_path = os.path.join(output_dir, f"{base_name}{part_idx}.h5")
            with h5py.File(out_path, "w") as f_out:
                img_dset = f_out.create_dataset(
                    "image",
                    shape=(0, C, H, W),
                    maxshape=(None, C, H, W),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, C, H, W)
                )
                lbl_dset = f_out.create_dataset(
                    "label",
                    shape=(0, L),
                    maxshape=(None, L),
                    dtype=np.int32,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, L)
                )

                local_start = start
                while local_start < end:
                    local_end = min(local_start + sub_chunk_size, end)

                    img_chunk = images[local_start:local_end]
                    lbl_chunk = labels[local_start:local_end]

                    append_dataset(img_dset, img_chunk)
                    append_dataset(lbl_dset, lbl_chunk)

                    local_start = local_end

            print(f"Wrote {out_path} with {n_this_chunk} examples (from {start} to {end}).")


if __name__ == "__main__":
    split_h5_file(
        input_h5="DataCNN/train.h5",
        output_dir="./train_splits",
        base_name="train_part",
        chunk_size=1_000_000,
        sub_chunk_size=100_000
    )
