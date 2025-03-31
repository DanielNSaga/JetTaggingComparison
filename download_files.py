"""
Download and Extract JetClass Pythia Train Dataset (part0)

This script downloads the "JetClass_Pythia_train_100M_part0.tar" archive from Zenodo,
validates its MD5 checksum, and extracts its contents directly into the project's "Data" folder.

The archive contains a folder with ROOT files.

Adapted from: https://github.com/jet-universe/particle_transformer/blob/main/get_datasets.py

Usage:
    python download_files.py
"""

import argparse
import os
import tarfile
import hashlib
import requests
from tqdm import tqdm


def download_file(url, dest_path, chunk_size=1024):
    """
    Download a file from a given URL with a progress bar.

    Args:
        url (str): The URL to download the file from.
        dest_path (str): The destination path where the file will be saved.
        chunk_size (int): Number of bytes per chunk.

    Returns:
        str: The destination path.
    """
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, desc=os.path.basename(dest_path)
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return dest_path


def validate_file(file_path, expected_hash, hash_alg='md5', chunk_size=8192):
    """
    Validate the file at file_path by comparing its hash with the expected hash.

    Args:
        file_path (str): Path to the file.
        expected_hash (str): The expected hash string.
        hash_alg (str): Hash algorithm ('md5' or 'sha256').
        chunk_size (int): Number of bytes to read at a time.

    Returns:
        bool: True if the file hash matches the expected hash, else False.
    """
    if hash_alg.lower() == 'md5':
        hasher = hashlib.md5()
    elif hash_alg.lower() == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError("Unsupported hash algorithm. Use 'md5' or 'sha256'.")

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_hash


def extract_archive(archive_path, extract_to):
    """
    Extract a tar archive to a specified directory.

    Args:
        archive_path (str): Path to the tar archive.
        extract_to (str): Directory where the archive will be extracted.

    Raises:
        ValueError: If the file is not a valid tar archive.
    """
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r') as tar:
            tar.extractall(path=extract_to)
    else:
        raise ValueError("Unsupported archive format. Only tar archives are supported.")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract JetClass Pythia Train Dataset (part0) into the project's Data folder."
    )
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists.")
    args = parser.parse_args()

    # URL and expected MD5 for part0
    url = "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part0.tar"
    expected_hash = "de4fd2dca2e68ab3c85d5cfd3bcc65c3"

    # Destination: the "Data" folder in the project
    project_dir = os.getcwd()  # Assumes the script is run from the project root
    data_dir = os.path.join(project_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "JetClass_Pythia_train_100M_part0.tar")

    # Download file if necessary
    if os.path.exists(tar_path) and not args.force:
        print(f"File already exists at {tar_path}. Validating hash...")
        if validate_file(tar_path, expected_hash, hash_alg='md5'):
            print("Hash validation passed. Skipping download.")
        else:
            print("Hash validation failed. Re-downloading the dataset.")
            os.remove(tar_path)
            download_file(url, tar_path)
    else:
        download_file(url, tar_path)

    # Validate downloaded file
    print("Validating downloaded file...")
    if not validate_file(tar_path, expected_hash, hash_alg='md5'):
        raise RuntimeError("File hash does not match expected value. The download may be corrupted.")

    # Extract archive into the Data folder
    print("Extracting archive into the Data folder...")
    extract_archive(tar_path, data_dir)
    print(f"Dataset extracted to {data_dir}")


if __name__ == "__main__":
    main()
