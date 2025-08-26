import argparse
import glob
import os
import numpy as np
import pylas
from tqdm import tqdm
from utils.geometry import compute_knn_normals, estimate_height_above_ground
from utils.io import safe_makedirs

LABEL_MAP_HINT = {
    14: 1,  # powerline → class 1
    15: 1,  # powerline → class 1
    16: 2,  # tower/pole → class 2 (adjust if your data uses 17)
}

def process_one(in_path, out_dir, map_from_existing_labels=False, grid=2.0, k_normals=16):
    name = os.path.splitext(os.path.basename(in_path))[0]

    # Open the .laz file using pylas
    las = pylas.read(in_path)

    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    intensity = getattr(las, 'intensity', np.zeros(len(xyz), dtype=np.float32)).astype(np.float32)
    try:
        return_num = las.return_number.astype(np.float32)
        num_returns = las.number_of_returns.astype(np.float32)
    except Exception:
        return_num = np.zeros(len(xyz), dtype=np.float32)
        num_returns = np.ones(len(xyz), dtype=np.float32)

    # Height above ground via coarse ground grid min z
    hag = estimate_height_above_ground(xyz, cell=grid)

    # Local normals (optional but useful for wires)
    normals = compute_knn_normals(xyz, k=k_normals)

    feats = np.concatenate([
        xyz,                                 # 0:3
        intensity[:, None],                  # 3
        return_num[:, None], num_returns[:, None],  # 4,5
        hag[:, None],                        # 6
        normals                              # 7:9
    ], axis=1).astype(np.float32)

    labels = np.zeros((len(xyz),), dtype=np.int64)  # Default other = 0
    if map_from_existing_labels and hasattr(las, 'classification'):
        src = las.classification
        for k, v in LABEL_MAP_HINT.items():
            labels[src == k] = v

    out_path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(out_path, feats=feats, labels=labels)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_glob', required=True, help='Glob of input .laz files')
    ap.add_argument('--out_dir', required=True, help='Directory to save processed .npz files')
    ap.add_argument('--map_from_existing_labels', action='store_true', help='Map labels from existing .laz classification')
    ap.add_argument('--grid', type=float, default=2.0, help='Grid size for height above ground calculation')
    ap.add_argument('--k_normals', type=int, default=16, help='Number of nearest neighbors for normal estimation')
    args = ap.parse_args()

    # Create the output directory if it doesn't exist
    safe_makedirs(args.out_dir)

    # Get all the .laz files from the input path
    paths = sorted(glob.glob(args.in_glob))
    
    # Process each file and save to the output directory
    for p in tqdm(paths, desc='Preprocessing'):  
        try:
            process_one(p, args.out_dir, args.map_from_existing_labels, args.grid, args.k_normals)
        except Exception as e:
            print(f"[WARN] Failed {p}: {e}")




