import argparse
import numpy as np
import torch
import laspy
from models.pointnet import PointNetClassifier
from utils.blocks import iterate_blocks

# Map class predictions back to LAS codes
CLASS_TO_LAS_CODE = {0: 1, 1: 15, 2: 16}

# Define RGB colors for different classes
CLASS_TO_COLOR = {
    0: (255, 0, 0),  # Red for Poles
    1: (0, 255, 0),  # Green for Powerlines
    2: (0, 0, 255),  # Blue for Defaults (or other)
}

def load_model(ckpt_path, num_classes=3, device='cpu'):
    model = PointNetClassifier(in_dim=10, num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()
    return model

def predict_tile(model, feats, device='cpu'):
    preds = np.zeros((feats.shape[0],), dtype=np.int64)
    for idxs in iterate_blocks(feats, block_size=4096):
        block = torch.from_numpy(feats[idxs]).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(block)  # [1,C]
            logits_pts = logits.repeat(block.shape[1], 1)  # [N,C]
            pred = logits_pts.argmax(-1).cpu().numpy()
        preds[idxs] = pred
    return preds

def convert_to_point_format_2(input_laz, output_laz):
    """ Convert the LAS file to point format 2 (supports RGB) """
    with laspy.open(input_laz) as f:
        las = f.read()

    # Create a new LAS file with Point Format 2 (supports RGB)
    header = laspy.LasHeader(point_format=2, version="1.2")
    las_out = laspy.LasData(header)

    # Copy data from original LAS to the new LAS object
    las_out.x = las.x
    las_out.y = las.y
    las_out.z = las.z
    las_out.classification = las.classification
    las_out.intensity = las.intensity

    # Write the modified LAS file to the output path
    las_out.write(output_laz)
    print(f"Converted {input_laz} to point format 2 and saved as {output_laz}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_ckpt', required=True)
    ap.add_argument('--in_laz', required=True)
    ap.add_argument('--out_laz', required=True)
    args = ap.parse_args()

    # Convert input .laz to point format 2 (if necessary)
    temp_laz = "temp_converted.laz"
    convert_to_point_format_2(args.in_laz, temp_laz)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_ckpt, device=device)

    # Open the input converted LAS/LAZ file
    with laspy.open(temp_laz) as f:
        las = f.read()

    # Process point cloud data
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    intensity = getattr(las, 'intensity', np.zeros(len(xyz), dtype=np.float32)).astype(np.float32)
    try:
        return_num = las.return_number.astype(np.float32)
        num_returns = las.number_of_returns.astype(np.float32)
    except Exception:
        return_num = np.zeros(len(xyz), dtype=np.float32)
        num_returns = np.ones(len(xyz), dtype=np.float32)

    # Process geometry (HAG, normals)
    from utils.geometry import estimate_height_above_ground, compute_knn_normals
    hag = estimate_height_above_ground(xyz, cell=2.0)
    normals = compute_knn_normals(xyz, k=16)
    feats = np.concatenate([xyz, intensity[:, None], return_num[:, None], num_returns[:, None], hag[:, None], normals], axis=1).astype(np.float32)

    # Perform prediction on the point cloud data
    preds = predict_tile(model, feats, device=device)

    # Map predictions to LAS codes
    las.classification = np.vectorize(CLASS_TO_LAS_CODE.get)(preds).astype(np.uint8)

    # Map predictions to RGB colors based on the classification
    red, green, blue = [], [], []
    for pred in preds:
        color = CLASS_TO_COLOR.get(pred, (0, 0, 0))  # Default to black if the class is unknown
        red.append(color[0])
        green.append(color[1])
        blue.append(color[2])

    # Assign the RGB values to the LAS file
    las.red = np.array(red, dtype=np.uint16)  # LAS RGB values should be in range [0, 65535]
    las.green = np.array(green, dtype=np.uint16)
    las.blue = np.array(blue, dtype=np.uint16)

    # Save output as a new LAZ file
    las.write(args.out_laz)
    print(f"Wrote {args.out_laz}")

if __name__ == '__main__':
    main()
