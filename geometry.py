


# =====================================
# 8) utils/geometry.py
# =====================================
import numpy as np #    noqa: E402
from sklearn.neighbors import NearestNeighbors #  NearestNeighbors is used for KNN search


def compute_knn_normals(xyz, k=16): # Compute normals using k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k, len(xyz))).fit(xyz) #    # Fit the NearestNeighbors model to the point cloud
    # Find the k-nearest neighbors for each point in the point cloud
    idxs = nbrs.kneighbors(return_distance=False) #    # idxs is an array of shape (N, k) where N is the number of points in the point cloud
    normals = np.zeros_like(xyz, dtype=np.float32) #    # Initialize an array to store the normals for each point in the point cloud
    for i, nn_idx in enumerate(idxs): #    # Iterate over each point in the point cloud
        pts = xyz[nn_idx] - xyz[nn_idx].mean(0) #    # Get the k-nearest neighbors for the current point and center them around the mean
        cov = pts.T @ pts / max(pts.shape[0]-1, 1) #    # Compute the covariance matrix of the centered points
        # Perform Singular Value Decomposition (SVD) to find the normal vector
        _, _, v = np.linalg.svd(cov, full_matrices=False)   #    # Perform SVD on the covariance matrix to find the normal vector
        n = v[-1]   #    # The last row of V contains the normal vector
        normals[i] = n.astype(np.float32) #    # Store the normal vector for the current point
    return normals #    # Return the normals for the point cloud


def estimate_height_above_ground(xyz, cell=2.0): # Estimate height above ground using a coarse grid
    # Coarse ground: per XY grid, take min Z as ground, then HAG = z - zmin_cell
    xy = xyz[:, :2] #    # Extract the XY coordinates from the point cloud
    z = xyz[:, 2] #    # Extract the Z coordinates from the point cloud
    mn = xy.min(0) #    # Find the minimum XY coordinates in the point cloud
    ij = np.floor((xy - mn) / cell).astype(int) #    # Map the XY coordinates to a grid by subtracting the minimum and dividing by the cell size
    keys, inverse = np.unique(ij, axis=0, return_inverse=True) #    # Find the unique grid keys and their corresponding indices in the original point cloud
    zmin = np.zeros(len(keys), dtype=np.float32) #    # Initialize an array to store the minimum Z value for each grid cell
    for k in range(len(keys)):  #    # Iterate over each unique grid key
        zmin[k] = z[inverse==k].min() #    # Find the minimum Z value for the points in the current grid cell
    hag = z - zmin[inverse] #    # Compute the height above ground by subtracting the minimum Z value for the corresponding grid cell from the original Z values
    return hag.astype(np.float32)   #    # Return the height above ground for the point cloud

