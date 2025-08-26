
# =====================================
# 10) utils/blocks.py
# =====================================
import numpy as np

def iterate_blocks(feats, block_size=4096):
    N = feats.shape[0]
    idx = 0
    order = np.arange(N)
    while idx < N:
        j = min(idx+block_size, N)
        yield order[idx:j]
        idx = j
