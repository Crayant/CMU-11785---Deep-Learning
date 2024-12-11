import numpy as np

def break_image_overlap(im, sq_len):
    """
    Split a large square image into overlapping patches using a half-step increment.
    For each dimension, patches are extracted starting at intervals of sq_len/2,
    resulting in a denser sampling.

    Parameters:
        im (ndarray): 2D array (H,W) representing a grayscale image.
        sq_len (int): Patch size.
    
    Returns:
        out (ndarray): 3D array, shape (sq_len, sq_len, total_patches), where total_patches = (2*num_pieces-1)^2.
    """
    h, w = im.shape[:2]
    num_pieces = h // sq_len
    # total patches:
    total_patches = (2*num_pieces - 1)*(2*num_pieces - 1)
    out = np.zeros((sq_len, sq_len, total_patches), dtype=np.uint8)
    
    i_values = np.arange(1, num_pieces+0.001, 0.5)
    j_values = np.arange(1, num_pieces+0.001, 0.5)
    
    patch_idx = 0
    for i_val in i_values:
        for j_val in j_values:
            row_start = int((i_val-1)*sq_len)
            row_end = row_start + sq_len
            col_start = int((j_val-1)*sq_len)
            col_end = col_start + sq_len
            patch = im[row_start:row_end, col_start:col_end]
            out[:,:,patch_idx] = patch
            patch_idx += 1
    
    return out
