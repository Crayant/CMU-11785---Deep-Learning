import numpy as np

def break_image(im, sq_len):
    """
    Split a large square image into smaller non-overlapping patches of size sq_len x sq_len.
    
    Parameters:
        im (ndarray): Input 2D array representing a grayscale image. Shape: (H, W)
        sq_len (int): The size of each square patch.
    
    Returns:
        out (ndarray): 3D array of shape (sq_len, sq_len, num_patches), where each slice along
                       the third dimension is one patch.
    """
    h, w = im.shape[:2]  # Assume im is 2D (e.g., grayscale)
    num_pieces = h // sq_len
    
    # Initialize the output array
    # total patches = num_pieces * num_pieces
    out = np.zeros((sq_len, sq_len, num_pieces*num_pieces), dtype=np.uint8)
    
    patch_idx = 0
    for i in range(num_pieces):
        for j in range(num_pieces):
            row_start = i * sq_len
            row_end = row_start + sq_len
            col_start = j * sq_len
            col_end = col_start + sq_len
            patch = im[row_start:row_end, col_start:col_end]
            out[:,:,patch_idx] = patch
            patch_idx += 1
    
    return out
