import numpy as np

def break_image(im, sq_len):
    # im is a 2D NumPy array
    # sq_len is the size of the square patch
    s = im.shape
    # Compute how many full patches along one dimension
    num_pieces = s[0] // sq_len  # assuming s[0] == s[1] and divisible by sq_len
    
    # The number of patches along one dimension is (2*num_pieces - 1)
    # because we are stepping in increments of 0.5
    out_height = sq_len
    out_width = sq_len
    total_patches = (2 * num_pieces - 1) * (2 * num_pieces - 1)
    
    # Initialize output array
    out = np.zeros((out_height, out_width, total_patches), dtype=np.uint8)
    
    # i and j go from 1 to num_pieces in steps of 0.5
    # np.arange(1, num_pieces + 0.1, 0.5) generates [1,1.5,2,2.5,...,num_pieces]
    i_values = np.arange(1, num_pieces + 0.001, 0.5)
    j_values = np.arange(1, num_pieces + 0.001, 0.5)
    
    # Iterate over all i and j positions
    for i_val in i_values:
        for j_val in j_values:
            # Compute the patch index in the output
            patch_index = int((i_val * 2 - 2) * (num_pieces * 2 - 1) + (j_val * 2 - 1) - 1)
            
            # Compute the slice indices
            row_start = int((i_val - 1) * sq_len)
            row_end = int(i_val * sq_len)
            col_start = int((j_val - 1) * sq_len)
            col_end = int(j_val * sq_len)
            
            # Extract the patch
            patch = im[row_start:row_end, col_start:col_end]
            
            # Assign to out
            out[:,:,patch_index] = patch
    
    return out
