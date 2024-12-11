def break_image(im, sq_len):
    """
    Overlapping patch extraction version:
    This final version uses half-step increments as described previously.
    """
    h, w = im.shape[:2]
    num_pieces = h // sq_len
    total_patches = (2*num_pieces - 1)*(2*num_pieces - 1)
    out = np.zeros((sq_len, sq_len, total_patches), dtype=np.uint8)

    i_values = np.arange(1, num_pieces+0.001, 0.5)
    j_values = np.arange(1, num_pieces+0.001, 0.5)
    patch_idx = 0
    for i_val in i_values:
        for j_val in j_values:
            row_start = int((i_val-1)*sq_len)
            col_start = int((j_val-1)*sq_len)
            patch = im[row_start:row_start+sq_len, col_start:col_start+sq_len]
            out[:,:,patch_idx] = patch
            patch_idx += 1
    return out
