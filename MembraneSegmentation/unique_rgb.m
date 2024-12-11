import numpy as np

def unique_rgb(in_array):
    """
    Convert a 4D RGBA-like array (H,W,3,Frames) into a single-channel representation
    by treating the 3 channels as a 24-bit number: R*(256^2) + G*(256) + B.
    
    Then permute dimensions from (H,W,3,Frames) to (H,W,Frames) - essentially
    collapsing RGB into a single channel.

    The original code:
    out = permute(double(in(:,:,3,:)) + double(in(:,:,2,:))*256 + double(in(:,:,1,:))*256^2,[1 2 4 3]);
    means:
    - Take the R,G,B channels as in_array[:,:,1,:], in_array[:,:,2,:], in_array[:,:,3,:] if 1-based indexing.
    - In zero-based indexing (Python): R is in_array[:,:,0,:], G in_array[:,:,1,:], B in_array[:,:,2,:].
    - Construct: B + G*256 + R*(256^2), then permute.
    
    Parameters:
        in_array (ndarray): shape (H,W,3,F) - F could be frames or slices.
    
    Returns:
        out (ndarray): shape (H,W,F) single-channel
    """

    R = in_array[:,:,0,:].astype(np.float64)
    G = in_array[:,:,1,:].astype(np.float64)
    B = in_array[:,:,2,:].astype(np.float64)
    
    combined = B + G*256 + R*(256**2)
    # combined shape: (H,W,F)
    # The original permute([1 2 4 3]) from (H,W,3,F) means:
    # Original: (H,W,3,F)
    # After selecting B,G,R separately and combining them, we now have (H,W,F).
    # So we may not need an additional permute if we constructed correctly.
    # If original had [1 2 4 3], it was swapping the last two dims:
    # But we ended up with (H,W,F) directly, no extra dimension left.

    
    out = combined # no further permute needed since we've directly constructed the final shape
    return out
