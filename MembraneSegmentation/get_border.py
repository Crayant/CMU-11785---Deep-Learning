import cv2
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion

def unique_rgb(x):
    """
    Placeholder function to convert an RGB label image into a single-channel label.
    Implement this according to your needs.
    """
    # For demonstration, assume x is already single-channel.
    # If x is RGB and each unique color represents a different class, you'd map them accordingly.
    if x.ndim == 3:
        # Just take one channel or implement logic
        I_s = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        return I_s
    return x

def get_border(x):
    """
    Given a label image (possibly RGB) representing cell membranes, extract the border.
    1. Convert RGB label to single-channel if needed.
    2. Dilate and erode to find boundary: BW = dilate(I_s) != erode(I_s)
    3. Set regions where I_s == 0 to 1.

    Parameters:
        x (ndarray): Input image, could be RGB or single-channel.
    Returns:
        BW (ndarray): Binary image where cell membrane borders are marked as 1.
    """
    # Convert to single-channel if needed
    try:
        I_s = unique_rgb(x)
    except:
        I_s = x
    
    # Perform dilation and erosion
    # Structuring element of size (5,5)
    se = np.ones((5,5), dtype=np.uint8)
    dilated = grey_dilation(I_s, footprint=se)
    eroded = grey_erosion(I_s, footprint=se)
    
    # border = dilated != eroded
    BW = (dilated != eroded).astype(np.uint8)
    
    # Where I_s == 0, set BW to 1
    BW[I_s==0] = 1
    
    return BW
