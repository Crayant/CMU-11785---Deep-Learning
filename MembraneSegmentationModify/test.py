import os
import cv2
import numpy as np

def break_image(im, sq_len):

    h, w = im.shape[:2]
    num_pieces = h // sq_len
    out = np.zeros((sq_len, sq_len, num_pieces*num_pieces), dtype=np.uint8)
    
    patch_idx = 0
    for i in range(num_pieces):
        for j in range(num_pieces):
            patch = im[i*sq_len:(i+1)*sq_len, j*sq_len:(j+1)*sq_len]
            out[:,:,patch_idx] = patch
            patch_idx += 1
    return out

output_size = 128
labelDir = "ac3_dbseg_images_bw_new_128"
targetDir = "ac3_dbseg_images_bw_patch_new_128"
prefix = 'ac3_daniel_s'
r1, r2 = 0, 255
os.makedirs(targetDir, exist_ok=True)

for i in range(r1, r2+1):
    fname = f"{prefix}{i:04d}.png"
    I = cv2.imread(os.path.join(labelDir, fname), cv2.IMREAD_GRAYSCALE)
    patches = break_image(I, output_size)
    for patch_j in range(patches.shape[2]):
        outname = f"{prefix}{i:04d}_{patch_j+1}.png"
        cv2.imwrite(os.path.join(targetDir,outname), patches[:,:,patch_j])
