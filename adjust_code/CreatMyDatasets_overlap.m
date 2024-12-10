import os
import cv2
import numpy as np

# break_image_overlap  overlapping patch
def break_image_overlap(img, output_size=256, overlap=0):
    

    h, w = img.shape[:2]
    patches = []
    step = output_size  # when no overlap，move output_size
    for y in range(0, h - output_size + 1, step):
        for x in range(0, w - output_size + 1, step):
            patch = img[y:y+output_size, x:x+output_size]
            patches.append(patch)
    # output shape (output_size, output_size, N)
    if len(patches) > 0:
        out = np.stack(patches, axis=2)
    else:
        out = None
    return out

def get_border(I):
    # using Canny to decte get_border
    edges = cv2.Canny(I, 100, 200)
    # return binary
    return (edges > 0).astype(np.uint8)



output_size = 256

labelDir = "ac3_EM"
targetDir = "ac3_EM_patch_256_overlap"
prefix = 'Thousand_highmag_256slices_2kcenter_1k_inv_'
r1=0
r2=255
os.makedirs(targetDir, exist_ok=True)

# make it small patches
for i in range(r1, r2+1):

    # i*1e-4 = 0.0000xxx
    # format '%1.4f'->'0.00xx'，ie., i=0 -> '0.0000'
    s_val = i * 1e-4
    s = f"{s_val:1.4f}"  # '0.0000'
    s = s[2:]  # remove '0.'
    # s_f = strcat(prefix,s)
    s_f = prefix + s

    I = cv2.imread(os.path.join(labelDir, s_f + '.png'), cv2.IMREAD_GRAYSCALE)

    out = break_image_overlap(I, output_size)
    if out is not None:
        # out[:,:,patch_j] 在Python中 out.shape = (256,256,N)
        num_patches = out.shape[2]
        for patch_j in range(num_patches):
            filename = os.path.join(targetDir, f"{s}_{patch_j+1}.png")
            train_data = out[:,:,patch_j]
            cv2.imwrite(filename, train_data.astype(np.uint8))


#--------------------------------------------
labelDir = "ac3_dbseg_images"
targetDir = "ac3_dbseg_images_bw_new_256_overlap"
prefix = 'ac3_daniel_s'
r1=1
r2=256
os.makedirs(targetDir, exist_ok=True)

# label membrane 
for i in range(r1, r2+1):
    s_val = i * 1e-4
    s = f"{s_val:1.4f}"
    s = s[2:] 
    
    I = cv2.imread(os.path.join(labelDir, prefix+s+'.png'), cv2.IMREAD_GRAYSCALE)

    # s = r2 - str2num(s)  => in Python:
    s_float = float(s)
    s_float = r2 - s_float
    # refomated 1e-4
    s_val2 = s_float * 1e-4
    s2 = f"{s_val2:1.4f}"
    s2 = s2[2:]  # again strip
    
    out = get_border(I)
    filename = os.path.join(targetDir, prefix + s2 + '.png')
    cv2.imwrite(filename, (out*255).astype(np.uint8))


#--------------------------------------------
labelDir = "ac3_dbseg_images_bw_new_256_overlap"
targetDir = "ac3_dbseg_images_bw_patch_new_256_overlap"
prefix = 'ac3_daniel_s'
r1=0
r2=255
os.makedirs(targetDir, exist_ok=True)

for i in range(r1, r2+1):
    filename_in = os.path.join(labelDir, f"{prefix}{i:04d}.png")
    I = cv2.imread(filename_in, cv2.IMREAD_GRAYSCALE)
    out = break_image_overlap(I, output_size)
    if out is not None:
        num_patches = out.shape[2]
        for patch_j in range(num_patches):
            filename_out = os.path.join(targetDir, f"{i:04d}_{patch_j+1}.png")
            cv2.imwrite(filename_out, out[:,:,patch_j])
