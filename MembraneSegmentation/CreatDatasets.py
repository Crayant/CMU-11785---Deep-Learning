import os
import cv2
import numpy as np

# Parameters
output_size = 256

labelDir = "ac3_EM"
targetDir = "ac3_EM_patch_256"
prefix = 'Thousand_highmag_256slices_2kcenter_1k_inv_'
r1, r2 = 0, 255
os.makedirs(targetDir, exist_ok=True)

# Process original images into small patches
for i in range(r1, r2+1):
    # Convert i to the required string format
    # For i=0 -> '0.0000', s(3:end) means skip first two chars '0.'

    s_val = i * 1e-4
    s_str = f"{s_val:1.4f}"  # '0.0000'
    s = s_str[2:]  # remove the '0.' prefix, now s could be '0000' when i=0.
    s_f = prefix + s

    I = cv2.imread(os.path.join(labelDir, s_f + '.png'), cv2.IMREAD_GRAYSCALE)
    out = break_image(I, output_size)

    # Save each patch
    for patch_j in range(out.shape[2]):
        filename = os.path.join(targetDir, f"{s}_{patch_j+1}.png")
        train_data = out[:,:,patch_j]
        # Expand to 3-channel by repeating the same mask:
        mask = train_data
        train_data_3ch = np.stack([train_data, mask, mask], axis=-1)
        cv2.imwrite(filename, train_data_3ch)

# Process label images to create a border-only image
labelDir = "ac3_dbseg_images"
targetDir = "ac3_dbseg_images_bw_new_256"
prefix = 'ac3_daniel_s'
r1, r2 = 1, 256
os.makedirs(targetDir, exist_ok=True)

for i in range(r1, r2+1):
    s_val = i * 1e-4
    s_str = f"{s_val:1.4f}"
    s = s_str[2:]  # remove '0.'
    I = cv2.imread(os.path.join(labelDir, prefix+s+'.png'), cv2.IMREAD_GRAYSCALE)

    # Inverse index: s = r2 - str2num(s)
    # s was string of form '0001', convert back to float:
    s_float = float(s)
    s_float = r2 - s_float
    s_val2 = s_float * 1e-4
    s2_str = f"{s_val2:1.4f}"
    s2 = s2_str[2:]

    out = get_border(I)
    filename = os.path.join(targetDir, prefix+s2+'.png')
    cv2.imwrite(filename, (out*255).astype(np.uint8))

# Now break the processed label images into patches
labelDir = "ac3_dbseg_images_bw_new_256"
targetDir = "ac3_dbseg_images_bw_patch_new_256"
prefix = 'ac3_daniel_s'
r1, r2 = 0, 255
os.makedirs(targetDir, exist_ok=True)

for i in range(r1, r2+1):
    fname = f"{prefix}{i:04d}.png"
    I = cv2.imread(os.path.join(labelDir, fname), cv2.IMREAD_GRAYSCALE)
    out = break_image(I, output_size)
    for patch_j in range(out.shape[2]):
        filename = os.path.join(targetDir, f"{i:04d}_{patch_j+1}.png")
        cv2.imwrite(filename, out[:,:,patch_j])
