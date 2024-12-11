import os
import numpy as np

# Step 1: Load 3D volume data
data, _ = readvolume('ac4_EM')  # data shape: (H,W,D)

# Step 2: Load the model (deeplab_epoch200)
# In the original code: load('deeplab_epoch200.mat')
# We must have a corresponding .h5 model or a code to build the same model.
deeplab_model = load_model_from_matfile('deeplab_epoch200.mat')  # Implement inside load_model_from_matfile

chunk_size = 256
s = data.shape
num_pieces = s[0] // chunk_size

# Prepare result arrays
Y_pred = np.zeros(s, dtype=np.uint8)
score = np.zeros((chunk_size, chunk_size, s[2]), dtype=np.float32)
bigger_image = np.pad(data, ((64,64),(64,64),(0,0)), mode='constant', constant_values=0)

# Perform segmentation on each chunk
for im in range(s[2]):  # im: slice index along depth dimension
    print(f"Processing slice {im+1}/{s[2]}")
    for j in range(num_pieces):
        for k in range(num_pieces):
            # Extract patch from data
            patch = data[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im]
            # Use semanticseg to get predicted class and scores
            result_class, result_score = semanticseg(patch, deeplab_model)
            # Place results into Y_pred and score arrays
            Y_pred[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im] = result_class
            score[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im] = result_score

# After predictions, map class indices to pixel values:
# Original code: Y_pred(Y_pred==1)=255; Y_pred(Y_pred==2)=0;
# Adjust accordingly. 
Y_pred_a = np.zeros_like(Y_pred, dtype=np.uint8)
Y_pred_a[Y_pred==1] = 255
Y_pred_a[Y_pred==0] = 0

targetDir = "ac4_seg_images_deeplab_epoch200_pred"
os.makedirs(targetDir, exist_ok=True)

# Save each slice as PNG
r1, r2 = 1, s[2]
for i in range(r1, r2+1):
    filename = os.path.join(targetDir, f"{i:04d}_.png")
    cv2.imwrite(filename, Y_pred_a[:,:,i-1])
