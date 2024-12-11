import os
import numpy as np
import cv2
import tensorflow as tf

# Load the model
model_path = 'MyNet_tv9901_epoch8.h5'
model = tf.keras.models.load_model(model_path, compile=False)

# 'data' is a numpy 3D volume [H, W, D]
# For example:
# data = np.load('ac4_EM.npy')
data = ... # Load your data here as a numpy array, shape [H, W, D]

chunk_size = 256
s = data.shape
num_pieces = s[0] // chunk_size

Y_pred = np.zeros(s, dtype=np.uint8)
score = np.zeros((chunk_size, chunk_size, s[2]), dtype=np.float32)

def semanticseg(patch, model):
    # patch: (chunk_size, chunk_size)

    if patch.ndim == 2:
        patch = np.expand_dims(patch, axis=-1)
    patch = patch.astype(np.float32)/255.0
    patch = np.expand_dims(patch, axis=0)  # batch dimension
    
    pred = model.predict(patch)
    # pred shape: (1, H, W, num_classes)
    pred = pred[0]  # remove batch dim
    pred_class = np.argmax(pred, axis=-1)  # (H,W)
    pred_score = np.max(pred, axis=-1)     # max probability per pixel
    return pred_class, pred_score

# Forecast - process main patches
for im_idx in range(s[2]):
    print(im_idx+1)
    for j in range(num_pieces):
        for k in range(num_pieces):
            patch = data[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im_idx]
            result_class, result_score = semanticseg(patch, model)
            Y_pred[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im_idx] = result_class
            score[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im_idx] = result_score

# Vertical patch refinement
for im_idx in range(s[2]):
    print("Vertical:", im_idx+1)
    for j in range(num_pieces):
        for k in range(num_pieces-1):
            # vertical patch
            patch = data[j*chunk_size:(j+1)*chunk_size, (k*chunk_size + chunk_size//2):(k*chunk_size + chunk_size//2 + chunk_size), im_idx]
            result_a, result_b = semanticseg(patch, model)

            row_start = j*chunk_size
            row_end = (j+1)*chunk_size
            col_start = (k*chunk_size + chunk_size*3//4)
            col_end = (k*chunk_size + chunk_size//4 + chunk_size) 
            # Actually, chunk_size/4 +1: chunk_size*3/4 means [chunk_size/4 : chunk_size*3/4)
            
            inner_row_start = 0 # entire patch height
            inner_row_end = chunk_size
            inner_col_start = chunk_size//4
            inner_col_end = (chunk_size*3)//4
            

            Y_pred[row_start:row_end, col_start:col_end, im_idx] = result_a[:, inner_col_start:inner_col_end]

# Horizontal patch refinement
offset = 20
for im_idx in range(s[2]):
    print("Horizontal:", im_idx+1)
    for j in range(num_pieces):
        for k in range(num_pieces-1):
            patch = data[(k*chunk_size + chunk_size//2):(k*chunk_size + chunk_size//2 + chunk_size),
                          j*chunk_size:(j+1)*chunk_size, im_idx]
            result_a, result_b = semanticseg(patch, model)
            
            
            row_start = (k*chunk_size + chunk_size*3//4)
            row_end = (k*chunk_size + chunk_size//4 + chunk_size)
            col_start = (j*chunk_size + offset)
            col_end = ((j+1)*chunk_size - offset)
            
            inner_row_start = chunk_size//4
            inner_row_end = (chunk_size*3)//4
            inner_col_start = offset
            inner_col_end = chunk_size - offset
            
            Y_pred[row_start:row_end, col_start:col_end, im_idx] = result_a[inner_row_start:inner_row_end, inner_col_start:inner_col_end]

targetDir = "ac4_seg_images_MyNet_Pred_batchsize4_epoch30_overlap"
os.makedirs(targetDir, exist_ok=True)


Y_pred_a = Y_pred.copy()
Y_pred_a[Y_pred == 1] = 255
Y_pred_a[Y_pred == 0] = 0

# Save images
r1 = 1
r2 = s[2]
for i in range(r1, r2+1):
    filename = os.path.join(targetDir, f"{i:04d}_.png")
    cv2.imwrite(filename, Y_pred_a[:,:,i-1])  # i-1 for zero-based indexing
