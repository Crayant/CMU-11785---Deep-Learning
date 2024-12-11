import os
import numpy as np
import cv2
import tensorflow as tf

# Load your 3D volume data
# data: numpy array with shape (H, W, D)
data = ... # Load your volume here, e.g. np.load('ac4_EM.npy')

chunk_size = 256
s = data.shape
num_pieces = s[0] // chunk_size

# Load the trained model
model_path = 'MyNet_batchsize_4_epoch30.h5' 
model = tf.keras.models.load_model(model_path, compile=False)

Y_pred = np.zeros(s, dtype=np.uint8)
score = np.zeros((chunk_size, chunk_size, s[2]), dtype=np.float32)
bigger_image = np.pad(data, ((64,64),(64,64),(0,0)), mode='constant', constant_values=0)

def semanticseg(patch, model):
    """
    Simulate semantic segmentation inference.
    patch: 2D numpy array (chunk_size x chunk_size), grayscale
    model: a segmentation model that outputs probabilities
    Returns:
        result_class: 2D array of predicted class indices
        result_score: 2D array of max probability scores
    """
    if patch.ndim == 2:
        patch = patch[..., np.newaxis]
    patch = patch.astype(np.float32)/255.0
    patch_in = np.expand_dims(patch, axis=0) # shape: (1,H,W,1)

    patch_in = np.repeat(patch_in, 3, axis=-1) # (1,H,W,3)
    
    pred = model.predict(patch_in)
    # pred: (1,H,W,C)
    pred_class = np.argmax(pred, axis=-1)[0] # (H,W)
    pred_score = np.max(pred, axis=-1)[0]    # (H,W)
    return pred_class.astype(np.uint8), pred_score.astype(np.float32)

# First prediction pass for each chunk
for im in range(s[2]):
    for j in range(num_pieces):
        for k in range(num_pieces):
            patch = data[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im]
            result_class, result_score = semanticseg(patch, model)
            Y_pred[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im] = result_class
            score[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im] = result_score

# Vertical patch refinement
for im in range(s[2]):
    for j in range(num_pieces):
        for k in range(num_pieces-1):

            patch = data[j*chunk_size:(j+1)*chunk_size, k*chunk_size+int(chunk_size/2):k*chunk_size+int(chunk_size/2)+chunk_size, im]
            result_a, result_b = semanticseg(patch, model)
            # Update Y_pred in the specified subregion
            # Indices in MATLAB: (j-1)*chunk_size +1 + chunk_size*3/4 to k*chunk_size+chunk_size/4
            # Adjust for 0-based:
            # chunk_size*3/4 = int(chunk_size*3/4)
            # chunk_size/4 = int(chunk_size/4)
            top = j*chunk_size
            left = k*chunk_size + int(chunk_size*3/4)
            bottom = top + chunk_size
            right = k*chunk_size + int(chunk_size/4) + chunk_size
            # The extracted portion from result_a: result_a[:, chunk_size/4+1 : chunk_size*3/4]

            Y_pred[top:bottom, left:right, im] = result_a[:, (chunk_size//4):(chunk_size*3//4)]

# Horizontal patch refinement
offset = 0 
for im in range(s[2]):
    for j in range(num_pieces):
        for k in range(num_pieces-1):
            patch = data[k*chunk_size+int(chunk_size/2):k*chunk_size+int(chunk_size/2)+chunk_size, j*chunk_size:(j+1)*chunk_size, im]
            result_a, result_b = semanticseg(patch, model)
            # Update Y_pred

            # and (j-1)*chunk_size +1 : j*chunk_size
            top = k*chunk_size + int(chunk_size*3/4)
            left = j*chunk_size
            bottom = k*chunk_size + int(chunk_size/4) + chunk_size
            right = left + chunk_size
            # From result_a: result_a(chunk_size/4+1:chunk_size*3/4,:)

            Y_pred[top:bottom, left:right, im] = result_a[chunk_size//4:chunk_size*3//4,:]

targetDir = "ac4_seg_images_MyNet_Pred_batchsize4_epoch30_overlap"
os.makedirs(targetDir, exist_ok=True)
r1, r2 = 1, s[2]

# Map classes to pixel values:
Y_pred_a = np.zeros_like(Y_pred, dtype=np.uint8)
Y_pred_a[Y_pred==0] = 0
Y_pred_a[Y_pred==1] = 255

# Save each slice
for i in range(r1, r2+1):
    filename = os.path.join(targetDir, f"{i:04d}_.png")
    cv2.imwrite(filename, Y_pred_a[:,:,i-1])
