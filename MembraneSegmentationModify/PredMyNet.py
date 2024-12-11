import os
import numpy as np
import cv2
import tensorflow as tf

# Parameters
chunk_size = 256
model_path = 'MyNet_batchsize_4_epoch30.h5'  # Converted or saved model in .h5 format
targetDir = "ac4_seg_images_MyNet_Pred_batchsize4_epoch30"
os.makedirs(targetDir, exist_ok=True)

# Load the 3D volume data
# For example:
# data = np.load('ac4_EM.npy')  
data = ... # Load your 3D volume here

s = data.shape  # s = (H, W, D)
num_pieces = s[0] // chunk_size

# Load the pre-trained model
model = tf.keras.models.load_model(model_path, compile=False)

# Pad the image if needed
bigger_image = np.pad(data, ((64,64),(64,64),(0,0)), mode='constant', constant_values=0)

# Prepare arrays for predictions
Y_pred = np.zeros_like(data, dtype=np.uint8)  # predicted labels
score = np.zeros((chunk_size, chunk_size, s[2]), dtype=np.float32) # if needed

def semanticseg(patch, model):

    # Ensure patch shape and type
    if patch.ndim == 2:
        patch = np.expand_dims(patch, axis=-1)
    patch = patch.astype(np.float32)/255.0
    patch_in = np.expand_dims(patch, axis=0)  # (1,H,W,1)
    # If model expects 3-ch, replicate channels or modify as needed:
    # Suppose the model is trained on 3-channel inputs:
    patch_in = np.repeat(patch_in, 3, axis=-1)  # (1,H,W,3)

    pred = model.predict(patch_in)
    # pred shape: (1,H,W,numClasses), assume numClasses=2
    # Argmax along the channel dimension to get class index
    pred_class = np.argmax(pred, axis=-1)[0] # (H,W)
    # Get max score if needed
    pred_score = np.max(pred, axis=-1)[0] # (H,W)
    return pred_class.astype(np.uint8), pred_score.astype(np.float32)

# Run inference on each chunk
for im in range(s[2]):
    # im index from 0-based in Python
    for j in range(num_pieces):
        for k in range(num_pieces):
            patch = data[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im]
            result_class, result_score = semanticseg(patch, model)
            Y_pred[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im] = result_class
            score[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im] = result_score

# Convert class indices to pixel values
Y_pred_a = np.zeros_like(Y_pred, dtype=np.uint8)
Y_pred_a[Y_pred == 0] = 0
Y_pred_a[Y_pred == 1] = 255

# Save each slice as PNG
r1, r2 = 1, s[2]
for i in range(r1, r2+1):
    filename = os.path.join(targetDir, f"{i:04d}_.png")
    # index i-1 for 0-based
    cv2.imwrite(filename, Y_pred_a[:,:,i-1])
