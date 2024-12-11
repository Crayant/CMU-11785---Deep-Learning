import os
import numpy as np

data, _ = readvolume('ac4_EM')
chunk_size = 256
s = data.shape
num_pieces = s[0] // chunk_size

# Load another model:
# Suppose you have unet_batchsizes4_epoch25.h5 converted and ready:
unet_model = tf.keras.models.load_model('unet_batchsizes4_epoch25.h5', compile=False)

Y_pred = np.zeros(s, dtype=np.uint8)
score = np.zeros((chunk_size, chunk_size, s[2]), dtype=np.float32)
bigger_image = np.pad(data, ((64,64),(64,64),(0,0)), mode='constant', constant_values=0)

for im in range(s[2]):
    print(f"Processing slice {im+1}/{s[2]} with UNet")
    for j in range(num_pieces):
        for k in range(num_pieces):
            patch = data[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im]
            result_class, result_score = semanticseg(patch, unet_model)
            Y_pred[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im] = result_class
            score[j*chunk_size:(j+1)*chunk_size, k*chunk_size:(k+1)*chunk_size, im] = result_score

# Map classes to pixel values if needed
Y_pred_a = np.zeros_like(Y_pred, dtype=np.uint8)
Y_pred_a[Y_pred==1] = 255
Y_pred_a[Y_pred==0] = 0

# If you wish to save results, specify the directory:
targetDir = "ac4_seg_images_unet_batchsize4_epoch25_pred"
os.makedirs(targetDir, exist_ok=True)

for i in range(1, s[2]+1):
    filename = os.path.join(targetDir, f"{i:04d}_.png")
    cv2.imwrite(filename, Y_pred_a[:,:,i-1])
