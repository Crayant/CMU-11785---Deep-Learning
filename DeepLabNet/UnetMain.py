import os
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

output_size = 256
imageSize = (output_size, output_size, 1)
numClasses = 2
encoderDepth = 5
load_net = 0

if load_net == 1:
    # load existing model
    net = tf.keras.models.load_model('unet_batchsizes4_epoch25.h5', compile=False)
else:
    net = unetLayers(imageSize, numClasses, encoderDepth)

# imageDir and labelDir
imageDir = "ac3_EM_patch_256_overlap"
labelDir = "ac3_dbseg_images_bw_patch_new_256_overlap"

image_paths = sorted(glob.glob(os.path.join(imageDir, "*.png")))
label_paths = sorted(glob.glob(os.path.join(labelDir, "*.png")))

# Map label 255->1,0->0
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)/255.0
    img = img[...,np.newaxis] # (H,W,1)
    return img

def load_label(path):
    lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    lbl[lbl==255] = 1
    lbl[lbl==0] = 0
    lbl = lbl[...,np.newaxis]
    return lbl

indices = np.arange(len(image_paths))
# Split data: 0.99 train, 0.01 val, 0 test
train_idx, val_idx = train_test_split(indices, test_size=0.01, random_state=42)
test_idx = [] 

train_image_paths = [image_paths[i] for i in train_idx]
train_label_paths = [label_paths[i] for i in train_idx]
val_image_paths = [image_paths[i] for i in val_idx]
val_label_paths = [label_paths[i] for i in val_idx]

def data_generator(img_paths, lbl_paths, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, lbl_paths))
    def parse_func(img_p, lbl_p):
        img = tf.numpy_function(load_image, [img_p], tf.float32)
        lbl = tf.numpy_function(load_label, [lbl_p], tf.float32)
        img.set_shape([output_size, output_size, 1])
        lbl.set_shape([output_size, output_size, 1])
        return img, lbl
    dataset = dataset.map(parse_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = data_generator(train_image_paths, train_label_paths, 4)
val_dataset = data_generator(val_image_paths, val_label_paths, 4)

# Define weighted loss if needed, or simple cross entropy
def weighted_loss(y_true, y_pred):
    # For simplicity, use standard cross-entropy
    # If need class weights, compute them before and apply
    y_true = tf.cast(y_true, tf.int32)
    y_true_onehot = tf.one_hot(y_true, depth=numClasses)
    loss = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred+1e-7), axis=-1)
    return tf.reduce_mean(loss)

Maxepoch = 4
optimizer = tf.keras.optimizers.Adam(1e-3)
net.compile(optimizer=optimizer, loss=weighted_loss, metrics=['accuracy'])

for training_iter in range(1,51):
    if training_iter == 1:
        net.fit(train_dataset, epochs=Maxepoch, validation_data=val_dataset)
    else:
        net.fit(train_dataset, epochs=Maxepoch, validation_data=val_dataset)
    net.save(f'unet_tv9901_epoch{training_iter*Maxepoch}.h5')
