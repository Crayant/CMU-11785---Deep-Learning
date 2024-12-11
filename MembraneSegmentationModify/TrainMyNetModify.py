import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Paths
imageDir = "ac3_EM_patch_256"
labelDir = "ac3_dbseg_images_bw_patch_new_256"
os.makedirs(labelDir, exist_ok=True) # Just ensure directory structure

classNames = ["border","no_border"]
labelIDs   = [255, 0]
numClasses = 2

# Data loading
image_paths = sorted(glob.glob(os.path.join(imageDir, "*.png")))
label_paths = sorted(glob.glob(os.path.join(labelDir, "*.png")))

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)/255.0
    return img

def load_label(path):
    lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # map 255->1, 0->0
    lbl[lbl==255] = 1
    lbl[lbl==0] = 0
    lbl = lbl[..., np.newaxis]
    return lbl

# Split dataset: 80% train, 20% val, 0% test
indices = np.arange(len(image_paths))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
test_indices = [] # no test set

train_image_paths = [image_paths[i] for i in train_indices]
train_label_paths = [label_paths[i] for i in train_indices]
val_image_paths = [image_paths[i] for i in val_indices]
val_label_paths = [label_paths[i] for i in val_indices]

# Count class frequencies
def count_pixels(label_list):
    count_0 = 0
    count_1 = 0
    image_pixel_counts = []
    for lp in label_list:
        lbl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        total_pixels = lbl.size
        image_pixel_counts.append(total_pixels)
        c1 = np.sum(lbl==255)
        c0 = np.sum(lbl==0)
        count_0 += c0
        count_1 += c1
    return (count_0, count_1, image_pixel_counts)

count_0, count_1, image_pixel_counts = count_pixels(train_label_paths)
numberPixels = count_0 + count_1
frequency_0 = count_0/numberPixels
frequency_1 = count_1/numberPixels
# classWeights based on frequency

classWeights = [1/frequency_1, 1/frequency_0] # adjust order if needed


# Approximate imageFreq from our counts:
avg_image_size = np.mean(image_pixel_counts)
imageFreq = np.array([count_0,count_1]) / (len(train_label_paths)*avg_image_size)
median_val = np.median(imageFreq)
classWeights = median_val / imageFreq


# Build tf.data.Dataset
def data_generator(img_paths, lbl_paths, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, lbl_paths))
    def _parse(img_p, lbl_p):
        img = tf.numpy_function(load_image, [img_p], tf.float32)
        lbl = tf.numpy_function(load_label, [lbl_p], tf.float32)
        img.set_shape([256,256,3])
        lbl.set_shape([256,256,1])
        return img, lbl
    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = data_generator(train_image_paths, train_label_paths, batch_size=4)
val_dataset = data_generator(val_image_paths, val_label_paths, batch_size=4)


# Weighted loss
def weighted_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true_onehot = tf.one_hot(y_true, depth=numClasses)

    
    weights_map = y_true_onehot[...,0]*classWeights[0] + y_true_onehot[...,1]*classWeights[1]
    loss = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred+1e-7), axis=-1)
    loss = loss * weights_map
    return tf.reduce_mean(loss)

MaxEpoch = 30
optimizer = tf.keras.optimizers.Adam(1e-3)
net.compile(optimizer=optimizer, loss=weighted_loss, metrics=['accuracy'])

# Train multiple times and save after each cycle
for training_iter in range(1,51):
    if training_iter == 1:
        net.fit(train_dataset, epochs=MaxEpoch, validation_data=val_dataset)
    else:
        # Just continue training
        net.fit(train_dataset, epochs=MaxEpoch, validation_data=val_dataset)
    net.save(f'MyNet_batchsize_4_epoch{training_iter*MaxEpoch}.h5')
