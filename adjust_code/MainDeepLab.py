import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Parameters
imageDir = "ac3_EM_patch_256"
labelDir = "ac3_dbseg_images_bw_patch_new_256"  # Change to something like "segmentation_masks" if needed
output_size = 256
batch_size = 4
MaxEpoch = 4
numClasses = 2
classNames = ["border", "no_border"]
labelIDs = [255, 0]

# Read data list
image_paths = sorted(glob.glob(os.path.join(imageDir, "*.png")))
label_paths = sorted(glob.glob(os.path.join(labelDir, "*.png")))

# Map label images: 255 -> class 1, 0 -> class 0
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # shape: (H,W,1)
    return img

def load_label(path):
    lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    lbl[lbl == 255] = 1
    lbl[lbl == 0] = 0
    lbl = np.expand_dims(lbl, axis=-1)  # shape: (H,W,1)
    return lbl

# Split datasets, similar to dividerand(ds.NumObservations, 0.99, 0.01, 0)
indices = np.arange(len(image_paths))
train_indices, val_indices = train_test_split(indices, test_size=0.01, random_state=42)
test_indices = []  # No test set

train_image_paths = [image_paths[i] for i in train_indices]
train_label_paths = [label_paths[i] for i in train_indices]
val_image_paths = [image_paths[i] for i in val_indices]
val_label_paths = [label_paths[i] for i in val_indices]

# Count class frequencies, similar to countEachLabel
def count_pixels(label_list):
    # Count the number of pixels for class 0 and class 1
    count_0 = 0
    count_1 = 0
    for lp in label_list:
        lbl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        count_1 += np.sum(lbl == 255)
        count_0 += np.sum(lbl == 0)
    return count_0, count_1

count_0, count_1 = count_pixels(train_label_paths)
numberPixels = count_0 + count_1
frequency_0 = count_0 / numberPixels
frequency_1 = count_1 / numberPixels
classWeights = [1 / frequency_1, 1 / frequency_0] 
# Ensure classWeights follow the correct class order: 0=no_border, 1=border

# Build datasets
def data_generator(img_paths, lbl_paths, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, lbl_paths))
    def _parse_function(img_p, lbl_p):
        img = tf.numpy_function(load_image, [img_p], tf.float32)
        lbl = tf.numpy_function(load_label, [lbl_p], tf.float32)
        img.set_shape([output_size, output_size, 1])
        lbl.set_shape([output_size, output_size, 1])
        return img, lbl
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = data_generator(train_image_paths, train_label_paths, batch_size)
val_dataset = data_generator(val_image_paths, val_label_paths, batch_size)

# Define the network (similar to Lu's architecture)
inputSize = (256, 256, 1)
filterSize = 3
numFilters = 32

inputs = tf.keras.Input(shape=inputSize)
x = layers.Conv2D(numFilters, filterSize, dilation_rate=1, padding='same', kernel_initializer='he_normal')(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Conv2D(numFilters, filterSize, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Conv2D(numFilters, filterSize, dilation_rate=4, padding='same', kernel_initializer='he_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Conv2D(numFilters, filterSize, dilation_rate=4, padding='same', kernel_initializer='he_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Conv2D(numClasses, 1, padding='same')(x)  # Final layer, no activation, softmax in loss

# Softmax output
outputs = layers.Softmax()(x)

model = models.Model(inputs, outputs)

# Weighted loss function
def weighted_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true_onehot = tf.one_hot(y_true, depth=numClasses)
    weights = y_true_onehot[..., 0] * classWeights[0] + y_true_onehot[..., 1] * classWeights[1]
    loss = tf.nn.softmax_cross_entropy_with_logits(y_true_onehot, x)
    loss = loss * weights
    return tf.reduce_mean(loss)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=weighted_loss, 
              metrics=['accuracy'])

# Simulate training for 50 iterations, 4 epochs each
for training_iter in range(1, 51):
    if training_iter == 1:
        model.fit(train_dataset, epochs=MaxEpoch, validation_data=val_dataset, verbose=1)
    else:
        model.fit(train_dataset, epochs=MaxEpoch, validation_data=val_dataset, verbose=1)
    model.save(f'deeplab_tv9901_no_overlap_epoch{training_iter * MaxEpoch}.h5')
