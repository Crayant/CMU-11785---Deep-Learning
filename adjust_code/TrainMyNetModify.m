import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import train_test_split

##############################################
# User-defined parameters
##############################################
imageDir = "ac3_EM_patch_256_overlap"
labelDir = "ac3_dbseg_images_bw_patch_new_256_overlap"
numClasses = 2
batch_size = 4
learning_rate = 1e-5
max_epoch = 4
train_iterations = 50
img_height, img_width = 256, 256

classWeights = None  # Will be computed based on dataset

##############################################
# Data loading functions
##############################################
def load_image(path):
    # Load image as RGB and normalize to [0,1]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) / 255.0
    return img

def load_label(path):
    # Load label in grayscale, map 255->1, 0->0
    lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    lbl = cv2.resize(lbl, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    lbl[lbl==255] = 1
    lbl[lbl==0] = 0
    lbl = lbl[..., np.newaxis]  # (H,W,1)
    return lbl

image_paths = sorted(glob.glob(os.path.join(imageDir, "*.png")))
label_paths = sorted(glob.glob(os.path.join(labelDir, "*.png")))

# Split dataset: 99% train, 1% val
indices = np.arange(len(image_paths))
train_indices, val_indices = train_test_split(indices, test_size=0.01, random_state=42)
train_image_paths = [image_paths[i] for i in train_indices]
train_label_paths = [label_paths[i] for i in train_indices]
val_image_paths = [image_paths[i] for i in val_indices]
val_label_paths = [label_paths[i] for i in val_indices]

##############################################
# Count class frequencies to compute class weights
##############################################
def count_pixels(label_list):
    count_0 = 0
    count_1 = 0
    for lp in label_list:
        lbl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        lbl = cv2.resize(lbl, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        count_1 += np.sum(lbl == 255)
        count_0 += np.sum(lbl == 0)
    return count_0, count_1

count_0, count_1 = count_pixels(train_label_paths)
numberPixels = count_0 + count_1
frequency_0 = count_0 / numberPixels
frequency_1 = count_1 / numberPixels
# Assign higher weight to minority class
classWeights = [1.0/frequency_1, 1.0/frequency_0]

print("Class Weights:", classWeights)

##############################################
# TensorFlow Dataset construction
##############################################
def data_generator(img_paths, lbl_paths, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, lbl_paths))
    def _parse_function(img_p, lbl_p):
        img = tf.numpy_function(load_image, [img_p], tf.float32)
        lbl = tf.numpy_function(load_label, [lbl_p], tf.float32)
        img.set_shape([img_height, img_width, 3])
        lbl.set_shape([img_height, img_width, 1])
        return img, lbl
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = data_generator(train_image_paths, train_label_paths, batch_size)
val_dataset = data_generator(val_image_paths, val_label_paths, batch_size)

##############################################
# Deeplabv3+ Implementation (ResNet-50 backbone)
##############################################
# The following is a representative implementation of Deeplabv3+:
# - We use a ResNet-50 backbone to extract features.
# - We add ASPP (Atrous Spatial Pyramid Pooling) on top of the backbone.
# - Then we add the Deeplabv3+ decoder.
#
# Note: For simplicity, no pretrained weights are loaded. In a real scenario,
# load pretrained ResNet weights and ensure proper alignment with BN layers.
##############################################

def ASPP(x, filters=256, rate_scale=1):
    # ASPP with different dilation rates
    # rate_scale is typically related to output stride
    atrous_rates = [6, 12, 18]
    y1 = layers.Conv2D(filters, 1, padding='same', dilation_rate=1*rate_scale)(x)
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.ReLU()(y1)

    y2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=atrous_rates[0]*rate_scale)(x)
    y2 = layers.BatchNormalization()(y2)
    y2 = layers.ReLU()(y2)

    y3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=atrous_rates[1]*rate_scale)(x)
    y3 = layers.BatchNormalization()(y3)
    y3 = layers.ReLU()(y3)

    y4 = layers.Conv2D(filters, 3, padding='same', dilation_rate=atrous_rates[2]*rate_scale)(x)
    y4 = layers.BatchNormalization()(y4)
    y4 = layers.ReLU()(y4)

    # Image-level features
    y_pool = layers.GlobalAveragePooling2D()(x)
    y_pool = layers.Reshape((1,1,-1))(y_pool)
    y_pool = layers.Conv2D(filters, 1, padding='same')(y_pool)
    y_pool = layers.BatchNormalization()(y_pool)
    y_pool = layers.ReLU()(y_pool)
    y_pool = tf.image.resize(y_pool, (tf.shape(x)[1], tf.shape(x)[2]))

    y = layers.Concatenate()([y1, y2, y3, y4, y_pool])
    y = layers.Conv2D(filters, 1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    return y

def DeeplabV3Plus(input_shape=(256,256,3), num_classes=2):
    # Backbone: ResNet-50
    base_model = applications.ResNet50(weights=None, include_top=False, input_shape=input_shape)
    # Extract feature maps
    # C1: low-level features from early stage
    x = base_model.input
    c1 = base_model.get_layer('conv2_block3_out').output  # Example: low-level features
    c4 = base_model.get_layer('conv4_block6_out').output  # High-level features
    c5 = base_model.get_layer('conv5_block3_out').output  # Final features

    # ASPP on c5
    aspp = ASPP(c5, 256, rate_scale=1)

    # Decoder
    # Reduce channel dimension of c1
    low_level = layers.Conv2D(48, 1, padding='same')(c1)
    low_level = layers.BatchNormalization()(low_level)
    low_level = layers.ReLU()(low_level)

    # Upsample ASPP output
    aspp_up = tf.image.resize(aspp, size=(tf.shape(c1)[1], tf.shape(c1)[2]))
    concat = layers.Concatenate()([aspp_up, low_level])
    x = layers.Conv2D(256, 3, padding='same')(concat)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Final upsample to original size
    x = tf.image.resize(x, (input_shape[0], input_shape[1]))
    x = layers.Conv2D(num_classes, 1, padding='same')(x)
    x = layers.Softmax()(x)

    model = models.Model(inputs=base_model.input, outputs=x)
    return model

net = DeeplabV3Plus((img_height, img_width, 3), numClasses)

##############################################
# Weighted loss function
##############################################
def weighted_loss(y_true, y_pred):
    # y_true: (B,H,W,1), y_pred: (B,H,W,C) with softmax probability
    y_true = tf.cast(y_true, tf.int32)
    y_true_onehot = tf.one_hot(y_true, depth=numClasses)
    weights_map = y_true_onehot[...,0]*classWeights[0] + y_true_onehot[...,1]*classWeights[1]
    loss = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred+1e-7), axis=-1)
    loss = loss * weights_map
    return tf.reduce_mean(loss)

##############################################
# Compile and train
##############################################
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
net.compile(optimizer=optimizer, loss=weighted_loss, metrics=['accuracy'])

for training_iter in range(1, train_iterations+1):
    net.fit(train_dataset, epochs=max_epoch, validation_data=val_dataset)
    net.save(f'MyNet_tv5050_no_overlap_epoch{training_iter*max_epoch}.h5')
