import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# parameters
load_net = 0
output_size = 256
imageSize = (output_size, output_size, 1)
numClasses = 2
encoderDepth = 5  # depth

# U-net function
def build_unet(input_shape=(256,256,1), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    
    # downup sampling
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)

    c4 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2,2))(c4)

    # bottom layer
    c5 = layers.Conv2D(1024, (3,3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3,3), activation='relu', padding='same')(c5)

    # updonw sampling
    u6 = layers.UpSampling2D((2,2))(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2,2))(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2,2))(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2,2))(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(num_classes, (1,1), activation='softmax')(c9)

    model = models.Model(inputs, outputs)
    return model

if load_net == 1:
    # loading
    # net = tf.keras.models.load_model('unet_batchsizes4_epoch25.h5')
    pass
else:
    # new model of unet
    net = build_unet(input_shape=imageSize, num_classes=numClasses)


# MATLAB中plot(lgraph)，在Python中用 net.summary()
net.summary()

# path
imageDir = "ac3_EM_patch_256"
labelDir = "ac3_dbseg_images_bw_patch_new_256"

# ID
classNames = ["border","no_border"]
labelIDs   = [255, 0]  # border=255, no_border=0

# EM image and lable
image_paths = sorted(glob.glob(os.path.join(imageDir, "*.png")))
label_paths = sorted(glob.glob(os.path.join(labelDir, "*.png")))

# define the correlated function
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)/255.0
    img = np.expand_dims(img, axis=-1)
    return img

def load_label(path):
    lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # 255 and 0 for two types
    lbl[lbl==255] = 1
    lbl[lbl==0] = 0
    lbl = np.expand_dims(lbl, axis=-1)
    return lbl

# MATLAB dividerand(ds.NumObservations,0.99,0.01,0) equal to：
indices = list(range(len(image_paths)))
train_idx, val_idx = train_test_split(indices, test_size=0.01, random_state=42)
test_idx = []  

train_image_paths = [image_paths[i] for i in train_idx]
train_label_paths = [label_paths[i] for i in train_idx]
val_image_paths = [image_paths[i] for i in val_idx]
val_label_paths = [label_paths[i] for i in val_idx]

# define tf.data.Dataset
def data_generator(img_paths, lbl_paths, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, lbl_paths))
    def _parse_function(img_p, lbl_p):
        img = tf.numpy_function(load_image, [img_p], tf.float32)
        lbl = tf.numpy_function(load_label, [lbl_p], tf.float32)
        img.set_shape([output_size, output_size, 1])
        lbl.set_shape([output_size, output_size, 1])
        return img, lbl
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(4).prefetch(tf.data.AUTOTUNE)
    return dataset

pximds_train = data_generator(train_image_paths, train_label_paths)
pximds_val = data_generator(val_image_paths, val_label_paths)



# in Keras using compile & fit
Maxepoch = 4
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
net.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


import time
start_time = time.time()
for i in range(1, 51):
    net.fit(pximds_train, epochs=Maxepoch, validation_data=pximds_val, verbose=1)
    # update in Python no need layerGraph(net)
    # save the model
    net.save(f'unet_tv9901_no_overlap_epoch{i*Maxepoch}.h5')
end_time = time.time()
print("Training completed in {:.2f} seconds.".format(end_time - start_time))
