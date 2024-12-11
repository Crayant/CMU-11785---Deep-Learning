import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 假设您有以下函数用于构建deeplabv3+模型（需自行实现或引用现有代码）
# 此处仅为占位示例
def build_deeplabv3plus(image_size=(256,256,3), num_classes=2, backbone='resnet18'):
    # 请使用已有的deeplabv3+实现代码
    # 下方仅为一个空壳示例
    inputs = tf.keras.Input(shape=image_size)
    # ... deeplabv3+骨干网络和 ASPP模块等 ...
    x = tf.keras.layers.Conv2D(num_classes, (1,1), padding='same')(inputs) 
    outputs = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs, outputs)
    return model


imageDir = "ac3_EM_patch_256_overlap"
labelDir = "ac3_dbseg_images_bw_patch_new_256_overlap"

image_paths = sorted(glob.glob(os.path.join(imageDir, "*.png")))
label_paths = sorted(glob.glob(os.path.join(labelDir, "*.png")))

classNames = ["border","no_border"]
labelIDs   = [255, 0]

# 映射标签：255->1, 0->0
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)/255.0
    return img

def load_label(path):
    lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    lbl[lbl==255] = 1
    lbl[lbl==0] = 0
    lbl = lbl[..., np.newaxis]  # shape (H,W,1)
    return lbl

# 在MATLAB中： [train, val, test] = dividerand(ds.NumObservations,0.99,0.01,0)
# 这里实现类似的分割：99%训练，1%验证，无测试集
indices = np.arange(len(image_paths))
train_indices, val_indices = train_test_split(indices, test_size=0.01, random_state=42)
test_indices = []  # 无测试集

train_image_paths = [image_paths[i] for i in train_indices]
train_label_paths = [label_paths[i] for i in train_indices]
val_image_paths = [image_paths[i] for i in val_indices]
val_label_paths = [label_paths[i] for i in val_indices]

# countEachLabel替代：统计各类像素数
def count_pixels(label_list):
    pixel_counts = [0,0]  # 0类和1类像素计数
    image_pixel_count = []
    for lp in label_list:
        lbl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        total_pixels = lbl.size
        image_pixel_count.append(total_pixels)
        c0 = np.sum(lbl==0)
        c1 = np.sum(lbl==255)
        pixel_counts[0] += c0
        pixel_counts[1] += c1
    return pixel_counts, image_pixel_count

pixel_counts, image_pixel_count = count_pixels(train_label_paths)
numberPixels = sum(pixel_counts)
frequency = np.array(pixel_counts) / numberPixels
classWeights = 1.0 / frequency  # 与MATLAB一致：对频率小的类给予更高权重


pixel_counts_arr = np.array(pixel_counts)
image_pixel_count_arr = np.array(image_pixel_count)

# 一张图像中pixelCount是总像素数量中属于某类的像素数
# 简化处理，不同图像class频率平均下:
avg_image_size = np.mean(image_pixel_count_arr)
imageFreq = pixel_counts_arr / (len(train_label_paths)*avg_image_size)
median_val = np.median(imageFreq)
classWeights = median_val / imageFreq


# 在Python中需通过自定义损失函数实现加权
print("Class Weights:", classWeights)

load_name = 'MyNet_tv9901_epoch8.mat'
load_net = 1

# 无法直接从.mat加载keras模型，如已转换为.h5则：
if load_net == 1:
    # net = tf.keras.models.load_model('MyNet_tv9901_epoch8.h5', custom_objects={...})


    net = build_deeplabv3plus((256,256,3),2,"resnet18")
else:
    net = build_deeplabv3plus((256,256,3),2,"resnet18")

# 构建dataset
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

pximds_train = data_generator(train_image_paths, train_label_paths)
pximds_val = data_generator(val_image_paths, val_label_paths)

# 定义加权损失函数
def weighted_loss(y_true, y_pred):
    # y_true: (B,H,W,1), y_pred: (B,H,W,C)
    # 将y_true转换one-hot
    y_true = tf.cast(y_true, tf.int32)
    y_true_onehot = tf.one_hot(y_true, depth=numClasses)
    # classWeights与类顺序需对应，如0类no_border, 1类border
    # 根据上面处理classWeights即可
    weights_map = y_true_onehot[...,0]*classWeights[0] + y_true_onehot[...,1]*classWeights[1]
    # 使用logits计算loss需要y_pred是logits，此处y_pred是softmax输出需转logits or 用y_pred直接计算
    # 我们在build_deeplabv3plus中最后一层用softmax输出，需转log(y_pred)
    loss = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred+1e-7), axis=-1)
    loss = loss * weights_map
    return tf.reduce_mean(loss)

MaxEpoch = 4
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
net.compile(optimizer=optimizer, loss=weighted_loss, metrics=['accuracy'])

for training_iter in range(3,51):
    if training_iter == 1:
        net.fit(pximds_train, epochs=MaxEpoch, validation_data=pximds_val)
    else:
        net.fit(pximds_train, epochs=MaxEpoch, validation_data=pximds_val)
    net.save(f'MyNet_tv5050_no_overlap_epoch{training_iter*MaxEpoch}.h5')
