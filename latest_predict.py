import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import pandas as pd
import cv2
import numpy as np
from keras import backend as K
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use("ggplot")

#get_ipython().magic('matplotlib inline')

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


im_width = 480
im_height = 240
path_train = 'J:/DDSM/cancer_2/train/'
path_valid = 'J:/DDSM/cancer_2/valid/'
path_test = 'J:/DDSM/cancer_2/test/'


# train:A_1622_1.LEFT_MLO\B_3013_1.RIGHT_MLO\A_1468_1.RIGHT_CC\B_3041_1.LEFT_MLO\\A_1661_1.RIGHT_CC \\ C_0051_1.RIGHT_CC
# test:B_3498_1.LEFT_MLO\C_0066_1.RIGHT_CC
img_name = 'B_3068_1.RIGHT_MLO'  #B_3067_1.RIGHT_CC  B_3049_1.LEFT_CC  B_3065_1.RIGHT_CC  B_3109_1.RIGHT_CC  B_3412_1.LEFT_MLO  C_0064_1.LEFT_MLO
test_name = 'C_0360_1.LEFT_CC'
val_name = 'C_0229_1.RIGHT_MLO'  #C_0212_1.LEFT_CC  C_0229_1.RIGHT_MLO(注意力机制前后有所改进)
# img = load_img(path_train + '/images/image/' + img_name + '.png', color_mode="grayscale")
# # img_mask = load_img(path_train + '/masks/' + img_name + '.png', color_mode="grayscale")
# x_img = img_to_array(img)
x_img = cv2.imread(path_train + '/images/image/' + img_name + '.png',cv2.IMREAD_GRAYSCALE)
x_img = resize(x_img, (im_width, im_height, 1), mode='constant', preserve_range=True)/255
#x_img = x_img[20:(im_width-28), 8:(im_height-8)]
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory('J:/DDSM/cancer_2/train/images/', color_mode='grayscale',target_size=(im_width, im_height),
#                                                    batch_size=1,class_mode=None, shuffle=False)



x_show = cv2.imread(path_train + '/images/image/' + img_name + '.png')
x_show = resize(x_show, (im_width, im_height, 3), mode='constant', preserve_range=True)/255
#x_show = x_show[20:(im_width-28), 8:(im_height-8)]
#mask = img_to_array(img_mask)
mask = cv2.imread(path_train + '/masks/mask/' + img_name + '.png')
mask = resize(mask, (im_width, im_height, 3), mode='constant', preserve_range=True)/255
#mask = mask[20:(im_width-28), 8:(im_height-8)]

test_img = cv2.imread(path_test + '/image/' + test_name + '.png',cv2.IMREAD_GRAYSCALE)
test_img = resize(test_img, (im_width, im_height, 1), mode='constant', preserve_range=True)/255
#test_img = test_img[20:(im_width-28), 8:(im_height-8)]
test_show = cv2.imread(path_test + '/image/' + test_name + '.png')
test_show = resize(test_show, (im_width, im_height, 3), mode='constant', preserve_range=True)/255

test_mask = cv2.imread(path_test + '/mask/' + test_name + '.png')
test_mask = resize(test_mask, (im_width, im_height, 3), mode='constant', preserve_range=True)/255
#test_mask = test_mask[20:(im_width-28), 8:(im_height-8)]


val_img = cv2.imread(path_valid + '/images/image/' + val_name + '.png',cv2.IMREAD_GRAYSCALE)
val_img = resize(val_img, (im_width, im_height, 1), mode='constant', preserve_range=True)/255

val_show = cv2.imread(path_valid + '/images/image/' + val_name + '.png')
val_show = resize(val_show, (im_width, im_height, 3), mode='constant', preserve_range=True)/255


val_mask = cv2.imread(path_valid + '/masks/mask/' + val_name + '.png')
val_mask = resize(val_mask, (im_width, im_height, 3), mode='constant', preserve_range=True)/255



# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# In[15]:


def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

model = load_model('model-breast11_4-26.h5',custom_objects={'bce_dice_loss':bce_dice_loss,'dice_coef':dice_coef})

x_img1 = x_img.reshape(1, im_width, im_height, 1)
val_img1 = val_img.reshape(1, im_width, im_height, 1)
test_img1 = test_img.reshape(1,im_width, im_height, 1)
preds_train = model.predict(val_img1, verbose=1)
# test_generator.reset()
# pred = model.predict_generator(test_generator, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
# print(preds_train)
# plt.imshow(preds_train.squeeze())
# plt.show()
preds = np.zeros((im_width,im_height,3))
preds[:,:,0] = preds_train_t.squeeze()
preds[:,:,1] = preds_train_t.squeeze()
preds[:,:,2] = preds_train_t.squeeze()


plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
plt.imshow(val_show.squeeze())
plt.subplot(1,3,2)
plt.imshow(val_mask.squeeze())
plt.subplot(1,3,3)
plt.imshow(preds)
plt.show()