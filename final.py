import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from skimage.transform import resize
import keras
from keras import backend as K
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers import AveragePooling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import initializers


im_width = 480
im_height = 240
path_train = 'cancer_3/train/'
path_valid = 'cancer_3/valid/'
path_test = 'cancer_3/test/'
SEED = 42


def get_data(path, train=True):
    ids = next(os.walk(path + "/images/image/"))[2]  # 获取path+images目录下的所有文件名
    X = np.zeros((len(ids), im_width, im_height, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_width, im_height, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in enumerate(ids):
        # Load images
        img = cv2.imread(path + '/images/image/' + id_, cv2.IMREAD_GRAYSCALE)
        x_img = resize(img, (im_width, im_height, 1), mode='constant', preserve_range=True)
        #x_img = np.reshape(x_img, (im_width, im_height, 1))

        # Load masks
        if train:
            mask = cv2.imread(path + '/masks/mask/' + id_, cv2.IMREAD_GRAYSCALE)
            mask = resize(mask, (im_width, im_height, 1), mode='constant', preserve_range=True)
            #mask = np.reshape(mask, (im_width, im_height, 1))

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X


x_data, y_data = get_data(path_train, train=True)


batch_size = 8


def get_train_test_augmented(X_data=x_data, Y_data=y_data, validation_split=0.2, batch_size=batch_size, seed=SEED):
    X_train, X_test, Y_train, Y_test = train_test_split(X_data,
                                                        Y_data,
                                                        train_size=1 - validation_split,
                                                        test_size=validation_split,
                                                        random_state=seed)
    # Image data generator distortion options
    data_gen_args = dict(rotation_range=10.,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='nearest')  # use 'constant'??

    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)

    # Test data, no data augmentation, but we create a generator anyway
    X_datagen_val = ImageDataGenerator()
    Y_datagen_val = ImageDataGenerator()
    X_datagen_val.fit(X_test, augment=True, seed=seed)
    Y_datagen_val.fit(Y_test, augment=True, seed=seed)
    X_test_augmented = X_datagen_val.flow(X_test, batch_size=batch_size, shuffle=True, seed=seed)
    Y_test_augmented = Y_datagen_val.flow(Y_test, batch_size=batch_size, shuffle=True, seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)
    test_generator = zip(X_test_augmented, Y_test_augmented)

    return train_generator, test_generator, X_train, X_test, Y_train, Y_test


from sklearn.model_selection import train_test_split
train_generator, test_generator, X_train, X_val, Y_train, Y_val = get_train_test_augmented(X_data=x_data, Y_data=y_data, validation_split=0.2, batch_size=batch_size)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


from keras.layers import UpSampling2D, multiply
from keras.layers import AveragePooling2D

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               bias_initializer='zeros', padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               bias_initializer='zeros', padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def DenseBlock(channels,inputs):

    inputs = BatchActivate(inputs)
    conv1 = Conv2D(channels, (3, 3), kernel_initializer="he_normal",
               bias_initializer='zeros', padding='same')(inputs)

    conv2 =BatchActivate(conv1)
    conv2 = Conv2D(channels//4, (3, 3), kernel_initializer="he_normal",
               bias_initializer='zeros', padding='same')(conv2)

    concat1 = concatenate([conv1,conv2])

    conv3 = BatchActivate(concat1)
    conv3 = Conv2D(channels // 4, (3, 3), kernel_initializer="he_normal",
                   bias_initializer='zeros', padding='same')(conv3)

    concat2 = concatenate([conv1, conv2,conv3])

    conv4 = BatchActivate(concat2)
    conv4 = Conv2D(channels//4, (3, 3), kernel_initializer="he_normal",
               bias_initializer='zeros', padding='same')(conv4)

    concat3 = concatenate([conv1,conv2,conv3,conv4])

    conv5 = BatchActivate(concat3)
    conv5 = Conv2D(channels // 4, (3, 3), kernel_initializer="he_normal",
                   bias_initializer='zeros', padding='same')(conv5)

    result = concatenate([conv1, conv2, conv3, conv4, conv5])
    return result


def expend_as(tensor, rep, name):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},
                       name='psi_up' + name)(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape, name):  # g是更深层，x是较浅层

    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), kernel_initializer="he_normal",
                     bias_initializer='zeros', strides=(2, 2), padding='same', name='xl' + name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), kernel_initializer="he_normal",
                   bias_initializer='zeros', padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same', name='g_up' + name)(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), kernel_initializer="he_normal",
                 bias_initializer='zeros', padding='same', name='psi' + name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3], name)
    y = multiply([upsample_psi, x], name='q_attn' + name)

    result = Conv2D(shape_x[3], (1, 1), kernel_initializer="he_normal",
                    bias_initializer='zeros', padding='same', name='q_attn_conv' + name)(y)
    result_bn = BatchNormalization(name='q_attn_bn' + name)(result)
    return result_bn


def UnetGatingSignal(input, is_batchnorm, name):
    ''' this is simply 1x1 convolution, bn, activation '''
    shape = K.int_shape(input)
    x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same", kernel_initializer='he_normal',
               name=name + '_conv')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_act')(x)
    return x


def get_unet(input_img, n_filters=32, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    #p1 = Dropout(0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def attention_unet(input_img, n_filters=32, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    # p1 = Dropout(0.8)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.6)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    g1 = UnetGatingSignal(c5, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(c4, g1, n_filters * 8, '_1')
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same',
                         activation='relu', kernel_initializer='he_normal')(c5)
    u6 = concatenate([u6, attn1], name='u6')
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    g2 = UnetGatingSignal(c6, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(c3, g2, n_filters * 4, '_2')
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same',
                         activation='relu', kernel_initializer='he_normal')(c6)
    u7 = concatenate([u7, attn2], name='u7')
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    g3 = UnetGatingSignal(c7, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(c2, g3, n_filters * 2, '_3')
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same',
                         activation='relu', kernel_initializer='he_normal')(c7)
    u8 = concatenate([u8, attn3], name='u8')
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    g4 = UnetGatingSignal(c8, is_batchnorm=True, name='g4')
    attn4 = AttnGatingBlock(c1, g4, n_filters * 1, '_4')
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same',
                         activation='relu', kernel_initializer='he_normal')(c8)
    u9 = concatenate([u9, attn4], axis=3, name='u9')
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros',
                     name='final')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def dense_unet(input_img, n_filters=32, dropout=0.6, batchnorm=True):
    # contracting path
    c1 = Conv2D(n_filters*1, (3, 3), kernel_initializer="he_normal",
                   bias_initializer='zeros', padding='same')(input_img)
    c1 = DenseBlock(n_filters*1,c1)
    c1_1 = Conv2D(n_filters*1, (1, 1), kernel_initializer="he_normal",
                   bias_initializer='zeros', padding='same')(c1)
    p1 = AveragePooling2D((2, 2))(c1_1)
    #p1 = Dropout(0.5)(p1)

    c2 = DenseBlock(n_filters*2,p1)
    c2_1 = Conv2D(n_filters*2, (1, 1), kernel_initializer="he_normal",
                   bias_initializer='zeros', padding='same')(c2)
    p2 = AveragePooling2D((2, 2))(c2_1)
    p2 = Dropout(0.6)(p2)

    c3 = DenseBlock(n_filters*4,p2)
    c3_1 = Conv2D(n_filters*4, (1, 1), kernel_initializer="he_normal",
                   bias_initializer='zeros', padding='same')(c3)
    p3 = AveragePooling2D((2, 2))(c3_1)
    p3 = Dropout(0.6)(p3)

    c4 = DenseBlock(n_filters*8,p3)
    c4_1 = Conv2D(n_filters*8, (1, 1), kernel_initializer="he_normal",
                   bias_initializer='zeros', padding='same')(c4)
    p4 = AveragePooling2D((2, 2))(c4_1)
    p4 = Dropout(0.6)(p4)

    c5 = DenseBlock(n_filters*8,p4)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = Activation("relu")(u6)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = Activation("relu")(u7)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = Activation("relu")(u8)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = Activation("relu")(u9)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def attention_dense_unet(input_img, n_filters=32, dropout=0.6, batchnorm=True):
    # contracting path
    c1 = Conv2D(n_filters * 1, (3, 3), kernel_initializer="he_normal",
                bias_initializer='zeros', padding='same')(input_img)
    c1 = DenseBlock(n_filters * 1, c1)
    c1_1 = Conv2D(n_filters * 1, (1, 1), kernel_initializer="he_normal",
                  bias_initializer='zeros', padding='same')(c1)
    p1 = AveragePooling2D((2, 2))(c1_1)
    # p1 = Dropout(0.8)(p1)

    c2 = DenseBlock(n_filters * 2, p1)
    c2_1 = Conv2D(n_filters * 2, (1, 1), kernel_initializer="he_normal",
                  bias_initializer='zeros', padding='same')(c2)
    p2 = AveragePooling2D((2, 2))(c2_1)
    p2 = Dropout(dropout)(p2)

    c3 = DenseBlock(n_filters * 4, p2)
    c3_1 = Conv2D(n_filters * 4, (1, 1), kernel_initializer="he_normal",
                  bias_initializer='zeros', padding='same')(c3)
    p3 = AveragePooling2D((2, 2))(c3_1)
    p3 = Dropout(dropout)(p3)

    c4 = DenseBlock(n_filters * 8, p3)
    c4_1 = Conv2D(n_filters * 8, (1, 1), kernel_initializer="he_normal",
                  bias_initializer='zeros', padding='same')(c4)
    p4 = AveragePooling2D((2, 2))(c4_1)
    p4 = Dropout(dropout)(p4)

    c5 = DenseBlock(n_filters * 8, p4)  #

    # expansive path
    g1 = UnetGatingSignal(c5, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(c4_1, g1, n_filters * 16, '_1')
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same',
                         activation='relu', kernel_initializer='he_normal')(c5)
    u6 = concatenate([u6, attn1], name='u6')
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    g2 = UnetGatingSignal(c6, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(c3_1, g2, n_filters * 8, '_2')
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same',
                         activation='relu', kernel_initializer='he_normal')(c6)
    u7 = concatenate([u7, attn2], name='u7')
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    g3 = UnetGatingSignal(c7, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(c2_1, g3, n_filters * 4, '_3')
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same',
                         activation='relu', kernel_initializer='he_normal')(c7)
    u8 = concatenate([u8, attn3], name='u8')
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    g4 = UnetGatingSignal(c8, is_batchnorm=True, name='g4')
    attn4 = AttnGatingBlock(c1_1, g4, n_filters * 2, '_4')
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same',
                         activation='relu', kernel_initializer='he_normal')(c8)
    u9 = concatenate([u9, attn4], axis=3, name='u9')
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros',
                     name='final')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


input_img = Input((im_width, im_height, 1), name='img')
model = dense_unet(input_img, n_filters=32, dropout=0.6, batchnorm=True)

model.compile(optimizer=Adam(0.001),loss=bce_dice_loss, metrics=[dice_coef])
model.summary()


callbacks = [
    EarlyStopping(patience=20, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-breast11_4-28.h5', verbose=1, save_best_only=True, save_weights_only=False)
]


H = model.fit_generator(train_generator,
                    validation_data=test_generator,
                    validation_steps=len(X_val)/batch_size,
                    steps_per_epoch=len(X_train)/(batch_size),
                    epochs=100,
                    callbacks=callbacks)


print(H.history.keys())
plt.plot(H.history['dice_coef'])
plt.plot(H.history['val_dice_coef'])
plt.title('model dice_coef')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig("accuracy4-28.png")


plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model bce_dice_loss')
plt.ylabel('bce_dice_loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig("loss4-28.png")