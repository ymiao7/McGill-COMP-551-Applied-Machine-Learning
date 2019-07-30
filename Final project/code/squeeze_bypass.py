# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:05:48 2019

@author: COMSOL
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import os
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout,BatchNormalization, Flatten, Dense
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D,add
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils
from keras import optimizers




batch_size = 64
data_augmentation = False
num_classes = 10
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


def fire_module(x, fire_id, squeeze=16, expand1=64,expand3=64,bypass_simple=False,bypass_complex=False,bypass_conv=0):
    s_id = 'fire' + str(fire_id) + '/'

    input_prev = x
    
    x = Convolution2D(squeeze, (1, 1), padding='same', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)
    x = BatchNormalization()(x)
    

    left = Convolution2D(expand1, (1, 1), padding='same', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)
    left = BatchNormalization()(left)
    
    
    right = Convolution2D(expand3, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)
    right = BatchNormalization()(right)
    

    output = concatenate([left, right], axis=3, name=s_id + 'concat')
    
    if bypass_simple:
        output = add([output,input_prev])
        
    if bypass_complex:
        y = Convolution2D(bypass_conv, (1, 1), padding='same')(input_prev)
        y = Activation('relu')(y)
        y = BatchNormalization()(y)
        output = add([output,y])
    
    return output






def SqueezeNet(input_shape=(32,32,3), classes=10):
    """Instantiates the SqueezeNet architecture.
    """

    img_input = Input(input_shape)
    x = Convolution2D(96, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = BatchNormalization()(x)
    #x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand1=64, expand3=64,bypass_complex=True,bypass_conv=128)
    x = fire_module(x, fire_id=3, squeeze=32, expand1=128, expand3=128,bypass_complex=True,bypass_conv=256)
    #x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand1=128, expand3=128,bypass_simple=True)
    x = fire_module(x, fire_id=5, squeeze=64, expand1=256, expand3=256,bypass_complex=True,bypass_conv=512)
    #x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    #x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=64, expand1=256, expand3=256,bypass_simple=True)
    
    
    x = Dropout(0.5, name='drop9')(x)
    
    
    x = Convolution2D(10, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    #x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='loss')(x)
    
    model = Model(inputs=img_input,outputs=x)

    return model


model = SqueezeNet()

model.compile(optimizer=keras.optimizers.adam(lr=0.001),loss=['categorical_crossentropy'],metrics=['accuracy'])

# Set a learning rate annealer
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)







    # This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.5,  # set range for random shear
        zoom_range=(0.9,1.1),  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
hist = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=32),
                        epochs=50,
                        validation_data=(x_test, y_test),
                        verbose = 1, 
                        steps_per_epoch=x_train.shape[0]/32,
                        callbacks=[learning_rate_reduction])

# =============================================================================
# # Save model and weights
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)
# 
# # Score trained model.
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
# =============================================================================
import matplotlib.pyplot as plt

print ('History', hist.history)
print(hist.history.keys())
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss')
plt.show()
