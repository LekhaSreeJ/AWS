#!/usr/bin/env python
# coding: utf-8

# import 

from tensorflow import keras
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import glob
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
# import seaborn as sns
# from skimage import io
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
# get_ipython().run_line_magic('matplotlib', 'inline')
import argparse
import time
import boto3

BUCKET_NAME = 'sagemaker-hi-data'
s3 = boto3.resource('s3')

def load_image(path):
    bucket = s3.Bucket(BUCKET_NAME)
    img = bucket.Object(path).get().get('Body').read()
    nparray = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)
    return nparray

def pos_neg_diagnosis(mask_path):
    
    bucket = s3.Bucket(BUCKET_NAME)
    img = bucket.Object(mask_path).get().get('Body').read()
    nparray = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)
    value = np.max(nparray)
    if value > 0 : 
        return 1
    return 0

def DataGenerator(img , mask):  
    
    bucket = s3.Bucket(BUCKET_NAME)
    
    emp = []
    
    for i in img:
        img_arr = bucket.Object(i).get().get('Body').read()
        image = cv2.imdecode(np.asarray(bytearray(img_arr)), cv2.IMREAD_COLOR)
        emp.append(image)

    masks = []
    for j in mask:
        mask_arr = bucket.Object(j).get().get('Body').read()
        mask = cv2.imdecode(np.asarray(bytearray(mask_arr)), cv2.IMREAD_COLOR)
        masks.append(mask)

    X = np.array(emp , dtype ='float') / 255.0
    y = np.array(masks , dtype = 'float') / 255.0
    
    masks = y[:, :, :, 0]
    
    y = keras.utils.to_categorical(masks,2, dtype ='float32')
    
    return X , y

def tversky(y_true, y_pred):
    smooth = 1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def resblock(X, f):

  # make a copy of input
  X_copy = X

  X = Conv2D(f, kernel_size = (1,1) ,strides = (1,1),kernel_initializer ='he_normal')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X) 

  X = Conv2D(f, kernel_size = (3,3), strides =(1,1), padding = 'same', kernel_initializer ='he_normal')(X)
  X = BatchNormalization()(X)

  X_copy = Conv2D(f, kernel_size = (1,1), strides =(1,1), kernel_initializer ='he_normal')(X_copy)
  X_copy = BatchNormalization()(X_copy)

  # Adding the output from main path and short path together
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X

def upsample_concat(x, skip):
  x = UpSampling2D((2,2))(x)
  merge = Concatenate()([x, skip])

  return merge

def get_model():

    input_shape = (256,256,3)

    # Input tensor shape
    X_input = Input(input_shape)

    # Stage 1
    conv1_in = Conv2D(16,3,activation= 'relu', padding = 'same', kernel_initializer ='he_normal')(X_input)
    conv1_in = BatchNormalization()(conv1_in)
    conv1_in = Conv2D(16,3,activation= 'relu', padding = 'same', kernel_initializer ='he_normal')(conv1_in)
    conv1_in = BatchNormalization()(conv1_in)
    pool_1 = MaxPool2D(pool_size = (2,2))(conv1_in)

    # Stage 2
    conv2_in = resblock(pool_1, 32)
    pool_2 = MaxPool2D(pool_size = (2,2))(conv2_in)

    # Stage 3
    conv3_in = resblock(pool_2, 64)
    pool_3 = MaxPool2D(pool_size = (2,2))(conv3_in)

    # Stage 4
    conv4_in = resblock(pool_3, 128)
    pool_4 = MaxPool2D(pool_size = (2,2))(conv4_in)

    # Stage 5 (Bottle Neck)
    conv5_in = resblock(pool_4, 256)

    # Upscale stage 1
    up_1 = upsample_concat(conv5_in, conv4_in)
    up_1 = resblock(up_1, 128)

    # Upscale stage 2
    up_2 = upsample_concat(up_1, conv3_in)
    up_2 = resblock(up_2, 64)

    # Upscale stage 3
    up_3 = upsample_concat(up_2, conv2_in)
    up_3 = resblock(up_3, 32)

    # Upscale stage 4
    up_4 = upsample_concat(up_3, conv1_in)
    up_4 = resblock(up_4, 16)

    # Final Output
    output = Conv2D(2, (1,1), padding = "same", activation = "sigmoid")(up_4)

    model = Model(inputs = X_input, outputs = output )

    return model

def main(args):
    
    data_map = []

    bucket = s3.Bucket(BUCKET_NAME)

    for obj in bucket.objects.filter(Prefix='ml_project1_data/'):

        key = obj.key    

        dir_name = key.split("/")[0]+"/"+key.split("/")[1]
        image_path = key

        data_map.extend([dir_name, image_path])

    df = pd.DataFrame({
        "patient_id" : data_map[::2],
        "path" : data_map[1::2]
    })

    df = df[1:]

    df_images = df[~df['path'].str.contains("mask")]
    df_masks = df[df['path'].str.contains("mask")]

    images = sorted(df_images["path"].values)
    masks = sorted(df_masks["path"].values)

    df_brain = pd.DataFrame({
        "patient_id": df_images.patient_id.values,
        "image_path": images,
        "mask_path": masks
    })

    df_brain["mask"] = df_brain["mask_path"].apply(lambda x: pos_neg_diagnosis(x))

    #split data

    brain_df_mask = df_brain[df_brain['mask'] == 1]

    X_train, X_val = train_test_split(brain_df_mask, test_size=0.15)
    X_test, X_val = train_test_split(X_val, test_size=0.5)

    train_ids = list(X_train.image_path)
    train_mask = list(X_train.mask_path)

    val_ids = list(X_val.image_path)
    val_mask= list(X_val.mask_path)
    
    training_generator = DataGenerator(train_ids,train_mask)
    validation_generator = DataGenerator(val_ids,val_mask)
    
    X , y = training_generator
    val_X , val_y = validation_generator
    
    model = get_model()
    
    adam = tf.keras.optimizers.Adam(lr = 0.001,epsilon = 0.1)
    
    model.compile(optimizer = adam, loss = focal_tversky, metrics = [tversky])

    earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=2)

    checkpointer = ModelCheckpoint(filepath="ResUNet-new-weights-30-sample.hdf5", verbose=0, save_best_only=True)

    history = model.fit(X , y, epochs=20,validation_data = (val_X , val_y), callbacks = [checkpointer, earlystopping])
    
    score = model.evaluate(val_X, val_y, verbose=0)
    
    
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])

    val_predictions = model.predict(val_X)
    
    print(val_predictions)
    
    model.save(f'{os.environ["SM_MODEL_DIR"]}/{time.strftime("%m%d%H%M%S", time.gmtime())}', save_format='tf')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir',        type=str)
    
    args = parser.parse_args()
    
    main(args)
