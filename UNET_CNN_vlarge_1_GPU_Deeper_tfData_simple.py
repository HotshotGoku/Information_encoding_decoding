import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint , CSVLogger
from keras import backend as K

import cv2
from tensorflow.keras.callbacks import TensorBoard
import time

import os
from os import listdir
import numpy as np


import tensorflow as tf
import pickle


rfactor=128

img_length=rfactor
img_width=rfactor

# datadir_i="C:/Users/kinsh/Downloads/Test_test2final/Sim_input_3"

# datadir_i="K:/Large_set/Sim_input_6"



datadir_i='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_070723/Sim_input/simple'

path_i=os.path.join(datadir_i)

# for img_i in os.listdir(path_i):
#     img_i_array=cv2.imread(os.path.join(path_i,img_i),cv2.IMREAD_GRAYSCALE)

#     blurred = cv2.GaussianBlur(img_i_array, (7, 7), 0)
#     (T, img_i_array) = cv2.threshold(blurred, 200, 255,cv2.THRESH_BINARY )  #  | cv2.THRESH_OTSU
#     new_array_i=cv2.resize(img_i_array,(img_length,img_width))
    

#     # plt.imshow(img_i_array,cmap="gray")
#     # plt.imshow(new_array_i,cmap="gray")
#     # plt.show()
#     break


# output checking this against

# datadir_o="C:/Users/kinsh/Downloads/Test_test2final/Sim_output_3"
# datadir_o="K:/Large_set/Sim_output_6"

datadir_o='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_070723/Sim_output'

path_o=os.path.join(datadir_o)


# for img_o in os.listdir(path_o):
#     img_o_array=cv2.imread(os.path.join(path_o,img_o),cv2.IMREAD_GRAYSCALE)

   

#     new_array_o=cv2.resize(img_o_array,(img_length,img_width))

#     # blurred = cv2.GaussianBlur(img_o_array, (7, 7), 0)
#     # (T, img_o_array )= cv2.threshold(blurred, 10, 255,cv2.THRESH_BINARY)

#     # (T, img_o_array )= cv2.threshold(new_array_o, 90, 255,cv2.THRESH_BINARY_INV)

#     (T, new_array_o) = cv2.threshold(new_array_o, 0, 255,
# 	cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    
    
#     # plt.imshow(img_o_array,cmap="gray")
#     # plt.imshow(new_array_o,cmap="gray")

#     # plt.show()
#     break

r_factor=128

# change this later based on the performance 
img_length=r_factor
img_width=r_factor

training_data=[]
test_data=[]



def create_training_data():
    img_filenames_i = sorted(os.listdir(path_i))

    for img in img_filenames_i:

        img_array_i=cv2.imread(os.path.join(path_i,img),cv2.IMREAD_GRAYSCALE)
       
        # blurred = cv2.GaussianBlur(img_array_i, (7, 7), 0)
        # (T, img_array_i) = cv2.threshold(blurred, 200, 255,cv2.THRESH_BINARY)   # removing _INV
        new_array_i=cv2.resize(img_array_i,(img_length,img_width))
        

        training_data.append([new_array_i])
       

create_training_data()


def create_test_data():
    img_filenames_o = sorted(os.listdir(path_o))

    for img in img_filenames_o:
        img_array_o=cv2.imread(os.path.join(path_o,img),cv2.IMREAD_GRAYSCALE)
        new_array_o=cv2.resize(img_array_o,(img_length,img_width))
        # blurred = cv2.GaussianBlur(img_array_o, (7, 7), 0)
        (T, new_array_o) = cv2.threshold(new_array_o, 0, 255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)

        
        test_data.append([new_array_o])
       

create_test_data()

X=[]

y=[]

for features in training_data:
    X.append(features)

for features in test_data:
    y.append(features)

X=(np.array(X).reshape(-1,img_length,img_width,1)) #/255.0  # last index is grayscale first minus one is just select all x without spec. #images
y=(np.array(y).reshape(-1,img_length,img_width,1)) 

from datetime import datetime

currentSecond= datetime.now().second
currentMinute = datetime.now().minute
currentHour = datetime.now().hour

currentDay = datetime.now().day
currentMonth = datetime.now().month
currentYear = datetime.now().year

NAME =f"Pixel_128_U-NET_CNN_Model_Simple_v{currentMonth}{currentDay}_Cluster_GPU_tfData-{int(time.time())}"  # change this later to incorporate exact date 
tensorboard=TensorBoard(log_dir=f'logs/{NAME}')


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

example_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

def normalize_img(image, label):
  return (tf.cast(image, tf.float32) / 255.0, tf.cast(label, tf.float32) / 255.0)

example_dataset = example_dataset.map(normalize_img,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

train_dataset = train_dataset.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.cache()

train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.batch(512)

train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

test_dataset = test_dataset.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)



test_dataset = test_dataset.batch(512)

test_dataset = test_dataset.cache()

test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)


inputs = Input((r_factor, r_factor, 1))

# s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D((2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
p5 = MaxPooling2D(pool_size=(2, 2)) (c5)


c6 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p5)
c6 = Dropout(0.3) (c6)
c6 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)



u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c5])
c7 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)



u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c4])
c8 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.2) (c8)
c8 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c3])
c9 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.2) (c9)
c9 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c9)
u10 = concatenate([u10, c2])
c10 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u10)
c10 = Dropout(0.1) (c10)
c10 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c10)

u11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c10)
u11 = concatenate([u11, c1], axis=3)
c11 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u11)
c11 = Dropout(0.1) (c11)
c11 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c11)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

# filepath = f"{NAME}.h5"


folder_path=f"/hpc/group/youlab/ks723/miniconda3/saved_models/v{currentMonth}{currentDay}{currentHour}/{NAME}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
logger_filename=f"{NAME}.log"
logger_filepath=os.path.join(folder_path,logger_filename)

csv_logger = CSVLogger(logger_filepath, separator=',', append=False)    

# csvlog_filepath = os.path.join(folder_path, 'logs.csv')


# filepath = '{NAME}_epoch{epoch:03d}.h5'

filename = f"{NAME}_epoch{{epoch:03d}}.h5"

filepath = os.path.join(folder_path, filename)

# earlystopper = EarlyStopping(verbose=1)   #patience=10 removing patience to see the performance of the network later 

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                            save_best_only=False, mode='min',period=5)



callbacks_list = [checkpoint,tensorboard,csv_logger]  # earlystopper,

# history = model.fit_generator(generator=train_generator,
#                               validation_data=val_generator,
#                               epochs=10,
#                               callbacks=callbacks_list)
        

# model.fit_generator(generator=data_generator,validation_split=0.1, epochs=10, steps_per_epoch=len(X) // batch_size)

start = time.process_time()

history = model.fit(train_dataset, validation_data=test_dataset, epochs=1000, callbacks=callbacks_list) # def is 32 batch_size=32

print(time.process_time() - start)

# model.save(NAME)
