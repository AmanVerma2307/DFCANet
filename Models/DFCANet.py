####### Importing Libraries
import numpy as np
import tensorflow as tf
from FC_Conv import SC_Conv2D
from CAM import CAM_Module

####### DFCANet

###### Defining Layers

##### Defining Essentials
H = 224
W = 224

##### Residual Calibration Block

#### RC Block -1
SC_Conv11 = SC_Conv2D(128,(3,3),128,(11,11),(7,7))
SC_Conv12 = SC_Conv2D(128,(3,3),128,(11,11),(7,7))
SC_Conv13 = SC_Conv2D(128,(3,3),128,(11,11),(7,7))

#### RC Block -2
SC_Conv14 = SC_Conv2D(128,(3,3),128,(11,11),(7,7))
SC_Conv15 = SC_Conv2D(128,(3,3),128,(11,11),(7,7))
SC_Conv16 = SC_Conv2D(128,(3,3),128,(11,11),(7,7))

#### RC Block-3
Conv2_up = tf.keras.layers.Conv2D(filters=256,kernel_size=(1,1),padding='same',activation='relu')
SC_Conv21 = SC_Conv2D(256,(3,3),256,(9,9),(5,5))
SC_Conv22 = SC_Conv2D(256,(3,3),256,(9,9),(5,5)) 
SC_Conv23 = SC_Conv2D(256,(3,3),256,(9,9),(5,5))

#### RC Block4
SC_Conv24 = SC_Conv2D(256,(3,3),256,(9,9),(5,5))
SC_Conv25 = SC_Conv2D(256,(3,3),256,(9,9),(5,5)) 
SC_Conv26 = SC_Conv2D(256,(3,3),256,(9,9),(5,5))

#### RC Block-5
Conv3_up = tf.keras.layers.Conv2D(filters=512,kernel_size=(1,1),padding='same',activation='relu')
SC_Conv31 = SC_Conv2D(512,(3,3),512,(7,7),(3,3))
SC_Conv32 = SC_Conv2D(512,(3,3),512,(7,7),(3,3))
SC_Conv33 = SC_Conv2D(512,(3,3),512,(7,7),(3,3))

##### CAM Module
CAM_Module_1 = CAM_Module(512,1)

###### Defining Model

##### Base Module
pt_model = tf.keras.applications.DenseNet121(input_shape=(H,W,3),weights='imagenet',include_top=False)
pt_model.trainable = True  
pt_model_op = ((pt_model.layers)[51]).output

##### Residual Calibration Network

#### RC Block-1
SC_Conv11 = SC_Conv11(pt_model_op)
SC_Conv12 = SC_Conv12(SC_Conv11)
SC_Conv13 = SC_Conv13(SC_Conv12)
SC_Conv13 = tf.keras.layers.Add()([SC_Conv13,SC_Conv11])

#### RC Block-2
SC_Conv14 = SC_Conv14(SC_Conv13)
SC_Conv15 = SC_Conv15(SC_Conv14)
SC_Conv16 = SC_Conv16(SC_Conv15)
SC_Conv16 = tf.keras.layers.Add()([SC_Conv16,SC_Conv14])
SC_Conv16 = tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2))(SC_Conv16)

#### RC Block-3
Conv2_up = Conv2_up(SC_Conv16)
SC_Conv21 = SC_Conv21(Conv2_up)
SC_Conv22 = SC_Conv22(SC_Conv21)
SC_Conv23 = SC_Conv23(SC_Conv22)
SC_Conv23 = tf.keras.layers.Add()([SC_Conv23,SC_Conv21])

#### RC Block-4
SC_Conv24 = SC_Conv24(SC_Conv23)
SC_Conv25 = SC_Conv25(SC_Conv24)
SC_Conv26 = SC_Conv26(SC_Conv25)
SC_Conv26 = tf.keras.layers.Add()([SC_Conv26,SC_Conv24])
SC_Conv26 = tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2))(SC_Conv26)

#### RC Block-5
Conv3_up = Conv3_up(SC_Conv26)
SC_Conv31 = SC_Conv31(Conv3_up)
SC_Conv32 = SC_Conv32(SC_Conv31)
SC_Conv33 = SC_Conv33(SC_Conv32)
SC_Conv33 = tf.keras.layers.Add()([SC_Conv33,SC_Conv31])

##### CAM Module
CAM_Module_1 = CAM_Module_1(SC_Conv33)

##### Output Module
GAP_op = tf.keras.layers.GlobalAveragePooling2D()(CAM_Module_1)
Flattened_op = tf.keras.layers.Flatten()(GAP_op)
dense1 = tf.keras.layers.Dense(256,activation='relu')(Flattened_op)
dropout1 = tf.keras.layers.Dropout(rate=0.1)(dense1)
dense2 = tf.keras.layers.Dense(256,activation='relu')(dropout1)
dropout2 = tf.keras.layers.Dropout(rate=0.1)(dense2)
dense3 = tf.keras.layers.Dense(1,activation='sigmoid')(dropout2)

###### Building Model
model = tf.keras.models.Model(inputs=pt_model.input,outputs=dense3)