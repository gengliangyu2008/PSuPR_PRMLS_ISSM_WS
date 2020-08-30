#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:44:44 2019

@author: doppiomovimento
"""


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate


def createDualTwModel():
    shared1 = Conv2D(32,(3,3),padding='same',activation='relu')
    shared2 = Conv2D(48,(3,3),padding='same',activation='relu')
    shared3 = Conv2D(64,(3,3),padding='same',activation='relu')
    
    Lin = Input(shape=(32,32,3))
    Lx = Conv2D(16,(3,3),padding='same',activation='relu')(Lin)
    Lx = Conv2D(16,(3,3),padding='same',activation='relu')(Lx)
    Lx = MaxPooling2D(pool_size=(2,2))(Lx)
    Lx = shared1(Lx)
    Lx = MaxPooling2D(pool_size=(2,2))(Lx)
    Lx = shared2(Lx)
    Lx = MaxPooling2D(pool_size=(2,2))(Lx)
    Lx = shared3(Lx)
    Lx = MaxPooling2D(pool_size=(2,2))(Lx)
    
    Rin = Input(shape=(16,16,3))
    Rx = Conv2D(16,(3,3),padding='same',activation='relu')(Rin)
    Rx = shared1(Rx)
    Rx = MaxPooling2D(pool_size=(2,2))(Rx)
    Rx = shared2(Rx)
    Rx = MaxPooling2D(pool_size=(2,2))(Rx)
    Rx = shared3(Rx)
    Rx = MaxPooling2D(pool_size=(2,2))(Rx)
    
    x = concatenate([Lx,Rx],axis=-1)
    x = Flatten()(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(3,activation='softmax')(x)
    
    model = Model(inputs=[Lin,Rin],outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


modelDualTw = createDualTwModel()
modelDualTw.summary()




from tensorflow.keras.utils import plot_model

plot_model(modelDualTw, 
           to_file='DualTw_model.pdf', 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')