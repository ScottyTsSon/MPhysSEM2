#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:05:57 2021

@author: josephcullen
"""
import numpy as np 
import tensorflow as tf 
import pickle
import keras 
import string 
import sys
from tensorflow import math, concat
from keras import layers, Input, Model, Sequential, optimizers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.metrics import Recall, Precision, FalseNegatives, FalsePositives
from keras.optimizers import SGD
from sys import exit
#from math import sigmoid

# In[]
# Parameters for MIT BIH arrhythmia database (NB input is a 128x128 CWT)
MITparams = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 6,
          'num_leads': 2,
          'shuffle': True}

# Parameters for PTB - XL ECG database (NB input is a 12 lead raw 10s ECG)
PTBparams = {'batch_size': 25,
          'n_classes': 6, # Not sure how many classes there are 
          'num_leads': 12,
          'lr_length': 1000,
          'hr_length': 5000,
          'shuffle': True}

# In[]
# Define architecture taken from: ECG-based multi-class arrhythmia detection using spatio-temporal
# attention-based convolutional recurrent neural network

'''input_shape = Input(shape=(MITparams['hr_length'],MITparams['num_leads']))
conv1 = Conv1D(filters = 64, kernel_size = 3)(input_shape)
conv2 = Conv1D(filters = 64, kernel_size = 3)(conv1)
maxpool1 = MaxPooling1D(pool_size = 3, strides = 3)(conv2) # Perhaps error due to 5000/3
dropout1 = Dropout(0.2)(maxpool1)'''

#INSERT SPATIO TEMPORAL

# In[]
#SPATIAl
def spatial_attention(input_layer = tf.Tensor, input_type = string):
    decrease_ratio = 1
    maxpool_input, avgpool_input = input_layer
    
    if input_type == 'CWT':
        spat_maxpool = MaxPooling2D(pool_size = (3,3), strides = 3)(maxpool_input)
        spat_avgpool = AveragePooling2D(pool_size = (3,3), strides = 3)(avgpool_input)
        spat_flatmax = spat_maxpool.Flatten()
        spat_flatavg = spat_avgpool.Flatten()
        spat_concat1 = concat([spat_flatmax,spat_flatavg])
        spat_dense1 = Dense(units = MITparams['num_leads']/decrease_ratio, activation = 'relu')(spat_concat1)
        spat_dense2 = Dense(units = MITparams['num_leads'], activation = 'relu')(spat_dense1)
        
    elif input_type == 'raw_ecg':
        spat_maxpool = MaxPooling1D(pool_size = 3, strides = 3)(maxpool_input)
        spat_avgpool = AveragePooling1D(pool_size = 3, strides = 3)(avgpool_input)
        spat_flatmax = spat_maxpool.Flatten()
        spat_flatavg = spat_avgpool.Flatten()
        spat_concat1 = concat([spat_flatmax,spat_flatavg])
        spat_dense1 = Dense(units = PTBparams['num_leads']/decrease_ratio, activation = 'relu')(spat_concat1)
        spat_dense2 = Dense(units = PTBparams['num_leads'], activation = 'relu')(spat_dense1)
        
    else: 
        print('SPATIAL ERROR: Not a valid input type')
        exit()
    
    #THIS IS PROBS NOT RIGHT:
    spat_maxout = spat_dense2(spat_dense1(spat_flatmax))
    spat_avgout = spat_dense2(spat_dense1(spat_flatavg))

    spat_sum = spat_maxout + spat_avgout
    spat_weights = tf.math.sigmoid(spat_sum)
    spat_refined = spat_weights * input_layer 
    return spat_refined


# In[]
#TEMPORAL
# NB temp in variable names here refers to temporal not temporary
# Temporal takes an input of a spatio refined feature therefore a product 
# of input image and spatial weights
def temporal_attention(input_layer = tf.Tensor, input_type = string):
    if input_type == 'CWT':
        temp_maxpool = MaxPooling2D(pool_size = (3,3), strides = 3)(input_layer)
        temp_avgpool = AveragePooling2D(pool_size = (3,3), strides = 3)(input_layer)
        temp_concat = concat([temp_maxpool,temp_avgpool], axis = -1) # MAYBE NEED TO SPECIFY AXIS
        temp_conv = Conv2D(filters = 1, kernel_size = (7,7), strides = 1)(temp_concat)       
    elif input_type == 'raw_ecg':
        temp_maxpool = MaxPooling1D(pool_size = 3, strides = 3)(input_layer)
        temp_avgpool = AveragePooling1D(pool_size = 3, strides = 3)(input_layer)
        temp_concat = concat([temp_maxpool,temp_avgpool], axis = -1) # MAYBE NEED TO SPECIFY AXIS
        temp_conv = Conv1D(filters = 1, kernel_size = 7, strides = 1)(temp_concat)   
    else:
        print('TEMPORAL ERROR: Not a valid input type.')
        exit()   

    temp_attention = tf.math.sigmoid(temp_conv)

    return temp_attention

# In[]
# Define the model
# Basically takes an input for the kind of network we are creating and then 
# creates that specific network.
def createModel(input_type = string):
    # CWT is used for the MIT 
    if input_type == 'CWT':

        # CONVOLUTION BLOCK 1        
        try:
            conv1 = Conv2D(32, kernel_size = 10, activation='relu', input_shape=(MITparams['dim'], MITparams['num_leads']))
            conv2 = Conv2D(32, kernel_size = 10, activation='relu')(conv1)
            maxpool1 = MaxPooling2D((2, 2))(conv2)
            dropout1 = Dropout(0.5)(maxpool1)
            spat_refined1 = spatial_attention(dropout1)
            temp_refined1 = temporal_attention(spat_refined1*dropout1)
        except:
            print('CONV BLOCK 1 ERROR')
            exit()
        
        # CONVOLUTION BLOCK 2
        try:
            conv3 = Conv2D(32, kernel_size = 8, activation = 'relu')(temp_refined1)
            conv4 = Conv2D(32, kernel_size = 4, activation = 'relu')(conv3)
            maxpool2 = MaxPooling2D((2,2))(conv4)
            dropout2 = Dropout(0.5)(maxpool2)
            spat_refined2 = spatial_attention(dropout2)
            temp_refined2 = temporal_attention(spat_refined2*dropout2)
            
            flattened = temp_refined2.Flatten()
            dense1 = Dense(256, activation='relu')(flattened)
            dropout3 = Dropout(0.5)(dense1)
            out = Dense(6, activation='softmax')(dropout3)
            model = Model(conv1, out)
        except:
            print('CONV BLOCK 2 ERROR')
            exit()
    
    elif input_type == 'raw_ecg':
        conv1 = Conv1D(32, kernel_size = 10, activation='relu', input_shape=(PTBparams['hr_length'], PTBparams['num_leads']))
        conv2 = Conv1D(32, kernel_size = 10, activation='relu')(conv1)
        maxpool1 = MaxPooling1D(kernel_size = 2)(conv2)
        dropout1 = Dropout(0.5)(maxpool1)
        spat_refined1 = spatial_attention(dropout1, input_type)
        temp_refined1 = temporal_attention(spat_refined1*dropout1, input_type)
        conv3 = Conv1D(32, kernel_size = 8, activation = 'relu')(temp_refined1*dropout1)
        conv4 = Conv1D(32, kernel_size = 4, activation = 'relu')(conv3)
        maxpool2 = MaxPooling2D(2)(conv4)
        dropout2 = Dropout(0.5)
        spat_refined2 = spatial_attention(dropout2)

        flattened = Flatten()
        dense1 = Dense(256, activation='relu')(flattened)
        dropout3 = Dropout(0.5)()
        out = Dense(6, activation='softmax')()
        model = Model(conv1, out)
        
    else: 
        print('MODEL ERROR: not a valid input for model.')
        exit(1)
        
    
    return model

# In[]

model_type = 'CWT'

sgd = SGD(lr=0.001, decay=0.000001, momentum=0.8, nesterov=True)
model = createModel(model_type)
# Got rid of mse and mae 
model.compile(optimizer='sgd', loss ='categorical_crossentropy', metrics=['precision', 
                                                                          'recall', 
                                                                          'false_negatives',
                                                                          'flase_positives',
                                                                          'categorical_accuracy']) 




