#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:35:11 2021

@author: josephcullen
"""

# In[] 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import pickle
import keras 
import string 
import sys
from Arrhythmia_generator import DataGenerator
from tensorflow import math, concat
from keras import layers, Input, Model, optimizers
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Multiply, Add, Softmax
from keras.metrics import Recall, Precision, FalseNegatives, FalsePositives
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from sys import exit

# In[ ]:

# Parameters
params = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 6,
          'n_channels': 2,
          'shuffle': True}

params_val = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 6,
          'n_channels': 2,
          'shuffle': False}

params_test = {'dim': (128,128),
          'batch_size': 1,
          'n_classes': 6,
          'n_channels': 2,
          'shuffle': False}
        
# In[]

input_img = Input(shape = (128,128,2), name = 'CWTimage')

# CONV BLOCK 1
conv1 = Conv2D(32, kernel_size = 10, activation='relu')(input_img)
conv2 = Conv2D(32, kernel_size = 10, activation='relu')(conv1)
maxpool1 = MaxPooling2D((2, 2))(conv2)
dropout1 = Dropout(0.5)(maxpool1)

# SPATIAL ATTENTION 1
spat_maxpool1 = MaxPooling2D(pool_size = (3,3), strides = 3, padding = 'valid')(dropout1)
spat_avgpool1 = AveragePooling2D(pool_size = (3,3), strides = 3, padding = 'valid')(dropout1)
spat_flatmax1 = Flatten()(spat_maxpool1)
spat_flatavg1 = Flatten()(spat_avgpool1)

dense1_nodenum = 256
dense2_nodenum = 32       
spat_dense1 = Dense(units = dense1_nodenum, activation = 'relu')
spat_dense2 = Dense(units = dense2_nodenum, activation = 'relu')#These are the spatial wei
        
spat_maxout1 = spat_dense2(spat_dense1(spat_flatmax1))
spat_avgout1 = spat_dense2(spat_dense1(spat_flatavg1))
        
spat_add1 = Add()([spat_maxout1,spat_avgout1])
spat_weights1 = Softmax()(spat_add1)
spat_refined1  = Multiply()([dropout1,spat_weights1])

# TEMPORAL ATTENTION 1
feature_height = 55
temp_conv1 = Conv2D(filters = 1, kernel_size = (feature_height,1), strides = 1, activation = 'relu', name = 'TEMPCONV')(spat_refined1)
temp_softmax1 = Softmax()(temp_conv1)
spattemp_refined1 = Multiply()([spat_refined1,temp_softmax1])

# CONV BLOCK 2
conv3 = Conv2D(32, kernel_size = 8, activation = 'relu')(spattemp_refined1)
conv4 = Conv2D(32, kernel_size = 4, activation = 'relu')(conv3)
maxpool2 = MaxPooling2D((2,2))(conv4)
dropout2 = Dropout(0.5)(maxpool2)

# SPATIAL ATTENTION 2
spat_maxpool2 = MaxPooling2D(pool_size = (3,3), strides = 3)(dropout2)
spat_avgpool2 = AveragePooling2D(pool_size = (3,3), strides = 3)(dropout2)
spat_flatmax2 = Flatten()(spat_maxpool2)
spat_flatavg2 = Flatten()(spat_avgpool2)

dense3_nodenum = 256
dense4_nodenum = 32
        
spat_dense3 = Dense(units = dense3_nodenum, activation = 'relu')
spat_dense4 = Dense(units = dense4_nodenum, activation = 'relu')
        
spat_maxout2 = spat_dense4(spat_dense3(spat_flatmax2))
spat_avgout2 = spat_dense4(spat_dense3(spat_flatavg2))
spat_add2 = Add()([spat_maxout2,spat_avgout2])
spat_weights2 = Softmax()(spat_add2)
spat_refined2  = Multiply()([dropout2,spat_weights2]) # FIX MULTIPLICATION

# FINAL PREDICTION LAYERS
flattened = Flatten()(spat_refined2)
dense1 = Dense(256, activation='relu')(flattened)
dropout3 = Dropout(0.5)(dense1)
out = Dense(6, activation='softmax')(dropout3)

    
model = Model(input_img, out, name = 'STCNN-MIT-CWT')
sgd = optimizers.SGD(lr=0.001, decay=0.000001, momentum=0.8, nesterov=True)
model.compile(optimizer=sgd, loss ='categorical_crossentropy', metrics=['mse', 'mae', 'categorical_accuracy'])

tf.keras.utils.plot_model(model,to_file="model.png", show_shapes=True)

# In[]
# Load in  file indices and labels
with open('CNN_data_label_OHE.pkl', 'rb') as f:
    file_labels = pickle.load(f)
file_indices = []
for i in range(12000):
    file_indices.append(i)
    
# Create rhe training and testing data    
X_final = file_indices
Y_final = file_labels
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.20)


# In[]
train_generator = DataGenerator(X_train, Y_train, **params)
val_generator = DataGenerator(X_test, Y_test, **params_val)


print('TRAINING: ')
history = model.fit_generator(generator = train_generator,
                              epochs=40, validation_data=val_generator,
                              verbose=True)
print('EVALUATING SCORES:')
scores_generator = DataGenerator(X_test, Y_test, **params_test)
flat_predictions = model.predict_generator(scores_generator,verbose = True)
print('PREDICTIONS:')
print(flat_predictions)
flat_labels = np.array(Y_test)
# In[]
model.save('MIT_CWT_STCNN') # CREATE A SavedModel file

# In[]
# Visualising data 
plt.figure()
plt.plot(history.history['accuracy'], 'cornflowerblue')
plt.plot(history.history['val_accuracy'], 'lightcoral')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('Accuracy plot.pdf', dpi = 1200)
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'], 'cornflowerblue')
plt.plot(history.history['val_loss'], 'lightcoral')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('Loss plot.pdf', dpi = 1200)


        
        
        
    