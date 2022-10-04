#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: francesco

Re-implementation of the notebook referred to the 2019 paper by Chapman et al. 
(https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GL083662)

The purpose is to build a ConvNet able to postprocess the IVT (Integrated Vapour Transport)
images derived from NWP models given a set of ground truth IVT values derived from measurement.

## Schema ##

PREDICTED IMAGE --> ConvNet --> CORRECTED IMAGE


"""

import tensorflow as tf

import xarray as xr

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# IMPORTING THE DATA +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
AllDat = xr.open_zarr('All_Zarr/GFSIVT_F006_zarr')


# SPLIT DATASET IN: TRAIN, VALIDATION E TEST +++++++++++++++++++++++++++++++++
Dat_Training = AllDat.loc[dict(forecast_reference_time=slice('2006-10-09T12:00:00.000000000','2016-04-09T12:00:00.000000000'))]
Dat_Validation = AllDat.loc[dict(forecast_reference_time=slice('2016-10-09T12:00:00.000000000','2017-04-09T12:00:00.000000000'))]
Dat_Testing = AllDat.loc[dict(forecast_reference_time=slice('2017-10-09T12:00:00.000000000','2018-04-09T12:00:00.000000000'))]


'''
#single pixel selection
Dat_Training = Dat_Training.loc[dict(lat=slice(38,38),lon=slice(-121.875,-121.875))]
Dat_Validation = Dat_Validation.loc[dict(lat=slice(38,38),lon=slice(-121.875,-121.875))]
Dat_Testingg = Dat_Testing.loc[dict(lat=slice(38,38),lon=slice(-121.875,-121.875))]

#zone pixel selection
Dat_Training = Dat_Training.loc[dict(lat=slice(38,58),lon=slice(-141.875,-121.875))]
Dat_Validation = Dat_Validation.loc[dict(lat=slice(38,58),lon=slice(-141.875,-121.875))]
Dat_Testing = Dat_Testing.loc[dict(lat=slice(38,58),lon=slice(-141.875,-121.875))]
ì
# divisione dei set in input e labels
x_tr = np.transpose(Dat_Training.IVT.values.squeeze())
y_tr = Dat_Training.IVTm.values.squeeze()

x_v = Dat_Validation.IVT.values.squeeze()
y_v = Dat_Validation.IVTm.values.squeeze()

x_te = Dat_Testing.IVT.values.squeeze()
y_te = Dat_Testing.IVTm.values.squeeze()

# SINGOLO PIXEL: divisione dei set in input e labels - RESHAPING INCAPSULANDO OGNI PIXEL IN UN MINI VETTORE
x_tr = tf.reshape(Dat_Training.IVT.values, (6362,1))
y_tr = tf.reshape(Dat_Training.IVTm.values, (6362,1))
'''

# RESHAPING THE TRAINING SET IN ORDER TO OBTAIN A TIME SERIES (6362 epochs) OF IVT and IVTm WRT lat and lon
x_tr = tf.reshape(Dat_Training.IVT.values.squeeze(), (6362,101,113,1))
y_tr = tf.reshape(Dat_Training.IVTm.values.squeeze(), (6362,101,113,1))

x_v = tf.reshape(Dat_Validation.IVT.values.squeeze(), (365,101,113,1))
y_v = tf.reshape(Dat_Validation.IVTm.values.squeeze(), (365,101,113,1))

# RESHAPING THE TEST SET IN ORDER TO OBTAIN A TIME SERIES (365 epochs) OF IVT and IVTm WRT lat and lon
x_te = tf.reshape(Dat_Testing.IVT.values.squeeze(), (365,101,113,1))
y_te = tf.reshape(Dat_Testing.IVTm.values.squeeze(), (365,101,113,1))


'''
+++++ NOTA ++++++

6362 + 365 + 365 non somma a 8108 (numero di dati totali) perchè una il paper
sembrava saltare dei periodi. Anche nel codice sono stati saltati 
(considerati solo i periodi Ottobre Aprile)

+++++++++++++++++
'''



# BUILDING THE CONVOLUTIONAL NEURAL NETWORK ++++++++++++++++++++++++++++++++++
model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten(input_shape=(41,33,1)),
    tf.keras.layers.Conv2D(10,3,activation='relu',input_shape=(101,113,1),padding='same'),
    tf.keras.layers.Conv2D(8,3,activation='relu',padding='same'),
    tf.keras.layers.Conv2D(6,3,activation='relu',padding='same'),
    tf.keras.layers.Conv2D(1,3,activation='relu',padding='same'),
    #tf.keras.layers.Dense(1353, activation='relu'),
    #tf.keras.layers.Reshape((41,33,1))
    ])

# PRINTING THE SUMMARY OF THE MODEL ++++++++++++++++++++++++++++++++++++++++++
print(model.summary())

# SETTING UP LOSS AND OPTIMIZER IN ORDER TO COMPILE THE MODEL ++++++++++++++++
loss = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# COMPILE THE MODEL ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
model.compile(loss=loss,optimizer=optimizer, metrics=['mean_squared_error'])

# TRAINING OF THE MODEL ++++++++++++++++++++++++++++++++++++++++++++++++++++++
training = model.fit(x=x_tr,y=y_tr,epochs=2, validation_data=[x_v,y_v], batch_size=8)

'''
+++++ NOTA ++++++

i validation data vengono usati (linea sopra)

+++++++++++++++++
'''


# EVALUATION OF THE MODEL WITH THE TEST SET ++++++++++++++++++++++++++++++++++
model.evaluate(x_te, y_te)

# SAVE THE CORRECED IMAGES OF THE TEST SET IN ORDER TO CREATE THE PLOT +++++++
corrected_images = model.predict(x_te)


# PLOT OF IMAGES: 
    # - Original, Corrected, Difference between the two
    # - Ground truth, Corrected, Diffecence between the two
for i in range(len(x_te)):
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
    ax1.imshow(np.flip(x_te[i], axis=1))
    ax1.set_title('Original')
    ax2.imshow(np.flip(corrected_images[i], axis=1))
    ax2.set_title('Corrected')
    ax3.imshow(np.flip(x_te[i] - corrected_images[i], axis=1))
    ax3.set_title('Difference')
    ax4.imshow(np.flip(y_te[i], axis = 1))
    ax4.set_title('Ground truth')  
    ax5.imshow(np.flip(corrected_images[i], axis = 1))
    ax5.set_title('Corrected')
    ax6.imshow(np.flip(y_te[i] - corrected_images[i], axis=1))
    ax6.set_title('Difference')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

