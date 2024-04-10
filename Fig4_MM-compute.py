#!/usr/bin/env python
# coding: utf-8

# In[15]:

import math
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import struct
import cartopy
from cartopy import crs as ccrs
import matplotlib 
from matplotlib import pyplot as plt
import os
from os.path import join, exists
from os import mkdir
import scipy
import netCDF4
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
import pandas as pd
import matplotlib.path as mpath
from matplotlib.colors import TwoSlopeNorm 
from sklearn.model_selection import train_test_split
from scipy.fft import fft, ifft
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import pickle

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
import MM_util_AI
import keras
import random

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
#sess = tf.compat.v1.Session(config=config) 
#K.set_session(sess)
physical_devices = tf.config.list_physical_devices('GPU') 
print(physical_devices)
with open("/scratch/hz1994/blocking/data_MMmodel/filepath.txt","r") as fi:
    for ln in fi:
        if ln.startswith("Reduced_dim_variables"):
            rd_path=ln.strip().split('\t')[1]
        if ln.startswith("TMindex_filepath"):
            TMindex_path=ln.strip().split('\t')[1]   
        if ln.startswith("dimensionalized_filepath"):
            dim_path=ln.strip().split('\t')[1]   
        if ln.startswith("nondimensionalized_filepath"):
            nondim_path=ln.strip().split('\t')[1]
        if ln.startswith("conditionedT_filepath" ):
            train_path=ln.strip().split('\t')[1]
        if ln.startswith("model_filepath" ):
            models_path=ln.strip().split('\t')[1]
        if ln.startswith("fig_filepath" ):
            fig_path=ln.strip().split('\t')[1]            
            
print(rd_path)
print(TMindex_path)
print(dim_path)
print(train_path)
print(models_path)
print(fig_path)


def print_parameters():
    print("subname=",subname)
    print("cnnsize=",cnnsize)
    print("regularize=",regularize)
    print("epsilon=",epsilon)
    print("data_amount=",data_amount)
    print("random_seed=",random_seed)
    print("EPOCH=",EPOCH)
    print("subname=",subname)
    print("name=",name)
    print("save_dir=",models_path_setA+'models_T_%d_committor_'%T+name+subname)
    print("params_train=",params_train)
    print("params_test=", params_test)

# In[2]:


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, choleskyL ,datatype, batch_size=64, dim=(18,90), n_channels=3,
                 n_classes=2, epsilon=0.01, data_amount='full', regularize=False, shuffle=True):
        'Initialization'
        list_IDs_0=[k for k, v in labels.items() if np.all(v == np.array([True,False]))]
        list_IDs_1=[k for k, v in labels.items() if np.all(v == np.array([False,True]))]
        
        self.dim = dim
        self.batch_size = batch_size
        self.bbatch_size = batch_size//2
        self.labels = labels
        self.list_IDs_0 = list_IDs_0
        self.list_IDs_1 = list_IDs_1
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.epsilon=epsilon
        self.data_amount=data_amount
        self.choleskyL=choleskyL
        self.datatype=datatype
        self.regularize=regularize
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(max(len(self.list_IDs_0),len(self.list_IDs_1)) / self.bbatch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes=np.arange(index*self.bbatch_size,(index+1)*self.bbatch_size)
        indexes_0 = self.indexes_0[indexes%len(self.indexes_0)]
        indexes_1 = self.indexes_1[indexes%len(self.indexes_1)]
        
        # Find list of IDs
        list_IDs_temp_0 = [self.list_IDs_0[k] for k in indexes_0]
        list_IDs_temp_1 = [self.list_IDs_1[k] for k in indexes_1]

        list_IDs_temp=list_IDs_temp_0+list_IDs_temp_1
        random.shuffle(list_IDs_temp)
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes_0 = np.arange(len(self.list_IDs_0))
        self.indexes_1 = np.arange(len(self.list_IDs_1))
        if self.shuffle == True:
            np.random.shuffle(self.indexes_0)
            np.random.shuffle(self.indexes_1)
        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,2), dtype=bool)

        if self.datatype=='train':
            path='/scratch/hz1994/blocking/data_MMmodel/conditionT/T/data_X_T1_1250k_lowpass3dys_proc_training_1000k/'
        elif self.datatype=='test':
            path='/scratch/hz1994/blocking/data_MMmodel/conditionT/T/data_X_T1_1250k_lowpass3dys_proc_test/'
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(path+'dataX_' + ID + '.npy')
            # Store class
            y[i] = self.labels[ID]

        if self.regularize==True:
            rand=np.random.normal(0,self.epsilon,size=(self.choleskyL.shape[0],self.batch_size))
            X=X+(self.choleskyL@rand).T.reshape(X.shape)
        return X, y
# In[8]:


T=1
EPOCH=30
name="1250k_lowpass3dys"
setting=sys.argv
cnnsize=str(setting[1])
regularize=setting[2].lower() == 'true'
epsilon=float(setting[3])
data_amount=str(setting[4])
random_seed=int(setting[5])
choleskyL=np.load('/scratch/hz1994/blocking/data_MMmodel/conditionT/T/cholesky_L.npy') 

if regularize==True:
    subname="/batches_resampling_"+"data_"+data_amount+"_"+cnnsize+"_cnn_"+"regularize"+"_%.3e"%epsilon+"_rs_%d/"%random_seed
else:
    subname="/batches_resampling_"+"data_"+data_amount+"_"+cnnsize+"_cnn_/"

setA="T"
train_path_setA=train_path+setA+"/"
models_path_setA=models_path+setA+"/"

if not os.path.exists(models_path_setA+'models_T_%d_committor_'%T+name+subname):
    print(models_path_setA+'models_T_%d_committor_'%T+name+subname)
    os.makedirs(models_path_setA+'models_T_%d_committor_'%T+name+subname)

params_train = {'dim': (18,90),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 3,
          'epsilon':epsilon,
          'choleskyL':choleskyL,
          'data_amount':data_amount,
          'regularize':regularize,
          'shuffle': True,
            'datatype':"train"}

params_test = {'dim': (18,90),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 3,
          'epsilon':epsilon,
          'choleskyL':choleskyL,
          'data_amount':data_amount,
          'regularize':regularize,
          'shuffle': True,
          'datatype':"test"}

print_parameters()
Ysparse=np.load(train_path_setA+"/full/"+"data_Y_T%d_"%T+"1250k_lowpass3dys"+".npy")
# Ysparse=np.load(train_path_setA+data_amount+"/"+"data_Y_T%d_"%T+name+".npy")
# indices=np.arange(Ysparse.shape[0])
Y=np.zeros((Ysparse.size,2)).astype(bool)
Y[:,1][Ysparse==1]=True  #blocking
Y[:,0][Ysparse==0]=True

# rest_idx, test_idx  = train_test_split(indices, test_size=0.1, random_state=random_seed)
# test_labels=Ysparse[ test_idx  ]
# train_idx, val_idx  = train_test_split(rest_idx, test_size=0.1, random_state=random_seed)
# train_labels=Ysparse[train_idx]
# validation_labels=Ysparse[val_idx]

# np.save(models_path_setA+'models_T_%d_committor_'%T+name+subname+"test_idx", test_idx )
# np.save(models_path_setA+'models_T_%d_committor_'%T+name+subname+"train_idx", train_idx )
# np.save(models_path_setA+'models_T_%d_committor_'%T+name+subname+"val_idx", val_idx )

# Datasets
if data_amount=="10.0k":
    time=10000
if data_amount=="100.0k":
    time=100000
if data_amount=="1000.0k":   
    time=1000000
T_all=xr.open_dataarray("/scratch/hz1994/blocking/data_MMmodel/DGindex/Atl_Tk_1250k_lowpass3dys.nc")
ind=np.argwhere(T_all[:time].data==1).squeeze()

train_idx=np.arange(ind.size)
random.seed(random_seed)
random.shuffle(train_idx)
test_idx=np.load("/scratch/hz1994/blocking/data_MMmodel/conditionT/T/test_idx.npy").squeeze()

partition = {'train': [str(idx) for idx in train_idx], 'validation': [str(idx) for idx in test_idx]}
train_labels=dict(zip([str(idx) for idx in train_idx], Y[train_idx]))
test_labels=dict(zip([str(idx) for idx in test_idx], Y[test_idx]))
# Generators
training_generator = DataGenerator(partition['train'],train_labels, **params_train)
validation_generator = DataGenerator(partition['validation'], test_labels, **params_test)

# labels=dict(zip([str(idx) for idx in np.arange(Y.shape[0])], Y))
# training_generator = DataGenerator(partition['train'], labels, **params)
# validation_generator = DataGenerator(partition['validation'], labels, **params)


# In[17]:

if cnnsize=="smaller":
    model = MM_util_AI.make_s_model((18,90,3))
elif cnnsize=="smaller_smaller":
    model = MM_util_AI.make_ss_model((18,90,3))
elif cnnsize=="normal":
    model = MM_util_AI.make_model((18,90,3))
    

model.summary()


# In[11]:

checkpoint_path = models_path_setA+'models_T_%d_committor_'%T+name+subname+"cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model.save_weights(checkpoint_path.format(epoch=0))


# In[12]:


model_callbacks = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=2, 
    save_weights_only=True,
    save_freq="epoch")

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.Precision(class_id=1),tf.keras.metrics.Recall(class_id=1)],)

# Train model on dataset

history=model.fit(x=training_generator,
                    validation_data=validation_generator,
                    epochs=EPOCH, 
                  callbacks=[model_callbacks],
                 workers=10,
                 verbose=2,
                 shuffle=False)


with open(models_path_setA+'models_T_%d_committor_'%T+name+subname+'trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


# **Notice** that this output is different from *history* object during the training. There is nothing to be surprised, the metrics on the training set are just the mean over all batches during training, as the weights are changing with each batch.
# 
# Using model.evaluate will keep the model weights fixed and compute loss/accuracy for the whole data you give in. If you want to have the loss/accuracy on the training set, then you have to use model.evaluate and pass the training set to it. The history object does not have the true loss/accuracy on the training set.
# 
