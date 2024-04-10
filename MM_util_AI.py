import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import struct
import os
from os.path import join, exists
from os import mkdir
import scipy
import netCDF4
import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def make_s_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(18,90,3),name="conv2d"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu',name="conv2d_1"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu',name="conv2d_2"))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2))
    return model 
def make_ss_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(18,90,3),name="conv2d"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu',name="conv2d_1"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu',name="conv2d_2"))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(2))
    return model 

def make_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,name="conv2d"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',name="conv2d_1"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',name="conv2d_2"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))
    return model 

def make_grad(data,model, seperate=False, pred_index=None):
    image = tf.Variable(data.astype(dtype='float32'))
    with tf.GradientTape () as tape:
        tape.watch(image)
        prediction = model(image, training=False)  # Logits for this minibatch
        if pred_index is None:
            pred_class = tf.argmax(prediction[0])
        else:
            pred_class = pred_index
            
        class_channel = prediction[0, pred_class]
    grads = tape.gradient(class_channel,image)[0]
#     heatmap=grads
    heatmap=tf.squeeze(data*grads)
    heatmap=tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    if seperate==True:
        heatmap1 = heatmap *(data>0)
        heatmap2 = heatmap *(data<=0)
        return heatmap1,heatmap2
    else:
        return heatmap
    
def make_grad_pos_neg(data,model, seperate=False, pred_index=None):
    image = tf.Variable(data.astype(dtype='float32'))
    with tf.GradientTape () as tape:
        tape.watch(image)
        prediction = model(image, training=False)  # Logits for this minibatch
        if pred_index is None:
            pred_class = tf.argmax(prediction[0])
        else:
            pred_class = pred_index
            
        class_channel = prediction[0, pred_class]
    grads = tape.gradient(class_channel,image)[0]
#     heatmap=grads
    heatmap=tf.squeeze(data*grads)
    heatmap_increase=np.array(tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap))
    heatmap_reduce=np.array(tf.maximum(-heatmap, 0) / tf.math.reduce_max(-heatmap))
    
    if seperate==True:
        heatmap1_increase = heatmap_increase *(data>0)
        heatmap2_increase = heatmap_increase *(data<=0)
        heatmap1_reduce = heatmap_reduce *(data>0)
        heatmap2_reduce = heatmap_reduce *(data<=0)
        
        return heatmap1_increase,heatmap2_increase,heatmap1_reduce,heatmap2_reduce
    else:
        return heatmap_increase, heatmap_reduce
    
def composite_heatmap(Ysparse,pred,Yreal,Ypred,length,data_preprocessed,model,seperate=False, pred_index=None):
    if seperate==True:
        pdata_heatmap=[]
        ndata_heatmap=[]
        ind=np.where(np.logical_and(Ysparse==Yreal, pred==Ypred))[0]   
        for i in np.arange(0,ind.size,max( ind.size//length,1 )):
            heatmap1,heatmap2 = make_grad(data_preprocessed[ind[i]][None,...], model, \
                                          seperate=True, pred_index=pred_index)
            if ~np.any(np.isnan(heatmap)):
                pdata_heatmap.append(heatmap1)
                ndata_heatmap.append(heatmap2)
        pheatmap=np.array(pdata_heatmap).mean(axis=0)
        nheatmap=np.array(ndata_heatmap).mean(axis=0)
        return pheatmap,nheatmap
    else:
        data_heatmap=[]
        ind=np.where(np.logical_and(Ysparse==Yreal, pred==Ypred))[0]   
        for i in np.arange(0,ind.size,max( ind.size//length,1 )):
            heatmap= make_grad(data_preprocessed[ind[i]][None,...], model)
            if ~np.any(np.isnan(heatmap)):
                data_heatmap.append(heatmap)
        heatmap=np.array(data_heatmap).mean(axis=0)
        return heatmap
    