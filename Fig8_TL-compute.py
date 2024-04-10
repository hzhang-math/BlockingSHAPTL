#!/usr/bin/env python
# coding: utf-8

# This notebook does transfer learning. It first load the trained model on MM model, then use it as the initialization of training the last layer on ERA5 data. The training performace is statistically evaluated using 10 training-testing procedure. They are saved in the directory
# path="/scratch/hz1994/blocking/data_MMmodel/CNNmodels/T/era5_retrainCNN/extreme_%ddaysblocking/"%Duration+\
#     "data_"+data_amount+"_"+cnnsize+"_cnn_"+"regularize"+"_%.3e"%epsilon+"_rs_%d"%random_seed+"epoch_%d/"%epoch
# 
# 

# In[74]:


import xarray as xr
import numpy as np
import cartopy
from cartopy import crs as ccrs
import matplotlib 
matplotlib.rcParams["font.size"] = 12
from matplotlib import pyplot as plt
from os.path import join, exists
from os import mkdir
import scipy
import netCDF4
import sklearn
import sys
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)

import matplotlib.path as mpath
import importlib.util
import MM_util_AI
import MM_utilplot
import warnings
import pickle
from tensorflow import keras

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap
import tensorflow as tf
from sklearn.metrics import confusion_matrix,recall_score,precision_score
tf.config.experimental_run_functions_eagerly(True)
import os

spec = importlib.util.spec_from_file_location("MM_dataprepare",                         "/scratch/hz1994/blocking/MMmodel/MMmodel/notebooks/MM_dataprepare.py")
MM_dataprepare = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = MM_dataprepare
spec.loader.exec_module(MM_dataprepare)

spec = importlib.util.spec_from_file_location("MM_utilblocking",                         "/scratch/hz1994/blocking/MMmodel/MMmodel/notebooks/MM_utilblocking.py")
MM_utilblocking = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = MM_utilblocking
spec.loader.exec_module(MM_utilblocking)


   
            
with open("/scratch/hz1994/blocking/data_MMmodel/filepath.txt","r") as fi:
    for ln in fi:
        if ln.startswith("dimensionalized_filepath"):
            dim_path=ln.strip().split('\t')[1]
        if ln.startswith("nondimensionalized_filepath"):
            nondim_path=ln.strip().split('\t')[1]
        if ln.startswith("code_filepath"):
            code_path=ln.strip().split('\t')[1]            
        if ln.startswith("DGindex_filepath"):
            DGindex_path=ln.strip().split('\t')[1]  
        if ln.startswith("conditionedT_filepath" ):
            train_path=ln.strip().split('\t')[1]
        if ln.startswith("model_filepath" ):
            models_path=ln.strip().split('\t')[1]
        if ln.startswith("fig_filepath" ):
            fig_path=ln.strip().split('\t')[1] 
print(dim_path)
print(nondim_path)
print(code_path)
print(DGindex_path)
print(train_path)
print(models_path)
print(fig_path)


# # load the model to retrain and prepare the data for retraining. 

# In[75]:


train_path_setA=train_path+'T/'
models_path_setA=models_path+'T/'
epsilon=0.0

setting=sys.argv
cnnsize=str(setting[1])
data_amount=str(setting[2])
random_seed=int(setting[3])
epoch=int(setting[4])
Duration=int(setting[5])
trainable_layer_number=str(setting[6])
learning_rate=float(setting[7])

print("cnnnsize=",cnnsize)
print("data_amount=",data_amount)
print("random_seed=", random_seed)
print("epoch=",epoch)
print("Duration=",Duration)
print("trainable_layer_number=",trainable_layer_number)
print("learning_rate=", learning_rate)

subname="/batches_resampling_"+"data_"+data_amount+"_"+cnnsize+"_cnn_"+"regularize"+"_%.3e"%epsilon+"_rs_%d/"%random_seed
name="1250k_lowpass3dys"
if Duration==5:
    weight_location=models_path_setA+'models_T_1_committor_'+name+subname
if Duration==7:
    weight_location="/scratch/hz1994/blocking/data_MMmodel/conditionT/Duration_7d/"+'models_T_1_committor_'+name+subname
if Duration==9:
    weight_location="/scratch/hz1994/blocking/data_MMmodel/conditionT/Duration_9d/"+'models_T_1_committor_'+name+subname

weightpath=weight_location+"cp-%04d.ckpt"%epoch

X=np.load("/scratch/hz1994/blocking/data_era5/"+"test_data_1940-2022.npy")  
Ysparse=np.load("/scratch/hz1994/blocking/data_era5/"+"test_labels_1940-2022_T%d.npy"%Duration)
pos=Ysparse.sum()
total=Ysparse.size
neg=total-pos
print('In total: positive data=',pos,'negative data=',neg)
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
print("class_weight=", class_weight)
Y=np.zeros((Ysparse.size,2)).astype(bool)
Y[:,1][Ysparse==1]=True  #blocking
Y[:,0][Ysparse==0]=True


# # Make the training/testing dataset for ensemble training

# In[76]:


seed=10
num=10
num_total=X.shape[0]
ordered_ind=np.arange(num_total)
np.random.seed(seed)
np.random.shuffle(ordered_ind)
shuffled_ind=np.random.shuffle(ordered_ind)
test_data_list=(np.array_split(ordered_ind, num))
train_data_list=[]
for i in range(num):
    train_data_list.append( np.array( [j  for j in ordered_ind if j not in test_data_list[i]]   )  )


# # Retraining

#  set up the path and record the initialization (the parameter setting of MMtraining and the epoch we take from)

# In[77]:


path="/scratch/hz1994/blocking/data_MMmodel/CNNmodels/T/era5_retrainCNN/extreme_%ddaysblocking/trained_layer_%s/learning_rate_%.4f/"%( Duration,trainable_layer_number,learning_rate)+    "data_"+data_amount+"_"+cnnsize+"_cnn_"+"regularize"+"_%.3e"%epsilon+"_rs_%d"%random_seed+"epoch_%d/"%epoch
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)
with open(path+'base_model.txt', 'w') as f:
    f.write(weightpath)
    f.write("\n")
    f.write("Duration=%d"%Duration)
    f.write("learning_rate=%.4f"%learning_rate)

# training

# In[78]:


history_list=[]
EPOCHS=20
for i in range(num):
    # load data
    test_ind=test_data_list[i]
    train_ind=train_data_list[i]
    train_data=X[train_ind]
    train_labels=Y[train_ind]
    test_data=X[test_ind]
    test_labels=Y[test_ind]
    
    #load base_model: only train the last layer

    if cnnsize=="smaller":
        base_model = MM_util_AI.make_s_model((18,90,3))
    elif cnnsize=="smaller_smaller":
        base_model = MM_util_AI.make_ss_model((18,90,3))
    elif cnnsize=="normal":
        base_model = MM_util_AI.make_model((18,90,3))
        

    base_model.load_weights(weightpath).expect_partial()
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
   
    for layer in base_model.layers:
        layer.trainable = False
    if trainable_layer_number=="First_and_Last": 
        base_model.layers[0].trainable = True
        base_model.layers[-1].trainable = True
    else:
        base_model.layers[int(trainable_layer_number)].trainable = True
    
    path_i=path+"%d/"%i
    if not os.path.exists(path_i):
        os.makedirs(path_i)
    checkpoint_path = path_i+"cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    base_model.save_weights(checkpoint_path.format(epoch=0))
    model_callbacks = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq="epoch")
    optimizer= keras.optimizers.Adam(learning_rate=learning_rate)
    base_model.compile(optimizer =optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.Precision(class_id=1,thresholds=0),tf.keras.metrics.Recall(class_id=1,thresholds=0)],)

    history=base_model.fit(train_data,train_labels,
                        validation_data=[test_data,test_labels],
                        callbacks=[model_callbacks],
                        class_weight=class_weight,
                        epochs=EPOCHS,
                        workers=8)
    history_list.append(history.history)
    with open(path_i+'/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    del base_model
    del history


# # Plot everything

# In[79]:


precision_transfer=[]
recall_transfer=[]
for i,history in enumerate(history_list):
    val_precision=next(v for k,v in history.items() if 'val_precision' in k)
    val_precision =np.array(val_precision)
    val_recall=next(v for k,v in history.items() if 'val_recall' in k)
    val_recall =np.array(val_recall)
#     ax[0].plot( val_precision)
#     ax[1].plot(val_recall)
    precision_transfer.append(val_precision)
    recall_transfer.append(val_recall)


# ## Compute the first dots before training

# In[81]:


precision_0=[]
recall_0=[]
for i in range(num):
    test_ind=test_data_list[i]
    train_ind=train_data_list[i]
    train_data=X[train_ind]
    train_labels=Y[train_ind]
    test_data=X[test_ind]
    test_labels=Y[test_ind]    

    if cnnsize=="smaller":
        base_model = MM_util_AI.make_s_model((18,90,3))
    elif cnnsize=="smaller_smaller":
        base_model = MM_util_AI.make_ss_model((18,90,3))
    elif cnnsize=="normal":
        base_model = MM_util_AI.make_model((18,90,3))
     
    base_model.load_weights(weightpath).expect_partial()
    predictions = base_model.predict(test_data)
    pred=(predictions[:,0]<predictions[:,1])
    val_recall=recall_score(Ysparse[test_ind], pred , labels=1 )
    val_precision=precision_score(Ysparse[test_ind], pred , labels=1 )
    precision_0.append(val_precision)
    recall_0.append(val_recall)


# In[82]:



precision_transfer=[]
recall_transfer=[]
for i,history in enumerate(history_list):
    val_precision=next(v for k,v in history.items() if 'val_precision' in k)
    val_precision =np.array([precision_0[i]]+val_precision)
    val_recall=next(v for k,v in history.items() if 'val_recall' in k)
    val_recall =np.array([recall_0[i]]+val_recall)
#     ax[0].plot( val_precision)
#     ax[1].plot(val_recall)
    precision_transfer.append(val_precision)
    recall_transfer.append(val_recall)

np.save(path+"precision_transfer.npy", np.array(precision_transfer))
np.save(path+"recall_transfer.npy", np.array(recall_transfer))

