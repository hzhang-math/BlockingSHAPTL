import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import struct
import sys
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
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap
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
import MM_util_AI
import MM_utilplot
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
from sklearn.metrics import confusion_matrix,recall_score,precision_score


setting=sys.argv 
random_seed=int(setting[1])
t=int(setting[2])
direct_training_epoch=int(setting[3])
T=int(setting[4])


epsilon=0
cnnsize="normal"
data_amount="full"
X=np.load("/scratch/hz1994/blocking/data_era5/test_data_1940-2022.npy")
Ysparse=np.load("/scratch/hz1994/blocking/data_era5/test_labels_1940-2022_T%d.npy"%T)  
Y=np.zeros((Ysparse.size,2)).astype(bool)
Y[:,1][Ysparse==1]=True  #blocking
Y[:,0][Ysparse==0]=True

print(random_seed,t)
path="/scratch/hz1994/blocking/data_MMmodel/CNNmodels/T/era5_trainCNN/extreme_%ddaysblocking/%s/"%(T,cnnsize)\
        +"random_initCNN_%d/"%random_seed+"%d/"%t


weightpath=path+"cp-%04d.ckpt"%direct_training_epoch
model = MM_util_AI.make_model((18,90,3))
model.load_weights(weightpath)


probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
prob_predictions = probability_model.predict(X)

background=X
e = shap.DeepExplainer(probability_model, background)
shap_values = e.shap_values(X)
shap_values_list=[]
for i in range(X.shape[0]):
    shap_values = e.shap_values(X[[i]])
    shap_values_list.append(shap_values)
shap_values_list=np.array(shap_values_list)
np.save(path+"shap_values_era5_epoch%d.npy" %direct_training_epoch,shap_values_list)
print(path+"shap_values_era5_epoch%d.npy" %direct_training_epoch)

