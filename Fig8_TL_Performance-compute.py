import xarray as xr
import numpy as np
import matplotlib 
matplotlib.rcParams["font.size"] = 12
from matplotlib import pyplot as plt
from os.path import join, exists
from os import mkdir
import scipy
import netCDF4
import sys
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
import matplotlib.path as mpath
import importlib.util

import warnings
import pickle

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap

from sklearn.metrics import confusion_matrix,recall_score,precision_score

import os
import matplotlib as mpl
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import MM_util_AI
import MM_utilplot
from matplotlib import cm 
spec = importlib.util.spec_from_file_location("MM_dataprepare", \
                        "/scratch/hz1994/blocking/MMmodel/MMmodel/notebooks/MM_dataprepare.py")
MM_dataprepare = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = MM_dataprepare
spec.loader.exec_module(MM_dataprepare)

spec = importlib.util.spec_from_file_location("MM_utilblocking", \
                        "/scratch/hz1994/blocking/MMmodel/MMmodel/notebooks/MM_utilblocking.py")
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
from sklearn.metrics import confusion_matrix

setting=sys.argv
Duration=int(setting[1])
epoch_start=int(setting[2])
random_seed=int(setting[3])

X=np.load("/scratch/hz1994/blocking/data_era5/test_data_1940-2022.npy")
Ysparse=np.load("/scratch/hz1994/blocking/data_era5/test_labels_1940-2022_T%d.npy"%Duration)  
Y=np.zeros((Ysparse.size,2)).astype(bool)
Y[:,1][Ysparse==1]=True  #blocking
Y[:,0][Ysparse==0]=True


num_total=X.shape[0]
seed=10
ordered_ind=np.arange(num_total)
np.random.seed(seed)
np.random.shuffle(ordered_ind)
num=10
shuffled_ind=np.random.shuffle(ordered_ind)
test_data_list=(np.array_split(ordered_ind, num))

cnnsize='normal'
data_amount='1000.0k'

TL_tnfpfntp =True
if TL_tnfpfntp:
    tname="7"
    learning_rate="0.0001"
    path0="/scratch/hz1994/blocking/data_MMmodel/CNNmodels/T/era5_retrainCNN/"+\
                    "extreme_%ddaysblocking/"%Duration+\
                    "trained_layer_%s/learning_rate_%s/"%(tname, learning_rate)

    epsilon=0.0
    print("epoch=",epoch_start)
    precision_transfer=[]
    recall_transfer=[]
    for tind,t in enumerate(range(10)):# the 7th training-test set pair
        test_ind=test_data_list[tind]
        test_data=X[test_ind]
        test_labels=Y[test_ind]
        print("t=",t)
        tn_list=[]
        fp_list=[]
        fn_list=[]
        tp_list=[]
        path=path0+"data_"+data_amount+"_"+cnnsize+"_cnn_"+"regularize"+"_%.3e"%epsilon+\
                            "_rs_%d"%random_seed+"epoch_%d/"%epoch_start+"%d/"%t
        for epoch in range(21):
            weightpath=path+"cp-%04d.ckpt"%epoch
            model = MM_util_AI.make_model((18,90,3))
            model.load_weights(weightpath).expect_partial()
            y_pred = model.predict(test_data)
            y_pred = np.argmax(y_pred, axis=1)
            tn, fp, fn, tp = confusion_matrix(test_labels[:,1], y_pred).ravel()
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)
            tp_list.append(tp)
        np.save(path+"TN.npy",tn_list)
        np.save(path+"FP.npy",fp_list)
        np.save(path+"FN.npy",fn_list)
        np.save(path+"TP.npy",tp_list)
        recall_list=np.array(tp_list)/(np.array(tp_list)+np.array(fn_list)) 
        precision_list=np.array(tp_list)/(np.array(tp_list)+np.array(fp_list))

        precision_transfer.append(precision_list)
        recall_transfer.append(recall_list)
    np.save(path0+"data_"+data_amount+"_"+cnnsize+"_cnn_"+"regularize"+"_%.3e"%epsilon+\
                            "_rs_%d"%random_seed+"epoch_%d/"%epoch_start+"precision_transfer.npy",precision_transfer)    
    np.save(path0+"data_"+data_amount+"_"+cnnsize+"_cnn_"+"regularize"+"_%.3e"%epsilon+\
                            "_rs_%d"%random_seed+"epoch_%d/"%epoch_start+"recall_transfer.npy",recall_transfer)
    print(path0+"data_"+data_amount+"_"+cnnsize+"_cnn_"+"regularize"+"_%.3e"%epsilon+"_rs_%d"%random_seed+"epoch_%d/"%epoch_start+"recall_transfer.npy")    
