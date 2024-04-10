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

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap
import tensorflow as tf
from sklearn.metrics import confusion_matrix,recall_score,precision_score
import matplotlib.colors as colors
import os

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
train_path_setA=train_path+'T/'
models_path_setA=models_path+'T/'
import tensorflow as tf

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

font = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 16}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 1.5

from sklearn.metrics import confusion_matrix,recall_score,precision_score
def test_model(model,weightpath,test_data,test_classlabels,test_labels):
    model.load_weights(weightpath)
    predictions = model.predict(test_data)
    pred=(predictions[:,0]<predictions[:,1])
    bce=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss=bce(test_classlabels, predictions).numpy()
    TN,FP,FN,TP = confusion_matrix(test_labels, pred).flatten()
    recall=recall_score(test_labels, pred , labels=1 )
    precision=precision_score(test_labels, pred , labels=1 )
    return np.array([TN,FP,FN,TP]),recall,precision,loss

def polorplot(ax,data_xr,max_abs,levels,iv=0.02):
    norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    im=xr.plot.contourf( 
        data_xr,
        x="longitude", y="latitude", ax=ax,transform=ccrs.PlateCarree(),cmap='coolwarm',\
        levels=levels, add_colorbar=False,norm=norm
    )
    gl=ax.gridlines(draw_labels=False)
    gl.ylocator = mticker.FixedLocator([20,50,60,70])  
    ax.coastlines()
    
    return ax,im

def polorplot_levels(plotmap,latitudes,longitudes ,minval,maxval,label="SHAP values", number_levels=30,iv=0.02):
    fig,ax = plt.subplots(figsize=(9,3), 
            subplot_kw={'projection': ccrs.NorthPolarStereo()},ncols=3)
    titles=["Z200","Z500","Z800"]
    max_abs=max(abs(minval),abs(maxval))
    print("min plotmap=", plotmap.min() ,"max plotmap=", plotmap.max() ,)
    for i in range(3):
        y=plotmap[:,:,i]
        a = xr.DataArray(y, 
            coords={'latitude':latitudes,'longitude': longitudes,}, 
            dims=["latitude","longitude",])
        ax[i],im =polorplot( ax[i],a, max_abs=max_abs,\
                            levels=np.linspace(minval,maxval,30 ), iv=iv)
        ax[i].set_title(titles[i])
        
    cbar_ax = fig.add_axes([0.05, -0.1, .9, .05]) #left, bottom, width, height
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal",cmap='coolwarm',\
                        ticks= np.arange(iv*int(minval/iv),iv*int(maxval/iv)+iv,iv),\
                        label = label, shrink = 1,)
#     plt.subplots_adjust(wspace=0.1,width_ratios=[1,1,1])
    fig.tight_layout()
    return fig,ax



setting=sys.argv 
random_seed=int(setting[1])
num=int(setting[2])
epoch=int(setting[3]) #the epoch of pre-training
epoch_TL=int(setting[4]) #the epoch of fine-tuning
Duration=int(setting[5])

X=np.load("/scratch/hz1994/blocking/data_era5/"+"test_data_1940-2022.npy")  
Ysparse=np.load("/scratch/hz1994/blocking/data_era5/"+"test_labels_1940-2022_T%d.npy"%Duration)
Y=np.zeros((Ysparse.size,2)).astype(bool)
Y[:,1][Ysparse==1]=True  #blocking
Y[:,0][Ysparse==0]=True
latitudes = np.load(dim_path+'dataX_lat.npy')
longitudes = np.load(dim_path+'dataX_lon.npy')
X_dim=X[0].size

cnnsize="normal"
data_amount="1000.0k"
epsilon=0
trainable_layer_number=7
learning_rate=0.0001 
print("random_seed=",random_seed, "num=",num)
print("epoch=",epoch, "epoch_TL=",epoch_TL,"Duration=", Duration)

tf.config.experimental_run_functions_eagerly(False)
if cnnsize=="smaller":
    base_model = MM_util_AI.make_s_model((18,90,3))
    TL_model = MM_util_AI.make_s_model((18,90,3))
elif cnnsize=="smaller_smaller":
    base_model = MM_util_AI.make_ss_model((18,90,3))
    TL_model = MM_util_AI.make_ss_model((18,90,3))
elif cnnsize=="normal":
    base_model = MM_util_AI.make_model((18,90,3))
    TL_model = MM_util_AI.make_model((18,90,3))
path="/scratch/hz1994/blocking/data_MMmodel/CNNmodels/T/era5_retrainCNN/extreme_%ddaysblocking/trained_layer_%s/learning_rate_%.4f/"\
%( Duration,trainable_layer_number,learning_rate)+    "data_"+data_amount+"_"+cnnsize\
+"_cnn_"+"regularize"+"_%.3e"%epsilon+"_rs_%d"%random_seed+"epoch_%d/"%epoch


weightpath_TL=path+"%d/"%num +"cp-%04d.ckpt"%epoch_TL
TL_model.load_weights(weightpath_TL)
background=X
probability_model = tf.keras.Sequential([TL_model, 
                                    tf.keras.layers.Softmax()])
TL_e = shap.DeepExplainer(probability_model, background)

TL_shap_values_list=[]
for i in range(Y.shape[0]):
    TL_shap_values = TL_e.shap_values(X[[i]])
    TL_shap_values_list.append(TL_shap_values)
TL_shap_values_list=np.array(TL_shap_values_list)

np.save(path+"%d/"%num+"shapvalue-%04d_prob.npy"%epoch_TL, TL_shap_values_list )
# shap_values_list=np.load(path+"%d/"%num+"shapvalue-%04d_prob.npy"%0)
# TL_shap_values_list=np.load(path+"%d/"%num+"shapvalue-%04d_prob.npy"%epoch_TL)
