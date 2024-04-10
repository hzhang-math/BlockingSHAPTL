import xarray as xr
import numpy as np
import cartopy
from cartopy import crs as ccrs
import matplotlib 
matplotlib.rcParams["font.size"] = 12
from matplotlib import pyplot as plt
import os
from os.path import join, exists
from os import mkdir
import scipy
import netCDF4
import sklearn
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from itertools import groupby
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.colors as colors


def compute_forward_parallel(setA,setB):
    Aindex=np.where(setA==True)[0]
    LeftA=Aindex[np.where(np.diff(Aindex)!=1)[0]]
    Bindex=np.where(setB==True)[0]
    LeftB=Bindex[np.where(np.diff(Bindex)!=1)[0]]
    EnterB=Bindex[np.where(np.diff(Bindex)!=1)[0]+1]

    if Bindex.size>0:
        EnterB= np.insert(EnterB, 0, Bindex[0], axis=0)
        LeftB=np.insert(LeftB,LeftB.size,Bindex[-1], axis=0)  
    if Aindex.size>0:
        LeftA=np.insert(LeftA,LeftA.size,Aindex[-1], axis=0)    
    forward=setA*0
    for t in EnterB:
        if len(LeftA[LeftA<t])>0:
            lastleftA=LeftA[LeftA<t][-1]
            forward[lastleftA+1:t]=True
        else:
            forward[:t]=True
        if len(LeftB[LeftB>=t])>0:    
            nextleftB=LeftB[LeftB>=t][0]
            forward[t:nextleftB+1]=True
        else:
            forward[t:]=True  
    return forward

def compute_backward_parallel(setA,setB):        
    Aindex=np.where(setA==True)[0]
    Bindex=np.where(setB==True)[0]#backward: comes from A rather than B
#         LeftA=Aindex[np.where(np.diff(Aindex)!=1)[0]]
    LeftB=Bindex[np.where(np.diff(Bindex)!=1)[0]]
    EnterB=Bindex[np.where(np.diff(Bindex)!=1)[0]+1]
    EnterA=Aindex[np.where(np.diff(Aindex)!=1)[0]+1]

    if Bindex.size>0:
        EnterB= np.insert(EnterB, 0, Bindex[0], axis=0)
        LeftB=np.insert(LeftB,LeftB.size,Bindex[-1], axis=0)  
    if Aindex.size>0:
        EnterA= np.insert(EnterA, 0, Aindex[0], axis=0)
        
    backward=setA*0+1
    for t in LeftB:
        if len(EnterA[EnterA>t])>0:
            nextEnterA=EnterA[EnterA>t][0]
            backward[t:nextEnterA]=False
        else:
            backward[t:]=False

        if len(EnterB[EnterB<=t])>0:    
            lastEnterB=EnterB[EnterB<=t][-1]
            backward[lastEnterB:t]=False
        else:
            backward[:t]=False
    return backward

def compute_T(zprime,M): # mean time between the second and the fourth step:\
                                    # leave B->enter A -> leave A->enter B 
    T=0*zprime.astype(int)
    Bindex=np.where(zprime>=M)[0]
    LeftB=Bindex[np.where(np.diff(Bindex)!=1)[0]]
    EnterB=Bindex[np.where(np.diff(Bindex)!=1)[0]+1]
    
    if Bindex.size>0:
        EnterB= np.insert(EnterB, 0, Bindex[0], axis=0)
        LeftB=np.insert(LeftB,LeftB.size,Bindex[-1], axis=0)  
    
#     LeftB=min(LeftB+1,T.size)
    for t in EnterB:
        NextLeftB=LeftB[LeftB>=t]
        if NextLeftB.size>0:
            T[t:NextLeftB[0]+1]=np.arange(1,NextLeftB[0]-t+2)
        else:
            T[t:]=np.arange(T.size-t)
    return T

def computeTkandCommittor(zprime_reg,reg,name,path, M,T,setA,a=None):
    T_reg=compute_T(zprime_reg,M)
    T_reg.to_netcdf(path+reg+"_Tk_"+name+".nc")
    blockings_reg=T_reg>=T
    if setA=="a":
        setA_reg=zprime_reg<a
        print("setA: Threshold zprime_region<a") 
    elif setA=="T":
        setA_reg=(T_reg==0)
        print("setA: T_reg==0") 
    else:
        print("Unknown definition of A,function terminates")
        return
    DGreg_dataset = blockings_reg.rename("B").to_dataset()
    DGreg_dataset['A']=setA_reg
    DGreg_dataset['forward']=compute_forward_parallel(DGreg_dataset['A'],DGreg_dataset['B'])
    DGreg_dataset['backward']=compute_backward_parallel(DGreg_dataset['A'],DGreg_dataset['B'])
    if setA=="a":
        DGreg_dataset.to_netcdf(path+reg+"_committor_M_%d_T_%d_a_%d_"%(M,T,a)+name+".nc")
    elif setA=="T":
        DGreg_dataset.to_netcdf(path+reg+"_committor_M_%d_T_%d_"%(M,T)+name+".nc") 
        
    print("computed Tk, A, B, forward and backward for "+name+reg+" and saved to "+path)
    
def computeCommittor_only(reg,T_reg,name,path, M,T,setA,a=None,zprime_reg=None):
    blockings_reg=T_reg>=T
    if setA=="a":
        setA_reg=zprime_reg<a
        print("setA: Threshold zprime_region<a") 
    elif setA=="T":
        setA_reg=(T_reg==0)
        print("setA: T_reg==0") 
    else:
        print("Unknown definition of A,function terminates")
        return
    DGreg_dataset = blockings_reg.rename("B").to_dataset()
    DGreg_dataset['A']=setA_reg
    DGreg_dataset['forward']=compute_forward_parallel(DGreg_dataset['A'],DGreg_dataset['B'])
    DGreg_dataset['backward']=compute_backward_parallel(DGreg_dataset['A'],DGreg_dataset['B'])
    if setA=="a":
        DGreg_dataset.to_netcdf(path+reg+"_committor_M_%d_T_%d_a_%d_"%(M,T,a)+name+".nc")
    elif setA=="T":
        DGreg_dataset.to_netcdf(path+reg+"_committor_M_%d_T_%d_"%(M,T)+name+".nc") 
        
    print("computed A, B, forward and backward for "+name+reg+" and saved to "+path)    
    
def splitdataXfortraining(X_path,Tkpath, save_path, reg, name,M,T,setA,a=None):
    T_atl=xr.open_dataarray(Tkpath+reg+"_Tk_"+name+".nc")
    if setA=="a":
        path=save_path+setA+"_"+str(a)+"/"
    elif setA=="T":
        path=save_path+setA+"/"

    for i in ['200','500','800']:
        X=xr.open_dataarray(X_path+"dim_Zprime"+i+"_"+name+".nc").sel(latitude=slice(20,90)).data
        for t in [0,1,2,3,4,5]:
            np.save(path+"data_X"+i+"_T%d_"%t+name+".npy",X[T_atl==t])    
        print("splitted data for X "+i)
    print("splitted X saved to "+path)
    
def normalize_centralize_data(data_path, save_path, name):
    for T in range(6):
        data=np.load(data_path+"data_X_T%d_"%T+name+".npy")
        data_std=data.std(axis=0,keepdims=True)
        data_mean=data.mean(axis=0,keepdims=True)
        data_preprocessed=(data-data_mean)/data_std
        np.save(save_path+"data_X_T%d_"%T+name+"_proc.npy",data_preprocessed)
        print(T)
    print("saved to "+save_path)
    
def splitdataYfortraining(Y_path,Tkpath,save_path, reg, name,M,T,setA,a=None):
    T_atl=xr.open_dataarray(Tkpath+reg+"_Tk_"+name+".nc")
    if setA=="a":
        path=save_path+setA+"_"+str(a)+"/"
    elif setA=="T":
        path=save_path+setA+"/"    
    
    if setA=="a":
        Y=xr.open_dataset(Y_path+reg+"_committor_M_%d_T_%d_a_%d_"%(M,T,a)+name+".nc")['forward'].data
    elif setA=="T":
        Y=xr.open_dataset(Y_path+reg+"_committor_M_%d_T_%d_"%(M,T)+name+".nc")['forward'].data
    else:
        print("Unknown definition of A,function terminates")
        return
    
    for t in [0,1,2,3,4,5]:    
        np.save(path+"data_Y_T%d_"%t+name+".npy",Y[T_atl==t])
    print("splitted Y saved to "+path)
        
def combinelevelsofX(name,train_path, save_path):
    for t in [0,1,2,3,4,5]:
        X1=np.load(train_path+"data_X200_T%d_"%t+name+".npy")
        X2=np.load(train_path+"data_X500_T%d_"%t+name+".npy")
        X3=np.load(train_path+"data_X800_T%d_"%t+name+".npy")
        X=np.stack([X1,X2,X3],axis=-1)
        np.save(save_path+"data_X_T%d_"%t+name+".npy",X)
        print("saved to "+save_path+"data_X_T%d_"%t+name+".npy")