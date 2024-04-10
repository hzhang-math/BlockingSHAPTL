import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import struct
import matplotlib 
import os
from os.path import join, exists
from os import mkdir
import scipy
import netCDF4
import matplotlib.ticker as mticker
import cartopy
from cartopy import crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)

import numpy.ma as ma

def polorplot(ax,data_xr,levels=None,):
    min_data=data_xr.min()
    max_data=data_xr.max()
    if np.any(levels)==None:
        levels=np.linspace(min_data,max_data,30)
    im=xr.plot.contourf( 
        data_xr,
        x="longitude", y="latitude", ax=ax,transform=ccrs.PlateCarree(),cmap='coolwarm',\
        levels=levels,
#         levels=np.linspace(min_data,max_data,10),
#         levels=np.linspace(-100,200,10),
                cbar_kwargs={
                'label': "",'orientation':'vertical','shrink':0.8,'pad': 0.1,
            },

    )
    gl=ax.gridlines(draw_labels=True)
    gl.ylocator = mticker.FixedLocator([20,50,60,70])  
    #     fig.colorbar(im, orientation="horizontal", pad=0.2)
    ax.coastlines()
    return ax

def polorplot_levels(plotmap,latitudes,longitudes, suptitle ,minval=None,maxval=None, number_levels=30):
    fig,ax = plt.subplots(figsize=(10,3), 
            subplot_kw={'projection': ccrs.NorthPolarStereo()},ncols=3)

    titles=["Z200","Z500","Z800"]
    if minval==None and maxval==None:
        minval=plotmap.min()
        maxval=plotmap.max()
    for i in range(3):
        y=plotmap[:,:,i]
        a = xr.DataArray(y, 
            coords={'latitude':latitudes,'longitude': longitudes,}, 
            dims=["latitude","longitude",])
        ax[i]=polorplot( ax[i],a,np.linspace(minval,maxval,number_levels),)
        ax[i].set_title(titles[i])
    plt.suptitle(suptitle)
    fig.tight_layout()
    return fig,ax
    
def plot2d(x,y,xname,yname,Ysparse, otherinformation=None, number_bins=20,density_threshold=1e-4, ):
    xmin=x.min()
    xmax=x.max()
    intervalx=np.linspace(x.min(),x.max(),number_bins)
    dx=intervalx[1]-intervalx[0]
    
    ymin=y.min()
    ymax=y.max()
    intervaly=np.linspace(y.min(),y.max(),number_bins)
    dy=intervaly[1]-intervaly[0]    
    samples_class= [ [[]    for i in range(number_bins) ]    for j in range(number_bins) ]
    mean=np.zeros((number_bins,number_bins))
    std=np.zeros((number_bins,number_bins))
    density=np.zeros((number_bins,number_bins))
    for t in range(x.shape[0]):
        samples_class[int((x[t]-xmin)/dx)][int((y[t]-ymin)/dy)].append(Ysparse[t])
    for i in range(number_bins):
        for j in range(number_bins):
            samples=np.array(samples_class[i][j])
            mean[i,j]= samples.mean() 
            std[i,j]= samples.std() 
            density[i,j]=samples.size/x.shape[0]
    density=ma.masked_where(density<density_threshold, density) 
    mean=ma.masked_where(density<density_threshold, mean) 
    std=ma.masked_where(density<density_threshold, std) 
    
    fig,ax=plt.subplots(figsize=[10,3],ncols=3)
    c=ax[0].contourf(mean,cmap="coolwarm", vmax=1, vmin=0) 
    plt.colorbar(c,ax=ax[0])
    c=ax[1].contourf(std,cmap="coolwarm") 
    plt.colorbar(c,ax=ax[1])
    c=ax[2].contourf(density,cmap="coolwarm",locator=mticker.LogLocator(base=10.0, numticks=15)) 
    plt.colorbar(c,ax=ax[2])
    for a in ax:
        a.set_xticks(np.arange(0,len(intervaly),4))
        a.set_yticks(np.arange(0,len(intervalx),4))
        a.set_xticklabels(["%.1f"%_ for _ in intervaly[::4]])
        a.set_yticklabels(["%.1f"%_ for _ in intervalx[::4]])
        a.set_xlabel(yname)
        a.set_ylabel(xname)
    
    ax[0].set_title("committor")
    ax[1].set_title("std")
    ax[2].set_title("density")
    plt.suptitle(otherinformation)
    fig.tight_layout()
    return fig,ax
    
def plot_bunch_timeseqdist(ax,fullseq, index, time, figtype="shades", number_lines=None, label=None,color=None):
    #colomns are indicies, rows are time
    xt=np.arange(-time,time) 
    if figtype=="shades":
        matrix=np.zeros((2*time, index.size))*np.nan
        for col,i in enumerate(index):
            index_range=np.arange(max(0,i-time), min(i+time, fullseq.shape[0]))
            matrix[index_range-(i-time),col]=fullseq[index_range]
        mean=np.nanmean(matrix,axis=1)
        std=np.nanstd(matrix,axis=1)
        p = ax.plot(xt,mean ,label=label,color=color)
        
        color=p[0].get_color()
        ax.plot(xt,mean-std,color=color,alpha=0. )
        ax.plot(xt,mean+std,color=color,alpha=0. )
        ax.fill_between(xt, mean-std, mean+std,color=color,alpha=0.1)
    elif figtype=="lines":
        pltindex=index[ ::index.shape[0]//number_lines ]
        matrix=np.zeros((2*time, pltindex.size))*np.nan
        for col,i in enumerate(pltindex):
            index_range=np.arange(max(0,i-time), min(i+time, fullseq.shape[0]))
            matrix[index_range-(i-time),col]=fullseq[index_range]
            
        p = ax.plot(xt,matrix ,color=color, label=label)
    else:
        print("Please specify the figtype (shades or lines).")
        
def plot1d(ax,x ,Ysparse, xname, y2lim=None, label=None, otherinformation=None,num_bins=20):
    xmin=x.min()
    xmax=x.max()
    intervalx=np.linspace(x.min(),x.max()+1e-8,num_bins+1)
    dx=intervalx[1]-intervalx[0]    
    samples_class= [ []    for _ in range(num_bins) ]
    mean=np.zeros(num_bins)
    std=np.zeros(num_bins)
    count=np.zeros(num_bins)
    for t in range(x.shape[0]):
        samples_class[int((x[t]-xmin)/dx)].append(Ysparse[t])
    for i in range(num_bins):
        samples=np.array(samples_class[i])
        count[i]=samples.size
        mean[i]= samples.mean() 
        std[i]= samples.std() 
    plt_intervalx=(intervalx[1:]+intervalx[:-1])/2
    p = ax.plot(plt_intervalx,mean ,label=label)
    color=p[0].get_color()
    ax.plot(plt_intervalx,mean-std,color=color,alpha=0. )
    ax.plot(plt_intervalx,mean+std,color=color,alpha=0. )
    ax.fill_between(plt_intervalx, mean-std, mean+std,color=color,alpha=0.2)
    ax.set_xlabel(xname)
    ax.set_ylim([0,1])
    ax.set_title(otherinformation)
    ax1=ax.twinx()
    ax1.semilogy(plt_intervalx,count,'--', color='grey',alpha=0.4 )
    ax1.set_ylim([1, y2lim])
    return plt_intervalx, mean, count