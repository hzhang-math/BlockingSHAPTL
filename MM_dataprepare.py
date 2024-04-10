from __future__ import (absolute_import, division, print_function)

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
import scipy.signal

"""Tools for working with Gaussian grids."""

import functools

import numpy as np
import numpy.linalg as la
from numpy.polynomial.legendre import legcompanion, legder, legval



def __single_arg_fast_cache(func):
    """Caching decorator for functions of one argument."""
    class CachingDict(dict):
    
        def __missing__(self, key):
            result = self[key] = func(key)
            return result
            
        @functools.wraps(func)
        def __getitem__(self, *args, **kwargs):
            return super(CachingDict, self).__getitem__(*args, **kwargs)
            
    return CachingDict().__getitem__


@__single_arg_fast_cache
def gaussian_latitudes(n):
    """Construct latitudes and latitude bounds for a Gaussian grid.
    
    Args:
    
    * n:
        The Gaussian grid number (half the number of latitudes in the
        grid.
    Returns:
        A 2-tuple where the first element is a length `n` array of
        latitudes (in degrees) and the second element is an `(n, 2)`
        array of bounds.
    """
    if abs(int(n)) != n:
        raise ValueError('n must be a non-negative integer')
    nlat = 2 * n
    # Create the coefficients of the Legendre polynomial and construct the
    # companion matrix:
    cs = np.array([0] * nlat + [1], dtype=np.int)
    cm = legcompanion(cs)
    # Compute the eigenvalues of the companion matrix (the roots of the
    # Legendre polynomial) taking advantage of the fact that the matrix is
    # symmetric:
    roots = la.eigvalsh(cm)
    roots.sort()
    # Improve the roots by one application of Newton's method, using the
    # solved root as the initial guess:
    fx = legval(roots, cs)
    fpx = legval(roots, legder(cs))
    roots -= fx / fpx
    # The roots should exhibit symmetry, but with a sign change, so make sure
    # this is the case:
    roots = (roots - roots[::-1]) / 2.
    # Compute the Gaussian weights for each interval:
    fm = legval(roots, cs[1:])
    fm /= np.abs(fm).max()
    fpx /= np.abs(fpx).max()
    weights = 1. / (fm * fpx)
    # Weights should be symmetric and sum to two (unit weighting over the
    # interval [-1, 1]):
    weights = (weights + weights[::-1]) / 2.
    weights *= 2. / weights.sum()
    # Calculate the bounds from the weights, still on the interval [-1, 1]:
    bounds1d = np.empty([nlat + 1])
    bounds1d[0] = -1
    bounds1d[1:-1] = -1 + weights[:-1].cumsum()
    bounds1d[-1] = 1
    # Convert the bounds to degrees of latitude on [-90, 90]:
    bounds1d = np.rad2deg(np.arcsin(bounds1d))
    bounds2d = np.empty([nlat, 2])
    bounds2d[:, 0] = bounds1d[:-1]
    bounds2d[:, 1] = bounds1d[1:]
    # Convert the roots from the interval [-1, 1] to latitude values on the
    # interval [-90, 90] degrees:
    latitudes = np.rad2deg(np.arcsin(roots))
    return latitudes, bounds2d

def local_MMdata_out(oldN, newN, pathh,T, name,save_PSI=False,path=None):
    ## check file and load it
    if os.path.isfile(  pathh + oldN ):
        os.rename( pathh + oldN, pathh + newN)

    PSI = np.loadtxt( pathh + newN )

    T31 = [ 90, 23 ]
    time = int( PSI.shape[ 0 ] / 3 )

    PSI_a = ( PSI[ 0::3, : ] ).reshape( (time, T31[ 1 ], T31[ 0 ] ) )
    PSI_b = ( PSI[ 1::3, : ] ).reshape( (time, T31[ 1 ], T31[ 0 ] ) )
    PSI_c = ( PSI[ 2::3, : ] ).reshape( (time, T31[ 1 ], T31[ 0 ] ) )
    if save_PSI:
        np.save(path+'PSI_a_'+name,PSI_a[:T])
        np.save(path+'PSI_b_'+name,PSI_b[:T])
        np.save(path+'PSI_c_'+name,PSI_c[:T])
        print("saved "+path+'PSI_a,b,c_'+name)
        
# save the dimensionalized streamfunction to /scratch/hz1994/blocking/data_MMmodel/dim/
def prepare_PSI_Z(PSI,PSI_scale,dim_path, name,level):
    lon=np.linspace(0,360,90,endpoint=False)
    lon=(lon + 180) % 360 - 180
    lat=gaussian_latitudes(46//2)[0][23:]
    f0=np.sqrt(3)*2*np.pi/(86400)
    g=9.806 #m/s**2

    Z_xr = xr.DataArray((PSI)*PSI_scale*f0/g, 
    coords={'time':np.arange(PSI.shape[0]),'latitude': lat ,'longitude': lon,}, 
    dims=["time","latitude","longitude",]) 
    Z_xr = Z_xr.sortby(Z_xr.longitude) 
#         PSI_xr.attrs["timeunit"] = "days"
#         PSI_xr.attrs["unit"] = "m**2/s"
#         PSI_xr.attrs["quantity"] = level+" hPa level stream function"
#         PSI_xr.to_netcdf(dim_path+"dim_PSI"+level+'_'+name+".nc")
    Zprime_xr=Z_xr-Z_xr.mean(dim='time')
    Z_xr.to_netcdf(dim_path+"dim_Z"+level+'_'+name+".nc")
    print("saved level="+level+" dim_Z")
    Zprime_xr.to_netcdf(dim_path+"dim_Zprime"+level+'_'+name+".nc")
    print("saved level="+level+" dim_Zprime")

    
def prepareregionaldata(zprime,name,path):
    if min(zprime.longitude.data)==-180:
        zprime_atl=zprime.sel(latitude=[59.99427 , 63.864226],longitude=[-4,0,4],method='nearest')\
                            .mean(dim=['latitude','longitude'])
        zprime_ural=zprime.sel(latitude=[71.602968, 75.471138],longitude=[52,56,60],method='nearest')\
                            .mean(dim=['latitude','longitude'])
        zprime_pac=zprime.sel(latitude=[52.253789, 56.124102],longitude=[-164,-160,-156],method='nearest')\
                            .mean(dim=['latitude','longitude'])
        zprime_atl.to_netcdf(path+"Atlantic_zprime_"+name+".nc")
        zprime_ural.to_netcdf(path+"Ural_zprime_"+name+".nc")
        zprime_pac.to_netcdf(path+"Pacific_zprime_"+name+".nc")
        print("saved regional data")
    else:
        print('longitudes needs to be -180,...,180')
        
#When necessary, interpolate Z for some certain latitudes
def interpolate_Z():
    print("need to check whether the paths and filenames are correct, I didn't debug it")
    from scipy.interpolate import interp2d
    for level in ['200','500','800']:
        PSI=xr.open_dataarray(dim_path+"dim_Z%s.nc" %level)

        y=Z.latitude.data
        x=Z.longitude.data
#         X_new = np.arange(x.min(), (x.max())+2, 2)
        X_new = x
        Y_new=np.hstack( (y,np.array([36,40,44,56,60,64,76,80,84])))
        Y_new=np.sort(Y_new)

        z=Z.data
        z_new=np.zeros((z.shape[0],Y_new.size,X_new.size,))
        # interpolate for each timeframe
        for t in range(z_new.shape[0]):
#             f=interp2d(x, y, z[t],kind='cubic')
            f=interp2d(x, y, z[t],kind='linear')
            z_new[t] = f(X_new, Y_new)

        data_xr = xr.DataArray(z_new, 
        coords={'time':np.arange(z_new.shape[0]),'latitude': Y_new ,'longitude': X_new,}, 
        dims=["time","latitude","longitude",]) 
        data_xr.attrs["timeunit"] = "days"
        data_xr.attrs["unit"] = "m**2/s**2"
        data_xr.attrs["quantity"] = "Interpolated %s hPa level geopotential" %level
        data_xr.to_netcdf(dim_path+"linear_intp_dim_Z%s.nc" %level)
        
#lower-pass cutoff smooth data        
def butter_lowpass_filter(data, lowpassdays,  order=2):
    fs=data.shape[0]
    cutoff=fs/lowpassdays
    nyq=0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y
def smooth_data(dim_path,quantity,height,name,nameoutput,lowpassdays):
    Z=xr.open_dataarray(dim_path+"dim_"+quantity+str(height)+"_"+name+".nc")
    Z_low_pass=Z.data*0
    for i in range(Z.shape[1]):
        for j in range(Z.shape[2]):
            Z_low_pass[:,i,j]=butter_lowpass_filter(Z[:,i,j].data, lowpassdays,  order=2)
    print("saving...")
    Z.copy(deep=True, data=Z_low_pass).to_netcdf(dim_path+"dim_"+quantity+str(height)+"_"+nameoutput+\
                                                   "_lowpass%ddys.nc"%lowpassdays)
    print("saved "+dim_path+"dim_"+quantity+str(height)+"_"+nameoutput+\
                                                   "_lowpass%ddys.nc"%lowpassdays)
    
def smoothzprime(dim_path,height,name,nameoutput,lowpassdays):
    Z=xr.open_dataarray(dim_path+"dim_Zprime"+str(height)+"_"+name+".nc")
    Z_low_pass=Z.data*0
    for i in range(Z.shape[1]):
        for j in range(Z.shape[2]):
            Z_low_pass[:,i,j]=butter_lowpass_filter(Z[:,i,j].data, lowpassdays,  order=2)
    print("saving...")
    Z.copy(deep=True, data=Z_low_pass).to_netcdf(dim_path+"dim_Zprime"+str(height)+"_"+nameoutput+\
                                                   "_lowpass%ddys.nc"%lowpassdays)
    print("saved "+dim_path+"dim_Zprime"+str(height)+"_"+nameoutput+\
                                                   "_lowpass%ddys.nc"%lowpassdays)
