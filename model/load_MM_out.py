#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:49:15 2023

@author: sspringe
"""

import numpy as np
import matplotlib.pyplot as plt
import os

## check file and load it
oldN = 'fort.20'
newN = 'fort20.txt'
pathh = ''

if os.path.isfile(  pathh + oldN ):
    os.rename( pathh + oldN, pathh + newN)


PSI = np.loadtxt( pathh + newN )

T31 = [ 90, 23 ]
time = int( PSI.shape[ 0 ] / 3 )

PSI_a = ( PSI[ 0::3, : ] ).reshape( (time, T31[ 1 ], T31[ 0 ] ) )
PSI_b = ( PSI[ 1::3, : ] ).reshape( (time, T31[ 1 ], T31[ 0 ] ) )
PSI_c = ( PSI[ 2::3, : ] ).reshape( (time, T31[ 1 ], T31[ 0 ] ) )


