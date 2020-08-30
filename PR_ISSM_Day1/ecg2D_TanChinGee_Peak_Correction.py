# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:35:35 2019

@author: Jacky 
"""

import pandas as pd
from scipy.signal import find_peaks as findPeaks
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np

l2D = pd.read_csv('ecg2D.csv',header=None)

plt.style.use('ggplot')
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelright'] = True            
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False

def alsbase(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2],shape=(L,L-2))
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.spdiags(w,0,L,L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def peakNcorrection(x=l2D[0], e='1st ECG'):
    #(Pks,_) = findPeaks(x,prominence=0.75,distance=75)
    ecgbase = alsbase(x, 10^5, 0.000005, niter=50)
    ecgcorr = x - ecgbase
    (Pks,_) = findPeaks(ecgcorr,prominence=0.75,distance=75)
    
    plt.figure(figsize=(30,20))
    plt.subplot(211)
    plt.title(e + ' - Orignal signal',fontsize=30)
    plt.plot(x)
    plt.plot(ecgbase, color = 'C1', linestyle='dotted')
    plt.subplot(212)
    plt.title(e + ' - Identified peaks & baseline corrected',fontsize=30)
    plt.plot(ecgcorr)
    plt.plot(Pks,ecgcorr[Pks],'x', ms = 20, mew = 5)

peakNcorrection(l2D[0],'1st ECG')
peakNcorrection(l2D[1], '2nd ECG')
peakNcorrection(l2D[2], '3rd ECG')