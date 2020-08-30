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

plt.style.use('ggplot')
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelright'] = True            
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False

l2D = pd.read_csv('ecg2D.csv',header=None)
ECGs = l2D[1].values

def peakIdentified(x=ECGs):
    (Pks,_) = findPeaks(x,prominence=0.75,distance=75)
    return Pks

def extractECG(ecg=ECGs,pks=peakIdentified(ECGs),offset=15):
    plt.figure(figsize=(30,20))
    plt.subplot(211)
    plt.title('Orignal signal',fontsize=30)
    plt.plot(ecg)
    plt.subplot(212)
    plt.title('Identified peaks',fontsize=30)
    plt.plot(ecg)
    plt.plot(pks,ecg[pks],'x', ms = 20, mew = 5)
    
    ecg_segment=[]
    start_segment=0
    end_segment=0
    
    for i in range(len(pks)):
        start_segment=pks[i]-offset
        
        if i == len(pks) - 1:
            end_segment=len(ecg) - 1
        else:
            end_segment=pks[i+1]-offset
        
        ecg_segment.append(ecg[start_segment:end_segment])
        
    return ecg_segment

extractSegment = extractECG(ECGs,peakIdentified(ECGs),offset=15)

def DTW(x=extractSegment[0], y=extractSegment[1]):
    
    #Compute distance matrix
    dists = np.zeros((len(y),len(x)))

    for i in range(len(y)):
        for j in range(len(x)):
            dists[i,j] = (y[i]-x[j])**2
    
    def pltDistances(dists,xlab='X',ylab='Y',clrmap='viridis'):
        imgplt = plt.figure()
        plt.imshow(dists,
                   interpolation='nearest',
                   cmap=clrmap)
    
        plt.gca().invert_yaxis()
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid()
        plt.colorbar()
    
        return imgplt
    
    #Compute accumulated cost matrix
    acuCost = np.zeros(dists.shape)
    acuCost[0,0] = dists[0,0]

    for j in range(1,dists.shape[1]):
        acuCost[0,j] = dists[0,j]+acuCost[0,j-1]

    for i in range(1,dists.shape[0]):
        acuCost[i,0] = dists[i,0]+acuCost[i-1,0]

    for i in range(1,dists.shape[0]):
        for j in range(1,dists.shape[1]):
            acuCost[i,j] = min(acuCost[i-1,j-1],
                               acuCost[i-1,j],
                               acuCost[i,j-1])+dists[i,j]
    
    #Search the optimal path

    i = len(y)-1
    j = len(x)-1
    path = [[j,i]]
    while (i > 0) and (j > 0):
        if i==0:
            j = j-1
        elif j==0:
            i = i-1
        else:
            if acuCost[i-1,j] == min(acuCost[i-1,j-1],
                      acuCost[i-1,j],
                      acuCost[i,j-1]):
               i = i-1
            elif acuCost[i,j-1] == min(acuCost[i-1,j-1],
                                    acuCost[i-1,j],
                                    acuCost[i,j-1]):
               j = j-1
            else:
                i = i-1
                j = j-1
        path.append([j,i])
    path.append([0,0])
    
    def pltCostAndPath(acuCost,path,clrmap='viridis'):
        px = [pt[0] for pt in path]
        py = [pt[1] for pt in path]
    
        imgplt = pltDistances(acuCost,
                              clrmap=clrmap)
    
        plt.plot(px,py)
    
        return imgplt
    
    # Draw accumulative cost and path 
    costPathPlot = pltCostAndPath(acuCost,path,clrmap='Reds')

    plt.show()
    
    # Draw warp on path
    def pltWarp(s1,s2,path,xlab='idx',ylab='Value'):
        imgplt = plt.figure()
    
        for [idx1,idx2] in path:
            plt.plot([idx1,idx2],[s1[idx1],s2[idx2]],
                     color='C4',
                     linewidth=2)
            plt.plot(s1,
                     'o-',
                     color='C0',
                     markersize=3)
            plt.plot(s2,
                     's-',
                     color='C1',
                     markersize=2)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
    
        return imgplt
    
    wrapping = pltWarp(x,y,path)
    
    plt.show()

#Run DTW on segment 1 and 2
DTW(extractSegment[0], extractSegment[1])
print('Run DTW on segment 1 and 2')
print()

#Run DTW on segment 2 and 3
DTW(extractSegment[1], extractSegment[2])
print('Run DTW on segment 2 and 3')
print()

#Run DTW on segment 2 and 6
DTW(extractSegment[1], extractSegment[5])
print('Run DTW on segment 2 and 6')
print()