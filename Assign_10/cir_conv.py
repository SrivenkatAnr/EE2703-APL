#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:42:02 2020

@author: srivenkat
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import *
import time

#Plotting function with default params
def plotSig(y,title = None,xlim=[0,100],xlab = 'Time', ylab = "Magnitude"):
    plt.plot(abs(y))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim(xlim)
    plt.title(title)
    plt.show()

#To convert string input into complex no.
def getComplex(a):
    a = a.replace('\n','').replace('i','j')
    return complex(''.join(a))

#Take input coefficients for fir and zadoff-chu sequence
with open("h.csv",mode='r') as file:
    h = np.array([float(c) for c in file.readlines()])

with open("x1.csv",mode='r') as file:
    zch = np.array([getComplex(c) for c in file.readlines()])

w,H = signal.freqz(h)   #calculate spectra
plt.figure(figsize=(6,11))

#bode plot for FIR filter
ax1 = plt.subplot(2,1,1)
ax1.set_title("Magnitude Plot")
ax1.plot(w,20*np.log10(abs(H)))
ax1.set_ylabel("Magnitude in dB")
ax1.set_xlabel("Frequency")
ax2 = plt.subplot(2,1,2)
ax2.set_title("Phase plot")
ax2.plot(w,np.unwrap(np.angle(H)))
ax2.set_ylabel("Phase in angles")
ax2.set_xlabel("Frequency")
plt.show()

#mixed freq time signal
n = np.arange(1,1025)
x = np.cos( 0.2*np.pi*n ) + np.cos( 0.85*np.pi*n )

plt.plot(n,x)
plt.title("Input signal")
plt.xlabel("Frequency")
plt.xlim([0,100])
plt.ylabel("Magnitude")
plt.show()

#Linear Convolvution
y_lin = np.convolve(x,h)
plotSig(y_lin,"Linear Conv. output")

#Circular Convolution
y_cir = ifft(fft(x) * fft( np.concatenate( (h,np.zeros((len(x)-len(h))) ) ) ))
plotSig(y_cir,"Circular Conv. output")

#Linear Convilution with DFTs
#using m=4, i.e., a 16 sample block
P = len(h)
h = np.concatenate( (h, np.zeros((16-P))) ); P = len(h)    #zero padding filter response
y_block = np.zeros((1024+P-1))
for i in range(0,len(x),16):
    xr = np.concatenate( (x[i:i+P], np.zeros((P-1))) )
    yr = np.real(ifft( fft(xr) * fft( np.concatenate((h, np.zeros((P-1)))) ) ))
    y_block[i:i+ 2*P -1] += yr
plotSig(y_block,"Block Conv. output")

zch_5 = np.roll(zch,5)    #cyclic shift zadoff-chu sequence
corr = ifft(np.multiply( fft(zch_5), np.conj(fft(zch)) ))     #calc corr. function
plotSig(corr,"Correlation output",[0,20])
