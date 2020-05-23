#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:32:47 2020

@author: srivenkat
"""

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

#Defining lambda functons to calculate input signal
func_dict = {"cos0.86t":lambda t: np.cos(0.86*t)**3,
             "sin1.4t":lambda t: np.sin(np.sqrt(2)*t),
             "chirp":lambda t: np.cos(24*t + 8*t*t/np.pi)}

def findDFT(x,t_lim,N,wnd = True,disp=True,xlim=4):
    t = np.linspace(-t_lim,t_lim,N+1);t = t[:-1]
    dt = t[1] - t[0]
    fmax = 1/dt
    f = 0
    if type(x) is np.ndarray:
        y = x
    elif type(x) is str:
        y = func_dict[x](t)
        f = 1
    else:
        raise Exception("Enter a vector or a dictionary key")
    n = np.arange(N)
    if wnd:
        wnd = (0.54 + 0.46*np.cos(2*np.pi*n/(N)))
        y = y*fft.fftshift(wnd)
    #arbitrarily set First element to 0
    y[0]=0
    Y = fft.fftshift(fft.fft(fft.fftshift(y)))/N  #Clipping and normalising the DFT
    if disp:
        maxima = abs(Y).max()
        wo = np.where(abs(Y)==maxima)
        plt.figure()
        plt.plot(t,y,lw=2)
        plt.title(x if f else "input signal")
        plt.show()
        w = np.linspace(-fmax*np.pi,fmax*np.pi,N+1);w = w[:-1]
        plt.figure()
        plt.suptitle("{}-spectra".format(x) if f else "DFT")
        plt.subplot(2,1,1)
        plt.plot(w,abs(Y),w[wo],abs(Y[wo]),"bo",lw=2)
        plt.xlim([xlim,-xlim])
        plt.ylabel("magnitude")
        plt.grid(True)
        plt.subplot(2,1,2)
        plt.plot(w,np.angle(Y),"ro",lw=2)
        plt.xlim([xlim,-xlim])
        plt.ylabel("phase")
        plt.grid(True)
        plt.show()
    return Y

def estimateVEC(x):
    Y = findDFT(x,np.pi,128,wnd=True,disp=True)
    dt = 2*np.pi/128   #defined only for 128 element vectors
    fmax = 1/dt
    w = np.linspace(-np.pi*fmax,np.pi*fmax,129);w=w[:-1]
    ind = np.asarray(np.where(abs(Y)==abs(Y).max()))    #find index of fundamental freq
    try:
        ind = ind[0][1]
    except:
        try:
            ind = ind[1]
        except:
            pass
    delta = np.angle(Y[ind])    #finding phase shift in time domain
    wo = w[ind]
    return wo,delta

findDFT("sin1.4t",np.pi,64,wnd=False,disp=True)
findDFT("sin1.4t",np.pi,64,wnd=True,disp=True)
findDFT("sin1.4t",4*np.pi,256,wnd=True,disp=True)

findDFT("cos0.86t",4*np.pi,256,wnd=True,disp=True)
findDFT("cos0.86t",4*np.pi,256,wnd=False,disp=True)

f = 1
d = 2
t = np.linspace(-np.pi,np.pi,129);t = t[:-1]
x = np.cos(f*t + d)     #create arbitrary sinsusoidal vector
wo,delta = estimateVEC(x)
print("calculated {}, actual {}".format(wo,f))
print("calculated {}, actual {}".format(delta,d))

a = 0.1
x += a*np.random.randn(128)     #add gaussian noise
wo,delta = estimateVEC(x)
print("calculated omega{}, actual omage {}".format(wo,f))
print("calculated delta{}, actual delta{}".format(delta,d))

findDFT("chirp",np.pi,1024,wnd=False,disp=True,xlim=80)
t = np.linspace(-np.pi,np.pi,1025);t = t[:-1];dt = t[1]-t[0]
x = func_dict["chirp"](t)
x_mat = x.reshape((64,16))
x_t = x_mat.transpose()

Y_t = np.zeros((16,64))
i = 0
for row in x_t:
    Y_t[i] = abs(findDFT(row,np.pi,64,wnd = False,disp=False))      #find DFT for a specific time range
    i += 1
Y = Y_t.transpose()
fmax = 1/dt
wx = np.linspace(-np.pi*fmax,np.pi*fmax,65);wx=wx[:-1]
wy = np.linspace(-np.pi,np.pi,17);wy=wy[:-1]
Wx,Wy = np.meshgrid(wy,wx)
ax = p3.Axes3D(plt.figure())
ax.plot_surface(Y,Wy,Wx,rstride=1,cstride=1,cmap='jet')     #plot time-frequency surface plot
plt.show()


