#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:13:30 2020

@author: srivenkat
"""
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import scipy.linalg as sp
import sys
import copy

if len(sys.argv) >= 5:      #check for invalid commandline inputs
    print("Enter arguments for Nx, Ny, radius, Niter. \nnote: Nx and Ny should be odd numbers")
    sys.exit(0)

Nx = 25     #default parameters
Ny = 25
radius = 8
Niter = 1500 

try:        #if no input is given, use the default value itself
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
    radius = sys.argv[3]
    Niter = int(sys.argv[4])
except:
    pass

phi = np.zeros(shape=[Nx,Ny])       #model the metal plate as Nx-Ny grid
x = np.arange(int(-1*Nx/2),int(Nx/2)+1)
y = np.arange(int(-1*Ny/2),int(Ny/2)+1)
Y,X = np.meshgrid(y,x)

ii = np.where(X*X + Y*Y <= radius*radius)       #index matrix for coordinates of the wire 
phi[ii] = 1.0
phi[1:-1,Ny-1] = 0.0        #setting bottom boundary to GND

fig,fig1 = plt.subplots(figsize=[6.4,6.4])      #plot Initial Potential
fig1.contourf(Y,X,phi,cmap='autumn')
fig1.plot(ii[0]-(Ny-1)/2,ii[1]-(Nx-1)/2,'ro',label='Potential = 1V')
fig1.set_xlim(-12,12)
fig1.set_ylim(-12,12)
fig1.set_title("Initial Potential")
fig1.legend()
fig1.grid('true')
plt.show()

err = []
cum_err = []
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1,1:-1] = 0.25*(phi[0:-2,1:-1]+phi[2:,1:-1]+phi[1:-1,2:]+phi[1:-1,0:-2])           #solve Laplace equation in N iterations
    phi[:,0] = phi[:,1]       #setting gradient to 0 at boundaries
    phi[:,Nx-1] = phi[:,Nx-2]
    phi[0,1:-1] = phi[1,1:-1]
    phi[Ny-1,1:-1] = phi[Ny-2,1:-1]
    phi[ii] = 1.0       #reinforcing bnd conditions
    phi[1:-1,Ny-1] = 0.0
    err.append(abs(phi-oldphi).max())
    cum_err.append(np.sum(err))

log_y_mat = np.log(np.transpose(err))       #modelling the error in a exponential function
x_mat = np.c_[np.arange(0,Niter),np.ones(shape = [Niter,1])]
fit1 = sp.lstsq(x_mat,log_y_mat)[0]     #lstsq fit for whole err array
fit2 = sp.lstsq(x_mat[500:-1],log_y_mat[500:-1])[0]     #lstsq fit for err values post 500 iterations

fig,fig2 = plt.subplots(figsize=[6.4,6.4])      #plot error
fig2.semilogy(np.arange(0,Niter,50),err[::50],'r',label='Actual error')
fig2.semilogy(np.arange(0,Niter,50),np.exp(np.dot(x_mat,fit1))[::50],'bo',markersize=8,label='Fit 1')
fig2.semilogy(np.arange(0,Niter,50),np.exp(np.dot(x_mat,fit2))[::50],'y^',markersize=8,label='Fit 2')
fig2.set_xlabel("interations-->")
fig2.set_ylabel("error magnitude-->")
fig2.set_title('Semilog plot of error')
fig2.grid('true')
fig2.legend()
plt.show()

fig,fig6 = plt.subplots(figsize=[6.4,6.4])      #plot error
fig6.loglog(np.arange(0,Niter,50),err[::50],'r',label='Actual error')
fig6.set_xlabel("interations-->")
fig6.set_ylabel("error magnitude-->")
fig6.set_title('loglog plot of error')
fig6.grid('true')
fig6.legend()
plt.show()

fig,fig5 = plt.subplots(figsize=[6.4,6.4])      #plot cumulative error
fig5.semilogy(abs(np.array(cum_err)))
fig5.set_xlabel("interations-->")
fig5.set_ylabel("error magnitude-->")
fig5.set_title('Semilog plot of cumulative error')
fig5.grid('true')
plt.show()

phi = phi.T

fig3=plt.figure(3)      #plot surface plot of phi
ax=p3.Axes3D(fig3)
ax.set_title('The 3-D surface plot of the potential')
surf = ax.plot_surface(Y,-X, phi, rstride=1, cstride=1,cmap='jet')

fig,fig4 = plt.subplots(figsize=[6.4,6.4])     #plot contour plot of phi
fig4.contour(Y,-X,phi,cmap='jet')
fig4.plot(ii[0]-Ny/2,ii[1]-Nx/2,'ro')
fig4.set_title('Contour plot of potential')
fig4.grid('true')
plt.show()

Jx = np.zeros(shape=[Ny,Nx])    #solve for Jx and Jy
Jy = np.zeros(shape=[Ny,Nx])

Jx[1:-1,:] = 0.5*(phi[2:,:]-phi[:-2 ,:])
Jy[:,1:-1] = 0.5*(phi[:,:-2]-phi[:,2:])

fig,fig5 = plt.subplots(figsize=[6.4,6.4])      #plot Jx and Jy
fig5.quiver(y+Ny/2,x+Nx/2,Jy[::-1,:],Jx[::-1,:])
fig5.plot(ii[0],ii[1],'ro')
fig5.set_title('Vector plot of current flow')
fig5.grid('true')

