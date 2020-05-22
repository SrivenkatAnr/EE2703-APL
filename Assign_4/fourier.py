#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:12:05 2020

@author: srivenkat
"""
import numpy as np
import scipy as sp
import scipy.integrate as integrate
import math as ma
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
#to find the function vectors
def exp_create(time):
    return np.exp(time)

def coscos_create(time):
    return np.cos(np.cos(time))
#since we operate with 2 functions, this returns any one of them based on key
def func(key,x):
    if key == 0:
        return np.exp(x)
    else:
        return np.cos(np.cos(x))

#fourier series generation by direct integration
        
#defining functions to use with integrate.quad
def u_gen(x,k,key):
    return np.cos(k*x)*func(key,x)

def v_gen(x,k,key):
    return np.sin(k*x)*func(key,x)

def fourier_coeff(key,l,a):
    a.append(integrate.quad(u_gen,0,2*ma.pi,args=(0,key))[0]/(2*ma.pi))
    for k in range(1,l):     
        a.append(integrate.quad(u_gen,0,2*ma.pi,args=(k,key))[0]/ma.pi)
        a.append(integrate.quad(v_gen,0,2*ma.pi,args=(k,key))[0]/ma.pi)
#synthesizing the function for checking
def fourier_synth(t,a):
    f_val = a[0]
    n = 1
    while n < len(a):
        f_val += a[n]*sp.cos((n+1)*t/2) + a[n+1]*sp.sin((n+1)*t/2)
        n = n + 2
    return f_val
#checking if function if periodic
def per_check(key,data):
    data = np.round(data,3)
    check = 0
    per = -1
    for i in range(1,len(data)):
        per = -1
        if data[i] == data[0]:
            per = i
            for j in range(1,len(data)-per):
                if data[j+per] != data[j]:
                    break
                elif j == len(data)-per-1:
                    check = 1
                    break
        if check == 1:
            print('{0}(x) function is periodic'.format(func_array[key]))
            break
    if check == 0:
        print('{0}(x) function is not periodic'.format(func_array[key]))
#defining x-axis variables and getting fourier coeff.
time = np.linspace(-2*ma.pi,4*ma.pi,501)
l = 26
func_array = ['exp','coscos']
period = np.linspace(0,2*ma.pi,167)
exp_x = exp_create(time)     #array of exp(x) values
cos_cos_x = coscos_create(time)     #array of coscos(x) values
exp_x_per = exp_create(period)      #array of periodic extension of exp(x) values
exp_x_perext = np.concatenate([exp_x_per,exp_x_per])
exp_x_perext = np.concatenate([exp_x_perext,exp_x_per])
e_coeff = []
f_seq_e = []
cc_coeff = []
f_seq_cc = []

fourier_coeff(0,l,e_coeff)
fourier_coeff(1,l,cc_coeff)

#fourier synthesis
for t in time:
    f_seq_e.append(fourier_synth(t,e_coeff))
    f_seq_cc.append(fourier_synth(t,cc_coeff))

#plotting original exp(x) and cos(cos(x)) graphs
figure, graph1 = plt.subplots(figsize=(6.4,6.4))
graph1.semilogy(time/3,exp_x,'orange',label='exp(x)')
graph1.semilogy(time/3,exp_x_perext,'blue',label='exp(x) extension')
graph1.legend(loc='upper right')
graph1.set_title('original exp(x)')
graph1.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
graph1.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
graph1.grid('true')

figure, graph2 = plt.subplots(figsize=(6.4,6.4))
graph2.plot(time/3,cos_cos_x,'orange',label='cos(cos(x))')
graph2.legend(loc='upper right')
graph2.set_title('original cos(cos(x))')
graph2.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
graph2.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
graph2.grid('true')

plt.show()

print('')   #just for output alignment
per_check(0,exp_x)
per_check(1,cos_cos_x)

#semilog plot of exp(x)
figure, graph3 = plt.subplots(figsize=(6.4,6.4))
graph3.semilogy(np.abs(e_coeff),'ro',label='exp fourier coeff')
graph3.set_title('semilog plot of exp(x)')
graph3.legend(loc='upper right')
graph3.grid('true')

#loglog plot of exp(x)
figure, graph4 = plt.subplots(figsize=(6.4,6.4))
graph4.loglog(np.abs(e_coeff),'ro',label='exp fourier coeff')
graph4.set_title('loglog plot of exp(x)')
graph4.legend(loc='upper right')
graph4.grid('true')

#semilog plot of coscos(x)
figure, graph5 = plt.subplots(figsize=(6.4,6.4))
graph5.semilogy(np.abs(cc_coeff),'ro',label='coscos fourier coeff')
graph5.set_title('semilog plot of cos(cos(x))')
graph5.legend(loc='upper right')
graph5.grid('true')

#loglog plot of coscos(x)
figure, graph6 = plt.subplots(figsize=(6.4,6.4))
graph6.loglog(np.abs(cc_coeff),'ro',label='coscos fourier coeff')
graph6.set_yscale('log')
graph6.set_xscale('log')
graph6.set_title('loglog plot of cos(cos(x))')
graph6.legend(loc='upper right')
graph6.grid('true')

plt.show()

#fourier series generation by least squares alg
vect_x = np.linspace(0,2*ma.pi,401)
vect_x = vect_x[:-1]    #removing the last element for periodic integration
exp_vect = exp_create(vect_x)
coscos_vect = coscos_create(vect_x)

def lstsq_coeff(key,x):
    A = np.zeros(shape=[400,51])
    b = np.zeros(shape=[400,1])
    A[:,0] = 1
    b = np.transpose(func(key,x))
    for k in range(1,26):
        A[:,2*k-1] = np.cos(k*x)
        A[:,2*k] = np.sin(k*x)
    return sp.linalg.lstsq(A,b)[0],A,b

e_vect_coeff,e_cmat,e_bmat = lstsq_coeff(0,vect_x)
cc_vect_coeff,cc_cmat,cc_bmat = lstsq_coeff(1,vect_x)

vect_seq_e = np.dot(e_cmat,e_vect_coeff)
vect_seq_cc = np.dot(cc_cmat,cc_vect_coeff)

#plot exp(x) lstsq coeff
figure, graph7 = plt.subplots(figsize=(6.4,6.4))
graph7.semilogy(np.abs(e_vect_coeff),'go',label='exp lstsq coeff')
graph7.semilogy(np.abs(e_coeff),'ro',label='exp lstsq coeff')
graph7.set_title('exp(x) coeff from lstsq')
graph7.legend(loc='upper right')
graph7.grid('true')

#plot coscos(x) lstsq coeff
figure, graph8 = plt.subplots(figsize=(6.4,6.4))
graph8.semilogy(np.abs(cc_vect_coeff),'go',label='coscos lstsq coeff')
graph8.semilogy(np.abs(cc_coeff),'ro',label='coscos lstsq coeff')
graph8.set_title('cos(cos(x)) coeff from lstsq')
graph8.legend(loc='upper right')
graph8.grid('true')

#finding deviations in lstsq and integration
e_error = abs(np.subtract(e_coeff,e_vect_coeff))
e_maxerr = np.amax(e_error)
cc_error = abs(np.subtract(cc_coeff,cc_vect_coeff))
cc_maxerr = np.amax(cc_error)

print("\nMax error in finding fourier coefficients from integration is {} in exp(x) and {} in cos(cos(x)".format(e_maxerr,cc_maxerr))

#plot synthesised func from lstsq
figure, graph9 = plt.subplots(figsize=(6.4,6.4))
graph9.plot(vect_x/3,vect_seq_e,'g.',label='exp(x) lstsq estimate')
graph9.plot(vect_x/3,e_bmat,'yellow',label='exp(x) original')
graph9.legend(loc='upper right')
graph9.set_title('lstsq exp(x) error check')
graph9.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
graph9.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
graph9.grid('true')

#plot synthesised func from lstsq
figure, graph10 = plt.subplots(figsize=(6.4,6.4))
graph10.plot(vect_x/3,vect_seq_cc,'g.',label='cos(cos(x)) lstsq estimate')
graph10.plot(vect_x/3,cc_bmat,'yellow',label='cos(cos(x)) original')
graph10.legend(loc='upper right')
graph10.set_title('lstsq cos(cos(x)) error check')
graph10.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
graph10.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
graph10.grid('true')

plt.show()