#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:39:41 2020

@author: srivenkat
"""
import sympy as sym
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np
import math as ma

s = sym.symbols('s')

def lowpass(R1,R2,C1,C2,G,Vi):
    A = sym.Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],[0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b = sym.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return (A,b,V)

def highpass(R1,R3,C1,C2,G,Vi):
    A = sym.Matrix([[0,0,1,-1/G],[(-s*C2*R3)/(1+s*C2*R3),1,0,0],[0,-G,G,1],[-s*C1-s*C2-1/R1,s*C2,0,1/R1]])
    b = sym.Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return (A,b,V)

def plotbode(A,b,Vo,title):
    ww = np.logspace(0,8,801)
    ss = 1j*ww
    hf = sym.lambdify(s,Vo)
    v = hf(ss)
    plt.loglog(ww,abs(v))
    plt.grid(True)
    plt.title(title)
    plt.ylabel("Amplitude in dB-->")
    plt.xlabel("Frequency-->")
    plt.show()

def plotsig(x,title,time,y=None):
    if y is None:
        plt.plot(time,x)
    else:
        plt.plot(time,y,'orange',label="input")
        plt.plot(time,x,'b',label="output")
        plt.legend()
    plt.title(title)
    plt.xlabel("Time -->")
    plt.ylabel("Amplitude-->")
    plt.show()

def Hcalc(Lap):
    Lap = Lap.simplify()
    Lap = Lap.expand()
    num,den = sym.fraction(Lap)
    num_deg = sym.degree(num)
    den_deg = sym.degree(den)
    num_c = np.empty(num_deg+1)
    den_c = np.empty(den_deg+1)
    for i in range(num_deg+1):
        num_c[i] = num.coeff(s,i)
    for i in range(den_deg+1):
        den_c[i] = den.coeff(s,i)
    num_c = num_c[::-1]
    den_c = den_c[::-1]
    H = sig.lti(num_c,den_c)
    return(H)

t = np.linspace(0,0.001,100)

Vi = 1
A,b,V = lowpass(10000,10000,1e-9,1e-9,1.586,Vi)
Vl=V[3]
plotbode(A,b,Vl,"Low Pass-Transfer Function-Magnitude Plot")
Vl = Hcalc(Vl)
t,Vl_imp = sig.impulse(Vl,None,t)
plotsig(Vl_imp,"Low Pass-Impulse Response",t)

A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,Vi)
Vh=V[3]
plotbode(A,b,Vh,"High Pass-Transfer Function-Magnitude Plot")
Vh = Hcalc(Vh)
t,Vh_imp = sig.impulse(Vh,None,t)
plotsig(Vh_imp,"High Pass-Impulse Response",t)

Vi = 1/s
A,b,V = lowpass(10000,10000,1e-9,1e-9,1.586,Vi)
Vl_step=V[3]
Vl_step = Hcalc(Vl_step)
t,Vl_step = sig.impulse(Vl_step,None,t)
plotsig(Vl_step,"Low Pass-Step Response",t)

A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,Vi)
Vh_step=V[3]
Vh_step = Hcalc(Vh_step)
t,Vh_step = sig.impulse(Vh_step,None,t)
plotsig(Vh_step,"High Pass-Step Response",t)

t = np.linspace(0,0.01,100000)

Vl_input = np.sin(2e3*ma.pi*t) + np.cos(2e6*ma.pi*t)
t,Vl_output,svec = sig.lsim(Vl,Vl_input,t)
plotsig(Vl_output,"Low Pass-Filter Output",t,Vl_input)

Vh_input = np.sin(2e3*ma.pi*t) + np.cos(2e6*ma.pi*t)
t,Vh_output,svec = sig.lsim(Vh,Vh_input,t)
plotsig(Vh_output,"High Pass-Filter Output",t,Vh_input)

Vh_ds = np.sin(2e6*np.pi*t)*np.exp(-5e4*t)
t,Vh_output,svec = sig.lsim(Vh,Vh_ds,t)
plt.plot(t,Vh_output,t,Vh_ds,label=["output","input"])
plt.xlim([0,1e-5])
plt.legend()
plt.title("Damped Sinusoid-High Pass fliter response")
plt.show()
