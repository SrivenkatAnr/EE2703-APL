#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:30:59 2020

@author: srivenkat
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

#task 1
num = np.poly1d([1,0.5])
den = np.poly1d([1,1,2.5])
den = np.polymul(den,np.poly1d([1,0,2.25]))
X = sig.lti(num,den)    #finding X(s) as laplace of x(t)
time,x = sig.impulse(X,None,np.linspace(0,50,500))      #finding x(t) as inverse
plt.plot(time,x)
plt.title("x(t), decay = 0.5")
plt.xlabel("time")
plt.ylabel("x")
plt.show()

#task 2
num = np.poly1d([1,0.05])
den = np.poly1d([1,0.1,2.2525])
den = np.polymul(den,np.poly1d([1,0,2.25]))
X = sig.lti(num,den)       #finding X(s) as laplace of num/den
time,x = sig.impulse(X,None,np.linspace(0,50,500))      #finding x(t) as inverse
plt.plot(time,x)
plt.title("x(t), decay = 0.05")
plt.xlabel("time")
plt.ylabel("x")
plt.show()

#task 3
H = np.poly1d([1,0,2.25])
H = sig.lti(1,H)       #finding H(s) as laplace of num/den
time = np.linspace(0,50,500)
for i in range(5):
    var = 1.4 + i*0.05
    f = np.cos(var*time)*np.exp(-0.05*time)     #f is function value
    time,x,svec = sig.lsim(H,f,time)      #finding h(t) as inverse
    plt.plot(time,x,label = "freq = {}".format(np.round(var,2)))
    #plt.legend()
    #plt.title("freq = {}".format(np.round(var,2)))
    #plt.show()
plt.title("Multiple freq. plot")
plt.legend()
plt.show()

#task 4
"""
X' = sX-X(0)
X'' = s^2X - sX(0) - X'(0)
s^2 X(s) - s + X(s) - Y(s) = 0
2(Y(s)-X(s)) + s^2Y(s) = 0
2(s^2X(s) - s) + s^2(s^2X(s) - s + X(s)) = 0
X(s) = (s^3 + 2s)/(s^4 + 3s^2)
"""
Xnum = np.poly1d([1,0,2,0])
Xden = np.poly1d([1,0,3,0,0])
X = sig.lti(Xnum,Xden)    #finding X(s) as laplace of num/den
time,x = sig.impulse(X,None,np.linspace(0,20,200))      #finding x(t) as inverse
plt.plot(time,x)
plt.title("x(t)")
plt.xlabel("time")
plt.ylabel("x")
plt.show()
Ynum = np.polymul(Xnum,[1,0,1]) - np.polymul([1,0],Xden)
Y = sig.lti(Ynum,Xden)    #finding Y(s) as laplace of num/den
time,y = sig.impulse(Y,None,np.linspace(0,20,200))      #finding y(t) as inverse
plt.plot(time,y)
plt.title("y(t)")
plt.xlabel("time")
plt.ylabel("y")
plt.show()

#task 5
"""
R = 100
L = 0.000001s
C = 1000000/s
H = C/(R+L+C)
"""
H = sig.lti([1000000],[0.000001,100,1000000])    #finding H(s) as laplace of num/den
w,mag,phase = sig.bode(H)    #finding bode parameters of H
fig,a = plt.subplots(2,1)
a[0].set_title("Bode plot of H(s)")
a[0].semilogx(w,mag)
a[0].set_ylabel(r'$|H(s)|$')
a[1].semilogx(w,phase)
a[1].set_ylabel(r'$\angle(H(s))$')
plt.show()

#task 6
time = np.linspace(0,0.01,100000)
x = np.cos(1000*time) - np.cos(1000000*time)
time,y,svec = sig.lsim(H,x,time)      #finding y(t) as inverse
plt.plot(time,y)
plt.title(r'$v_{o}(t)$')
plt.xlabel("time")
plt.ylabel("V")
plt.show()
plt.plot(time[:300],y[:300])    #plot first 300 points of Vo
plt.title("initial response")
plt.xlabel("time")
plt.ylabel("V")
plt.show()
plt.plot(time[32000:35000],y[32000:35000])    #plot zoomed in image of Vo
plt.title("Very small time scale")
plt.xlabel("time")
plt.ylabel("V")
plt.show()




