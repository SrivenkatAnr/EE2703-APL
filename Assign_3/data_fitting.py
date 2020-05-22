#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:18:59 2020

@author: srivenkat
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special as sp
from sklearn.metrics import mean_squared_error

SIGMA = 'σ'     #declaring the greek symbols used
EPSILON = 'ε'
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")     #function to get subscript of a number

input_data = np.loadtxt('fitting.dat')      #extract data into numpy array
data = np.transpose(input_data)     #getting data samples for a specific σ in 1 row for ease of plotting
time = data[0]

'''Q3: multiple data sequences vs time'''
sigma_num=np.round(np.logspace(-1,-3,9),3)
fig,graph1 = plt.subplots(figsize=(6.4,6.4))
for i in range(9):
    label_name = '{0}{1}'.format(SIGMA,i+1).translate(SUB)
    label_name = '{0}={1}'.format(label_name,sigma_num[i])
    graph1.plot(time,data[i+1],label=label_name)
graph1.set_xlabel('t   --->')
graph1.set_ylabel('f(t)+noise   --->')
graph1.set_title("Q4. Data to be fitted to theory",fontsize=14)

'''Q4: get true value for A=1.05, B=-0.105 '''
true_value=[]
def func_g(t,A,B):      #func to predict g vector for assumed coefficients
    g_mat = A*sp.jn(2,t) + B*t
    return g_mat
for i in range(len(time)):
    true_value.append(func_g(time[i],1.05,-0.105))
true_value = np.asarray(true_value)     #convert list to array
graph1.plot(time,true_value,'black',label='true value')
graph1.legend(loc='upper right', labelspacing=0.1, fontsize=13)
plt.show()

'''Q5: plot error bars for col 1, σ=0.1'''
fig,graph2 = plt.subplots(figsize=(6.4,6.4))
graph2.set_title('Q5. Data points for %s = 0.10 along with exact function' % SIGMA,fontsize=14)
graph2.errorbar(time[0::5],data[1][0::5],sigma_num[0],fmt='ro',label='error bars')      #plot errorbars for downsampled array of column 1
graph2.plot(time,true_value,'black',label='$f(t)$')
graph2.set_xlabel('t   --->')
graph2.legend(loc='upper right', labelspacing=0.5, fontsize=13)
plt.show()

'''Q6: check if function calculated discretely is same as assumed true value'''
def vect_func_g(A0,B0,t):       #func to return g vector for discrete time samples
    data_col = sp.jn(2,t)
    time_col = np.transpose(t)
    M_mat = np.c_[data_col,time_col]    #to create matrix with column vectors
    p_mat = np.array([A0,B0])
    g_mat = np.dot(M_mat,p_mat)
    return g_mat,M_mat
g_mat,M_mat = vect_func_g(1.05,-0.105,time)
if (g_mat == true_value).all():     #for comparing 2 vectors
    print("\nQ6. Both vectors are equal")
else:
    print("\nQ6. Both vectors are different")

'''Q7: to calculate mean sq. error for different A,B values'''
A_set = np.arange(0,2.1,0.1)        #array of possible A values
B_set = np.arange(-0.2,0.01,0.01)        #array of possible B values
mse = np.zeros(shape=[21,21])
col_ref = np.transpose(data[1])
for i in range(21):
    for j in range(21):
        g,temp = vect_func_g(A_set[i],B_set[j],time)     #return g vector for a specific A,B value
        mse[i][j] = mean_squared_error(g,col_ref)       #using scipy funcion to calc. mean sq. error

'''Q8: make a contour plot of errors and finding minimas for specific A,B values'''
fig,graph3 = plt.subplots(figsize=(6.4,6.4))
cf = graph3.contour(A_set,B_set,mse,20)
graph3.plot(1.05,-0.105,'ro')
graph3.annotate("Exact Value",xy=(1.05,-0.105),xytext=(1.10,-0.100))    #to plot and label exact estimate 
graph3.set_aspect(aspect=8)
graph3.set_title('Q8. Contour plot of %sᵢⱼ' % EPSILON,fontsize=14)
graph3.clabel(cf,[0,0.025,0.05,0.1],inline=1)       #marking specific labels
fig.colorbar(cf,shrink=0.66,extend='both')
plt.show()

'''Q9: finding best estimate for A,B'''
AB_soln = scipy.linalg.lstsq(M_mat,true_value)
AB_optimal = AB_soln[0]
print('\nQ9. Best estimate of A and B is:',AB_optimal)   #optimal value of A,B

'''Q10: calculating A and B for different datasets and plotting the error estimates'''
fig,graph4 = plt.subplots(figsize=(6.4,6.4))
A_err = []
B_err = []
for i in range(len(data)-1):
    AB_collect = scipy.linalg.lstsq(M_mat,data[i+1])    #func to optimise M.x=P equation
    A_err.append(abs(AB_collect[0][0]-AB_optimal[0]))
    B_err.append(abs(AB_collect[0][1]-AB_optimal[1]))

graph4.plot(sigma_num,A_err,'r--o',label='A_error')
graph4.plot(sigma_num,B_err,'g--s',label='B_error')
graph4.set_xlabel('Noise standard deviation   --->')
graph4.set_ylabel('Absolute error   --->')
graph4.set_title('Q10. Variation of error with noise',fontsize=14)
graph4.legend(loc='upper right',fontsize=13)
plt.show()

'''Q11: analysing the A,B error estimates in loglog plots'''
fig,graph5 = plt.subplots(figsize=(6.4,6.4))
graph5.set_title('Q11. Variation of error with noise in loglog plot',fontsize=14)
graph5.stem(sigma_num,A_err,linefmt='r--',markerfmt='ro',basefmt='none',use_line_collection='true',label='A_error')
graph5.stem(sigma_num,B_err,linefmt='g--',markerfmt='go',basefmt='none',use_line_collection='true',label='B_error')
graph5.set_xscale('log')    #setting logarithmic scale in x and y axes
graph5.set_yscale('log')
graph5.set_xlabel('Noise standard deviation   --->')
graph5.set_ylabel('Absolute error   --->')
graph5.legend(loc='upper right',fontsize=13)
plt.show()