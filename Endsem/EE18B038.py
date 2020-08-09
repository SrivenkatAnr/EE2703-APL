"""
@author: srivenkat
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp

#pre-defined constants
Lx = 10
Ly = 20
Er = 2

#user-defined constants
delta = 0.1    #resolution of grid
Niter = 30000    #No. of iterations

M = int(Lx/delta)    #No. of nodes parallel to x
N = int(Ly/delta)    #No. of nodes parallel to y

def SolveLaplacian(k,Niter,delta=0.1,M=100,N=200,acc=1e-6):
    x = np.linspace(0,Lx,M)
    y = np.linspace(0,Ly,N)
    Y,X = np.meshgrid(y,x)   #Taking a grid across the tank

    phi = np.ones(shape=[M,N])   #Defining a initial potential distribution

    #Enforcing boundary conditions
    phi[:,N-1] = 1.0
    phi[:,0] = 0.0
    phi[0,:] = 0.0
    phi[M-1,:] = 0.0

    #Defining error vectors
    err = []
    cum_err = []
    for i in range(Niter):
        oldphi = phi.copy()
        #Enforcing Laplacian Equation at all coordinates
        phi[1:-1,1:-1] = 0.25*(oldphi[0:-2,1:-1]
                               + oldphi[2:,1:-1]
                               + oldphi[1:-1,2:]
                               + oldphi[1:-1,0:-2] )

        #Updating at the interface separately
        phi[1:-1,k] = (Er*phi[1:-1,k-1] + phi[1:-1,k+1])/(1+Er)

        #Taking max change as error
        err.append(abs(phi-oldphi).max())
        if(err[-1]<acc):
            break
        cum_err.append(np.sum(err))

    #Taking a double ended differential for Electric field
    Ex = np.zeros(shape=[M,N])
    Ey = np.zeros(shape=[M,N])
    Ex[1:-1,:] = (phi[2:,:]-phi[:-2,:])/delta
    Ey[:,1:-1] = (phi[:,2:]-phi[:,:-2])/delta

    return(X,Y,phi,Ex,Ey,err,i)

ratio = 0.5    #H/Ly ratio
h = ratio*Ly
k = int(ratio*N)
X,Y,phi,Ex,Ey,err,Nover = SolveLaplacian(k,Niter)

En_air = Ey[1:-1,k+1]    #Normal compinent in air
En_di = Ey[1:-1,k-1]    #Normal compinent in dielectric

Et_air = Ex[1:-1,k+1]    #Tangential compinent in air
Et_di = Ex[1:-1,k-1]    #Tangential compinent in dielectric

E_ratio = En_air/En_di
print("\n E2/E1 mean is {0}, variance is {1}\n ".format(E_ratio.mean(),E_ratio.var()))

theta = np.arctan(Ex/Ey)    #Angle of incidence
sine_ratio = np.sin(theta[:,k+1])/np.sin(theta[:,k-1])    #Check snell's law

tan_ratio = np.tan(theta[:,k+1])/np.tan(theta[:,k-1])
print("\n tan(I)/tan(R) mean is {0}, variance is {1}\n ".format(tan_ratio[1:-1].mean(),tan_ratio[1:-1].var()))

#Calculate accumulated charge
E_top = Ey[:,-2]
Q_top = np.sum(E_top)*delta   #units: 100(cm in E)*10^(-2) = C/m

E_bottom = -Ey[:,1]
Q_bottom = 2*np.sum(E_bottom)*delta   #units: 100(cm in E)*10^(-2) = C/m

E_side = -(Ex[-2,:] - Ex[1,:])
Q_side_fluid = 2*np.sum(E_side[0:k-1])*delta   #units: 100(cm in E)*10^(-2) = C/m
Q_side_air = np.sum(E_side[k:-1])*delta     #units: 100(cm in E)*10^(-2) = C/m
Q_fluid = Q_bottom + Q_side_fluid

#plot phi
plt.subplots(figsize=(4,8))
plt.title("Potential Distribution-{}".format(ratio))
cont = plt.contourf(X,Y,phi)
plt.xlabel("Lx ---->")
plt.ylabel("Ly ---->")
plt.colorbar(cont)
plt.show()

#plot field
plt.subplots(figsize=(4,8))
plt.title("Electric Field Distribution-{}".format(ratio))
skip=(slice(None,None,4),slice(None,None,4))
plt.quiver(X[skip],Y[skip],Ex[::-1,:][skip],-Ey[::-1,:][skip],headwidth=10,scale=5)
plt.xlabel("Lx ---->")
plt.ylabel("Ly ---->")
plt.show()

#plot error
plt.title("Error vs Niter")
plt.plot(err,label="error magnitude")
plt.xlabel("iter ---->")
plt.ylabel("error ---->")
plt.legend()
plt.show()

plt.title("Log(Error) vs Niter")
plt.semilogy(err,label="log error")
plt.xlabel("iter ---->")
plt.ylabel("log(error) ---->")
plt.legend()
plt.show()

#Fit a log expression for error
log_y_mat = np.log(np.transpose(err))       #modelling the error in a exponential function
x_mat = np.c_[np.arange(0,Nover+1),np.ones(shape = [Nover+1,1])]
fit1 = sp.lstsq(x_mat,log_y_mat)[0]     #lstsq fit for whole err array
fit2 = sp.lstsq(x_mat[500:-1],log_y_mat[500:-1])[0]

plt.semilogy(np.arange(0,Nover,50),err[::50],'r',label='Actual error')
plt.semilogy(np.arange(0,Nover,50),np.exp(np.dot(x_mat,fit1))[::50],'bo',markersize=8,label='Fit 1')
plt.semilogy(np.arange(0,Nover,50),np.exp(np.dot(x_mat,fit2))[::50],'y^',markersize=8,label='Fit 2')
plt.xlabel("interations-->")
plt.ylabel("error magnitude-->")
plt.title('Semilog plot of error')
plt.legend()
plt.show()

#plot sin(I)/sin(R)
plt.title("Snells Law verification")
plt.plot(sine_ratio,label="sin(I)/sin(R)")
plt.xlabel("Lx ---->")
plt.legend()
plt.show()

#plot change in angle
plt.plot(theta[:,k-1]-theta[:,k+1])
plt.title("Change in angle")
plt.xlabel("Lx ---->")
plt.ylabel("angle(in radian) ---->")
plt.show()
