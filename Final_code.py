# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:30:53 2023
Coding up the results from the Pauly paper again,in an attempt to reproduce their results exactly 

@author: Wstev
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numba import jit
import cvxpy as cp
from PIL import Image
@jit(nopython=True)
def calculate_magnetization(KX,KY, B1, gamma,FOV, grid_size,T,Tn,dw):
    """
    Calculate the magnetization using the given equation for a user-defined k-space trajectory.
    
    Parameters:
        k_trajectory (KX,KY): A 2D numpy array representing the k-space trajectory.
        B1: A function representing the RF pulse applied to the tissue.        
        gamma: The gyromagnetic ratio of the tissue being imaged.
        FOV: The field of view of the image (in cm).
        grid_size: The size of the grid (in pixels).
        T/Tn = dt : sampling rate of RF pulse
        Tn : number of points sampled in time
        
    Returns:
        M : a 1D numpy array representing the magnetization over the grid.
        store : A 2d system matrix used for the optimisation
        Mreal : a 1D numpy array representing the real magnetisation
        Mimaginary : a 1D numpy array representing the imaginary magnetisation
        dw : frequencey offset
    """
    #define meshgrid that is compatible with jit
    dt = T/Tn
    t = np.linspace(0,T,Tn)
    def meshgrid(x, y):
        xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
        yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
        for i in range(x.size):
            for j in range(y.size):            
                xx[i,j] = x[i]  
                yy[i,j] = y[j]  
                    
        return yy, xx
    grid_spacing = FOV / grid_size    
    x,y = np.linspace(-FOV/2, FOV/2, grid_size),np.linspace(-FOV/2, FOV/2, grid_size)
    X, Y = meshgrid(x, y)    
    # Initialize the magnetization
    B1sum = 1/((gamma*np.sum(B1))*dt)
    M0 = 1
    Mreal = np.ones((grid_size**2)) 
    Mimaginary = np.ones((grid_size**2)) 
    storei = np.ones(((grid_size**2),Tn))
    storer = np.ones(((grid_size**2),Tn))
    X1 = np.reshape(X,(grid_size**2))
    Y1 = np.reshape(Y,(grid_size**2))
    # Calculate the magnetization at each point on the grid
    for i in range(grid_size**2):
        # Compute the k-space coordinate for this grid point
        xi,yi = X1[i],Y1[i]
        r = np.array([xi, yi])
        # Compute the integral term
        integral = 0
        for nott in range(Tn):
            kx1,ky1 = KX[nott],KY[nott]
            kt = np.array([kx1,ky1])
            B1t = B1[nott]
            cmatrix = np.exp(1j*np.dot(r,kt))*np.exp(1j*dw*(t[nott]-T))*dt
            
            storer[i,nott-1] = np.real(cmatrix)
            storei[i,nott-1] = np.imag(cmatrix)
            #store[i,nott-1] = (np.real(cmatrix)**2 + np.imag(cmatrix)**2)**0.5 #Just collects modulus of coeffecients
            
            integral += M0*B1t*gamma*cmatrix
            Rintegral = float(np.real(integral))
            Iintegral = float(np.imag(integral))  
        
        
        Mreal[i] = (Rintegral)
        Mimaginary[i] = (Iintegral)
    return storei,storer,Mreal,Mimaginary

def conversion(KX,KY, B1, gamma,FOV, grid_size,T,Tn,dw):
        "Returns complex valued Magnetisation"
        storei,storer,Mreal,Mimaginary = calculate_magnetization(KX,KY, B1, gamma,FOV, grid_size,T,Tn,dw) 
        Mcomplex = 1j*(Mreal + 1j*Mimaginary)
        M = (Mcomplex*np.conjugate(Mcomplex))**0.5
        return M,storei,storer,Mcomplex
    
def leastsquares(A,b,B1):
    """
    Uses the least squares method from the module CVXPY to optimise the B1 pulse

    Parameters:
    A : system matrix as a 2D array of size Ngrid X Ntimepoints
    b : target magnetisation shaped as a 1D array
    B1 : previous B1 pulse,not really used
    
    Returns:
    B1 : optimised B1 pulse
    """
    # Define and solve the CVXPY problem.
    x = cp.Variable(len(B1)) #b1 pulse
    #constraints = [x <=1000*10**-6,x >=-1000*10**6] #adds constraints to the problem,first two are peak power and the last is overall power dissipated
    cost = cp.sum_squares(A @ x - b)
    #prob = cp.Problem(cp.Minimize(cost),constraints)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    B1 = x.value
    return B1
   
def leastsquaresregularised(A,b,B1,lambd):
    """
    Uses the least squares method from the module CVXPY to optimise the B1 pulse

    Parameters:
    A : system matrix
    b : target magnetisation shaped as a 1D array
    B1 : previous B1 pulse,not really used
    lambd : regularisation parameter
    Returns:
    B1 : optimised B1 pulse
    """
    def loss_fn(X, Y, beta):
        return cp.pnorm(X @ beta - Y, p=2)**2

    def regularizer(beta):
        return cp.pnorm(beta, p=2)**2
 
    def objective_fn(X, Y, beta, lambd):
        return loss_fn(X, Y, beta) + lambd * regularizer(beta)
    #beta = cp.Variable(len(B1),complex=True)
    beta = cp.Variable(len(B1))
    #constraints = [gamma*cp.sum(beta**2)*dt <=1,gamma*cp.sum(beta**2)*dt >=0.99] #These values are now in microtesla
    constraints = [cp.sum(beta**2)*dt ==1]
    problem = cp.Problem(cp.Minimize(objective_fn(A, b, beta, lambd)))
    #problem = cp.Problem(cp.Minimize(objective_fn(A, b, beta, lambd)),constraints)
    problem.solve()
    B1 = beta.value
    return B1

def circletarget(R,FOV,grid_size,alpha):
    """ Creates a circular target magnetisation pattern
    Parameters:
        R : The radius of the circular target
        FOV : The field of view of the image (in cm).
        grid_size : The size of the grid (in pixels).
        alpha : the amount of magnetisation in the target,will almost always be set as 1
    Returns:
        Mtarget : The target magnetisation as a 2D array
        Mtarget1 : The target magnetisation as a 1D array
        
    """
    count = 0
### Setting up target magnetisation
    grid_spacing = FOV / grid_size    
    r = R/grid_spacing 
    Mtarget = np.zeros((grid_size, grid_size))  
    m = int(grid_size/2) 
    for i in range(grid_size):
        for j in range(grid_size):
            if (i-m)**2+(j-m)**2 < r**2:
                Mtarget[i,j] = alpha
                count = count + 1
    Mtarget1 = np.reshape(Mtarget,(grid_size**2))
    Area = count
    print(Area)
    return Mtarget,Mtarget1    

def circletarget1(R,FOV,grid_size,alpha):
    """ Creates a circular target magnetisation pattern
    Parameters:
        R : The radius of the circular target
        FOV : The field of view of the image (in cm).
        grid_size : The size of the grid (in pixels).
        alpha : the amount of magnetisation in the target,will almost always be set as 1
    Returns:
        Mtarget : The target magnetisation as a 2D array
        Mtarget1 : The target magnetisation as a 1D array
        
    """
### Setting up target magnetisation
    grid_spacing = FOV / grid_size    
    r = R/grid_spacing 
    Mtarget = np.zeros((grid_size, grid_size))  
    m = int(grid_size/2) 
    for i in range(m):
        for j in range(grid_size):
            if (i-m)**2+(j-m)**2 < r**2:
                Mtarget[i,j] = alpha
    Mtarget1 = np.reshape(Mtarget,(grid_size**2))
    return Mtarget,Mtarget1   

def circletarget2(R,FOV,grid_size,alpha):
    """ Creates a circular target magnetisation pattern
    Parameters:
        R : The radius of the circular target
        FOV : The field of view of the image (in cm).
        grid_size : The size of the grid (in pixels).
        alpha : the amount of magnetisation in the target,will almost always be set as 1
    Returns:
        Mtarget : The target magnetisation as a 2D array
        Mtarget1 : The target magnetisation as a 1D array
        
    """
    count = 0
### Setting up target magnetisation
    grid_spacing = FOV / grid_size    
    r = R/grid_spacing 
    Mtarget = np.zeros((grid_size, grid_size))  
    m = int(grid_size/2) 
    for i in range(grid_size):
        for j in range(grid_size):
            if (i-m)**2+(j-m)**2 < r**2 and (i-m)**2+(j-m)**2 > ((r-1)**2):
                Mtarget[i,j] = alpha
                count = count + 1
    Mtarget1 = np.reshape(Mtarget,(grid_size**2))
    Area = count
    print(Area)
    return Mtarget,Mtarget1 

def krosette(R,n,t,T): 
    """
    Draws a rossette in k-space

    Parameters
    ----------
    r : radius of rossette
    n : number of petals
   

    Returns : figure of rossette
    -------
    None.

    """ 
    theta = 2*np.pi*t/T
    r = R* np.sin(n*theta)

    kx = r*np.cos(theta)
    ky = r*np.sin(theta)
    #plt.plot(kx,ky)

    return kx,ky
def circleRMS(R,FOV,grid_size,M):
    """ This function generates the statistic that describes how much of the target excitation has been excited  
    Parameters:
        R : The radius of the circular target
        FOV : The field of view of the image (in cm).
        grid_size : The size of the grid (in pixels).
        M : 2D magnetisation response generated from conversion function
    Returns:
        Magnetisation : A singel value that  is the sum of the magnetisation excited in the target area
        
    """
### Setting up target magnetisation
    
    grid_spacing = FOV / grid_size    
    r = R/grid_spacing # radius of the circle used to produce target measured in cm
    Mtarget = M#target magnetisation array
    m = int(grid_size/2) #midpoint of the grid
    Magnetisation = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if (i-m)**2+(j-m)**2 < r**2:
                Magnetisation += Mtarget[i,j]
    
    return Magnetisation
#%%### Load in target image

import imageio as iio
img = iio.imread(r"C:\Users\Wstev\OneDrive - The University of Nottingham\4th Year\Physics Project - Arbitrary Shaped regions at 7T\Codes\Picture6.png")
img = img[:,:,1]/255
img = img[0:400,0:400]
img1 = np.reshape(img,(400**2))
test = np.array([img,img1])
plt.imshow(test[0])
#%%
#%%
T,Tn,dw = 0.001,1000,0
t = np.linspace(0,T,Tn)
FOV,grid_size,dn = 10/100,400,20
grid_spacing = FOV / grid_size
A = 100 #keep at 100
n = 30
kx = A*(1 - t/T)*np.cos(2*np.pi*n*t/T) 
ky = A*(1 - t/T)*np.sin(2*np.pi*n*t/T)
kt = np.sqrt(kx**2 + ky**2)
R = A
#kx,ky = krosette(R,n,t,T)
gamma = (4.257*10**7)
Gx,Gy = -A/(gamma*T)*(2*np.pi*n*(1-t/T)*(np.sin(2*np.pi*n*t/T)) + np.cos(2*np.pi*n*t/T)),A/(gamma*T)*(2*np.pi*n*(1-t/T)*(np.cos(2*np.pi*n*t/T)) - np.sin(2*np.pi*n*t/T))

alpha = 1# scales the tip angle
beta  = 2
dt = T/Tn
B1 = gamma*(A/T)*np.exp(-(beta**2)*(1-t/T)**2)*np.sqrt(((2*np.pi*n*(1-t/T))**2)+1)
alpha = 1/(gamma*np.sum(B1)*dt)   
B1 = alpha*gamma*(A/T)*np.exp(-(beta**2)*(1-t/T)**2)*np.sqrt(((2*np.pi*n*(1-t/T))**2)+1)

Mo,storei,storer,Mocomplex = conversion(kx,ky, B1, gamma,FOV, grid_size,T,Tn,dw)    
Mocomplex = np.reshape(Mocomplex,(grid_size,grid_size))  
R,alpha = 2/100,1
#test = circletarget(R,FOV,grid_size,alpha)
testr = circletarget2(R,FOV,grid_size,alpha)
#%%
A = 1j*gamma*(1j*storei +storer) #complex system matrix
b = test[1]
lambd = 1
#B1f = leastsquaresregularised(A,b,B1,lambd)
#B1f = leastsquares(A,b,B1)
#B1f = scipy.linalg.lstsq(A,b)[0]
n = 0
l = 1
while n <1:
    B1f = scipy.sparse.linalg.lsqr(A,b,l*np.sqrt(gamma))[0]
    if np.abs((gamma*np.sum(B1f))*dt) < 1.5:
        n = n + 1 
    print(np.abs((gamma*np.sum(B1f))*dt))
    l = l + 0.1
#B1f = scipy.optimize.lsq_linear(A,b,lsq_solver="lsmr")

M,storei,storer,Mcomplex = conversion(kx,ky, B1f, gamma,FOV, grid_size,T,Tn,dw)  
M = np.reshape(Mcomplex,(grid_size,grid_size))  
#%%Reproducing results from original paper
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(2,3, figsize=(12, 6))

axs[0,0].set_title("A")
axs[0,0].plot(kx,ky)
axs[0,0].set_xlabel("kx")
axs[0,0].set_ylabel("$ky$")

axs[0,1].set_title("B")
axs[0,1].plot(t*1000,Gx,label='$G_x$')
axs[0,1].plot(t*1000,Gy,label='$G_y$')
axs[0,1].set_xlabel("t (ms)")
axs[0,1].set_ylabel("$\mu$T/m")
axs[0,1].set_yticklabels([])
axs[0,1].legend(loc="best")
axs[0,2].plot(t*1000,B1*10**6)
axs[0,2].set_title("C")
axs[0,2].set_xlabel("t (ms)")
axs[0,2].set_ylabel("B1 ($\mu$T)")

a = axs[1,0].imshow(np.imag(Mocomplex))
axs[1,0].set_title("D")
axs[1,0].set_yticklabels([])
axs[1,0].set_xticklabels([])

cbar = fig.colorbar(a, ax=axs[1,0])

a1 = axs[1,1].imshow(np.real(Mocomplex))
axs[1,1].set_title("E")
axs[1,1].set_yticklabels([])
axs[1,1].set_xticklabels([])
cbar = fig.colorbar(a1, ax=axs[1,1])

axs[1,2].set_title("F")
axs[1,2].set_yticklabels([])
axs[1,2].set_xticklabels([])
a2 = axs[1,2].imshow(np.abs(Mocomplex))
cbar = fig.colorbar(a2, ax=axs[1,2])

#%% Looking at optimised results
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(2,3, figsize=(12, 6))
Norm = np.sum(test[0])
RMSE = np.abs(np.sum((((test[0]-M)**2)/grid_size**2)**0.5))
q = circleRMS(R,FOV,grid_size,np.abs(M))/Norm
axs[0,0].set_title("A")
axs[0,0].plot(kx,ky)
axs[0,0].set_xlabel("kx")
axs[0,0].set_ylabel("$ky$")

axs[0,1].set_title("B")

axs[1,0].set_xticklabels([])
axs[0,1].set_yticklabels([])
axs[0,1].set_xticklabels([])
axs[1,1].set_yticklabels([])

axs[1,2].set_yticklabels([])
axs[1,2].set_xticklabels([])


a6 = axs[0,1].imshow(test[0])
cbar = fig.colorbar(a6, ax=axs[0,1])
axs[0,2].plot(t*1000,np.real(B1f)*10**6,label="Real")
axs[0,2].plot(t*1000,np.imag(B1f)*10**6,label="Imag")
axs[0,2].set_title("C")
axs[0,2].set_xlabel("t (ms)")
axs[0,2].set_ylabel("B1 ($\mu$T)")
axs[0,2].legend(loc='best')
a = axs[1,0].imshow(np.imag(M))

axs[1,0].set_title("D")
#axs[1,0].set_ylabel("(mm)")
axs[1,0].set_yticklabels([])
axs[1,0].set_xticklabels([])
cbar = fig.colorbar(a, ax=axs[1,0])
#a4 = axs[1,0].imshow(np.abs(testr[0])*np.max(np.imag(M))+np.imag(M))
a1 = axs[1,1].imshow(np.real(M))
#a7 = axs[1,1].imshow(np.abs(testr[0])*np.max(np.real(M))+np.real(M))
#a5 = axs[1,1].imshow(np.abs(testr[0])+np.real(M))
axs[1,1].set_title("E")
cbar = fig.colorbar(a1, ax=axs[1,1])
#axs[1,1].set_xlabel("(mm)")
axs[1,1].set_yticklabels([])
axs[1,1].set_xticklabels([])
axs[1,2].set_title("F")
a2 = axs[1,2].imshow(np.abs(M))
axs[1,2].set_yticklabels([])
axs[1,2].set_xticklabels([])
#axs[1,2].set_xlabel("{}% Target excitation".format(np.around(q,3)*100))
#a8 = axs[1,2].imshow(np.abs(testr[0])*np.max(np.abs(M))+np.abs(M))
#axs[1,2].set_ylabel("RMSE = {}".format(np.around(RMSE,3)))
#a3 = axs[1,2].imshow(np.abs(testr[0])*np.abs(M)+np.abs(M))
cbar = fig.colorbar(a2, ax=axs[1,2])


#%% profile plots
length = np.linspace(0,FOV/10,grid_size)*1000
central_profile = M[int(grid_size/2)]
plt.plot(length,np.abs(central_profile),linewidth = 3.5)
plt.title("Magnetisation profile")
plt.xlabel("FOV (cm)")
plt.ylabel("Magnetisation")
#%% uoptimised vs optimised pulse
fig, axs = plt.subplots(1,2, figsize=(12, 6))
axs[0].imshow(np.abs(Mocomplex))
axs[1].imshow(np.abs(M))

