# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:06:39 2023
MRSfunction#
Python module that simulates magnetisation from the Pauly paper for arbitrary shapes also can optimise the B! pulse 
for these shapes
copy for scaling purposes
code to use
@author: Wstev
"""
import scipy
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
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
            
            integral += B1t*gamma*cmatrix
            Rintegral = float(np.real(integral))
            Iintegral = float(np.imag(integral))  
        
        
        Mreal[i] = (Rintegral)
        Mimaginary[i] = (Iintegral)
    return storei,storer,Mreal,Mimaginary

def conversion(KX,KY, B1, gamma,FOV, grid_size,T,Tn,dw):
        """Returns complex valued Magnetisation from calculate magnetisation function
            
    Parameters:
        k_trajectory (KX,KY): A 2D numpy array representing the k-space trajectory.
        B1: A function representing the RF pulse applied to the tissue.        
        gamma: The gyromagnetic ratio of the tissue being imaged.
        FOV: The field of view of the image (in cm).
        grid_size: The size of the grid (in pixels).
        T/Tn = dt : sampling rate of RF pulse
        Tn : number of points sampled in time
          
    Returns:
        M : a 1D numpy array representing the magnitude of the magnetization over the grid.
        storei : A 2d system matrix used for the optimisation that is complex
        Mcomplex : the magnetisation over the grid expressed as a complex number
    
    """
        storei,storer,Mreal,Mimaginary = calculate_magnetization(KX,KY, B1, gamma,FOV, grid_size,T,Tn,dw) 
        Mcomplex = 1j*(Mreal + 1j*Mimaginary)
        M = (Mcomplex*np.conjugate(Mcomplex))**0.5
        store = 1j*gamma*(1j*storei +storer)
        return M,store,Mcomplex

    
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
### Setting up target magnetisation
    grid_spacing = FOV / grid_size    
    r = R/grid_spacing 
    Mtarget = np.zeros((grid_size, grid_size)) 
    m = int(grid_size/2) 
    for i in range(grid_size):
        for j in range(grid_size):
            if (i-m)**2+(j-m)**2 < r**2:
                Mtarget[i,j] = alpha
    Mtarget1 = np.reshape(Mtarget,(grid_size**2))
    return Mtarget,Mtarget1

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

def krosette(R,n,t,T,phi): 
    """
    Draws a rossette in k-space

    Parameters
    ----------
    r : radius of rossette
    n : number of petals
   

    Returns : kx,ky trajectories
    -------
    None.

    """ 
    theta = 2*np.pi*t/T
    r = R* np.sin(n*theta)

    kx = r*np.cos(theta - phi)
    ky = r*np.sin(theta - phi)
    kx = np.flip(kx)
    ky = np.flip(ky)
    #plt.plot(kx,ky)

    return kx,ky

def kspiral(R,n,t,T,phi): 
    """
    Draws a rossette in k-space

    Parameters
    ----------
    R : radius of spiral
    n : number of rings inwards
   

    Returns : kx,ky trajectories
    -------
    None.

    """ 
    theta = 2*np.pi*t*n/T
    kx = R*(1 - t/T)*np.cos(theta - phi) #spiral trajectories
    ky = R*(1 - t/T)*np.sin(theta - phi)
    #plt.plot(kx,ky,"s")

    return kx,ky

def Pulsetest(T,Tn,dw,FOV,grid_size,target,kxp,kyp,B1):
    """
    Packs together many previous functions to return an optmised pulse
    
    Parameters:
        T : length of pulse
        Tn : number of points in pulse
        dw : frequencey offset from resonance
        FOV : field of view
        grid_size : size of one size of square grid
        target : target shape we want to excite as a 1 D array
        kx,ky : sampling trajectories
        B1 : applied B1 if analytical version used
        
    Returns:
        M1 : a 1D numpy array representing the magnetization over the grid.
        Mimage : M in a 2D form for plotting
        A : A 2d system matrix used for the optimisation
        RMS : Root mean square error
        qRMS : just looks at the route mean square that occurs in the target region
        
    """
    M = conversion(kxp,kyp,B1,gamma,FOV,grid_size,T,Tn,dw)
    
    A = M[1]
    lambd = 1
    M1 = M[2]
    Mimage = np.reshape(M[2],(grid_size,grid_size))
    RMS = np.abs(np.sum((((target[1]-M1)**2)/grid_size**2)**0.5))
    q = test[0]*Mimage   # just looks at magnetisation occuring within target
    qRMS = np.sum((((target[0]-q)**2)/grid_size**2)**0.5)
    return M1,Mimage,A,RMS,qRMS

def Pulsetestoptimised(T,Tn,dw,FOV,grid_size,target,kxp,kyp,B1,lambd):
    """
    Packs together many previous functions to return an optmised pulse
    
    Parameters:
        T : length of pulse
        Tn : number of points in pulse
        dw : frequencey offset from resonance
        FOV : field of view
        grid_size : size of one size of square grid
        target : target shape we want to excite as a 1 D array
        kx,ky : sampling trajectories
        B1 : applied B1 if analytical version used
        lamda : lambda value applied in least squares fit
        
    Returns:
        M : a 1D numpy array representing the magnetization over the grid.
        Mimage : M in a 2D form for plotting
        A : A 2d system matrix used for the optimisation
        RMS : Root mean square error
        qRMS : just looks at the route mean square that occurs in the target region
        B1 : optimised B1 pulse is returned
    """
    M = conversion(kxp,kyp,B1,gamma,FOV,grid_size,T,Tn,dw)
    
    A = M[1]
    M = np.reshape(M[2],(grid_size,grid_size))
    #B1 = leastsquaresregularised(A,target[1],B1,lambd)
    B1 = scipy.sparse.linalg.lsqr(A,target[1],l*np.sqrt(gamma))[0]
    M = conversion(kxp,kyp,B1,gamma,FOV,grid_size,T,Tn,dw)
    M1 = M[2]
    B1 = B1
    Mimage = np.reshape(M[2],(grid_size,grid_size))
    RMS = np.abs(np.sum((((target[1]-M1)**2)/grid_size**2)**0.5))
    q = test[0]*Mimage   # just looks at magnetisation occuring within target
    qRMS = np.sum((((target[0]-q)**2)/grid_size**2)**0.5)
    return M1,Mimage,A,RMS,q,qRMS,B1

def Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,alpha):
    """
    Packs together many previous functions to investigate the effects of an off resonant pulse
    
    Parameters:
        frequencey_range : maximum off resonant frequency sampled, code checkd from -frequencey_range to frequencey_range
        dn : number of frequencies sampled
        T : length of pulse
        Tn : number of points in pulse
        dw : frequencey offset from resonance
        FOV : field of view
        grid_size : size of one size of square grid
        target : target shape we want to excite as a 1 D array
        kx,ky : sampling trajectories
        B1 : applied B1 if analytical version used
        alpha : set to 1
        
    Returns:
        B1 : B1 pulse that is optimised across all frequencies
        Storearray : One large 1D array that contains magnetisation across all off resonant frequencies
        RMSarray : An array that contains the RMSE over the entire grid
        qarray : An array that contains the ratio of target magnetisation over the entire grid
    """
    
    ###Off resonance code
    d,systemmatrix1 = -frequencey_range,np.array([])
    n,n1,dn,l = 0,0,dn,int(5)

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    ### Calculate system matrix
    storearray = np.array([])
    while d < frequencey_range:
        if n == l:
            n = 0
            n1 = n1 + 1
        check = Pulsetest(T,Tn,d,FOV,grid_size,test,kxp,kyp,B1)
        systemmatrix1 = np.append(systemmatrix1,check[2]) 
        n = n + 1
        d = d + 2*frequencey_range/dn
    ### Define target for furthest off resonance frequencies on resonance
    targettotal = test[1]
    #testRMS = test
    x = systemmatrix1[0:(grid_size**2)*Tn]
    x1 = systemmatrix1[(int(dn/2))*(grid_size**2)*Tn:(int(dn/2)+1)*(grid_size**2)*Tn]
    y = systemmatrix1[((dn-1)*(grid_size**2)*Tn):len(systemmatrix1)]
    targettotal = np.append(targettotal,test[1])
    targettotal = np.append(targettotal,test[1])
    x = np.append(x,y)
    x = np.append(x,x1)
    A = np.reshape(x,(3*grid_size**2,Tn)) 
    b = targettotal
    B1f = scipy.sparse.linalg.lsqr(A,b,l*np.sqrt(gamma))[0]
    scale = gamma*np.sum(B1f)*dt
 
  
    #B1f = scale(A,b,B1,T,Tn,dw,FOV,grid_size,kxp,kyp,R,alpha)
    ### Plotting and recording RMS and Target excitation fraction
    d,systemmatrix1 = -frequencey_range,np.array([])
    n,n1,dn,l = 0,0,dn,int(5)
    f,array = int(dn/l),np.arange(0,dn)   
    storearray = np.array([])
    RMSarray = np.array([])
    qarray = np.array([])
    darray = np.array([])
    while d < frequencey_range: 
        check = Pulsetest(T,Tn,d,FOV,grid_size,testRMS,kxp,kyp,B1f)
        magnetisation = check[0]
        RMS = check[3]
        q = circleRMS(R,FOV,grid_size,np.abs(check[1]))
        storearray = np.append(storearray,q)
        RMSarray = np.append(RMSarray,RMS)
        qarray = np.append(qarray,q)
        darray = np.append(darray,d)
        n = n + 1
        d = d + 2*frequencey_range/dn
    return B1f,storearray,RMSarray,qarray,darray

def segmented_trajectory_rossette(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,A):
    """
    Parameters
    
    N : number of segments used
    
    Returns :
    magtotal : magnetisation of all off resonance slices with the off resonant pulse applied
    """
    test = np.array([test[0]/N,test[1]/N],dtype="object")
    nt = 1
    magtotal,qtotal,RMStotal,kytotal,kxtotal,B1total = 0,0,0,np.array([]),np.array([]),np.array([])
    while nt < N+1:
        kxp,kyp = krosette(1,n,t,T,(nt/N)*(2*np.pi/n))
        kxp = A*kxp
        kyp = A*kyp
        B1f,magnetisation,RMSarray,qarray,darray = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,1/N)
        magtotal = magtotal + magnetisation
        qtotal = qtotal + qarray
        RMStotal = RMStotal + RMSarray
        kxtotal = np.append(kxtotal,kxp)
        kytotal = np.append(kytotal,kyp)
        B1total = np.append(B1total,B1f)
        nt = nt + 1
    magtotal = np.reshape(magtotal,(dn,grid_size,grid_size))
    return magtotal,RMStotal,qtotal,darray,B1f,kxtotal,kytotal,B1total
def segmented_trajectory_spiral(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,A):
    """
    Parameters
    
    N : number of segments used
    
    Returns :
    magtotal : magnetisation of all off resonance slices with the off resonant pulse applied
    """
    test = np.array([test[0]/N,test[1]/N],dtype="object")
    nt = 1
    magtotal,qtotal,RMStotal,kytotal,kxtotal,B1total = 0,0,0,np.array([]),np.array([]),np.array([])
    while nt < N+1:
        kxp,kyp = kspiral(1,n,t,T,(nt/N)*(2*np.pi/n))
        kxp = A*kxp
        kyp = A*kyp
        B1f,magnetisation,RMSarray,qarray,darray = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,1/N)
        magtotal = magtotal + magnetisation
        qtotal = qtotal + qarray
        RMStotal = RMStotal + RMSarray
        kxtotal = np.append(kxtotal,kxp)
        kytotal = np.append(kytotal,kyp)
        B1total = np.append(B1total,B1f)
        nt = nt + 1
    magtotal = np.reshape(magtotal,(dn,grid_size,grid_size))
    return magtotal,RMStotal,qtotal,darray,B1f,kxtotal,kytotal,B1total

### Pulse parameters
T,Tn,dw = 0.05,100,0
dt = T/Tn 
t = np.linspace(0,T,Tn)
#t = np.abs(np.linspace(-T/2,T/2,Tn))
#check k space limits
z = 100
FOV,grid_size = 10/z,100
grid_spacing = FOV / grid_size
A = z 
n = 10
frequencey_range = 1000
phi = 0
kxp,kyp = krosette(1,n,t,T,phi)
kxp = A*kxp
kyp = A*kyp

#kxp = A*(1 - t/T)*np.cos(2*np.pi*n*t/T) #spiral trajectories
#kyp = A*(1 - t/T)*np.sin(2*np.pi*n*t/T)
kt = (kxp**2 + kyp**2)**0.5
###Defining Spatial localisation
B,alpha,A,gamma = 2,1,1,4.257*10**7  
Dk = alpha*np.exp(-(B**2)*(kxp**2 + kyp**2)/A**2) 
### Calculating gradient waveforms and B1 pulse
Gxp,Gyp = (np.gradient(kxp)/gamma),(np.gradient(kyp)/gamma)
B1 = Dk*gamma*np.sqrt(Gxp**2 + Gyp**2)
### Define target
R,alpha = 2/z,1
test = circletarget(R,FOV,grid_size,alpha)
testRMS = circletarget(R,FOV,grid_size,alpha)
Norm = np.sum(test[0])

### Off resonance code

dn = 40 # number of panels used
Tarray = np.array([0.001,0.003,0.006,1])
narray = np.array([10,20,40,1])
a = 0
#%%
plt.rcParams.update({'font.size': 18})
fig, axs = plt.subplots(2, figsize=(12, 6))
while a < len(Tarray)-1:
    T = Tarray[a]
    dt = T/Tn 
    t = np.linspace(0,T,Tn)
    A = z 
    n = 30
    kxp1,kyp1 = krosette(1,n,t,T,phi)
    kxp1 = A*kxp1
    kyp1 = A*kyp1
    kxp = A*(1 - t/T)*np.cos(2*np.pi*n*t/T) #spiral trajectories
    kyp = A*(1 - t/T)*np.sin(2*np.pi*n*t/T)
    Gxp,Gyp = (np.gradient(kxp)/gamma),(np.gradient(kyp)/gamma)
    kt = (kxp**2 + kyp**2)**0.5
    ### Analytic B1 pulse
    Dk = alpha*np.exp(-(B**2)*(kxp**2 + kyp**2)/A**2) 
    ### Calculating gradient waveforms and B1 pulse
    B1 = Dk*gamma*np.sqrt(Gxp**2 + Gyp**2)
    
    B1f,storearray,storeRMSarray,qarray,darray  = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,alpha)
    B1f,storearray1,storeRMSarray1,qarray,darray  = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp1,kyp1,B1,alpha)
    
    axs[0].plot(darray,storearray/Norm,"--",label =' T= {}(s)'.format(T),linewidth = 3)
    #axs[0].plot(darray,storearray1/Norm,label =' T= {}(s)'.format(T),linewidth = 3)
    axs[1].plot(darray,storeRMSarray,"--",label =' T= {}(s)'.format(T),linewidth = 3)
    #axs[1].plot(darray,storeRMSarray1,label =' T= {}(s)'.format(T),linewidth = 3)
    axs[0].set_title("Ratio of magnetisation response to target ")
    axs[1].set_title("RMS Error as a function of offset frequencey ")
    axs[1].set_xlabel("Offset Frequencey")
    axs[0].set_ylabel("Fraction of Target excited")
    axs[1].set_ylabel("RMSE")
    plt.legend()
    a = a + 1
#%%
while a < len(Tarray)-1:
    T = Tarray[a]
    dt = T/Tn 
    t = np.linspace(0,T,Tn)
    A = z 
    n = 30
    kxp1,kyp1 = krosette(1,n,t,T,phi)
    kxp1 = A*kxp1
    kyp1 = A*kyp1
    kxp = A*(1 - t/T)*np.cos(2*np.pi*n*t/T) #spiral trajectories
    kyp = A*(1 - t/T)*np.sin(2*np.pi*n*t/T)
    Gxp,Gyp = (np.gradient(kxp)/gamma),(np.gradient(kyp)/gamma)
    kt = (kxp**2 + kyp**2)**0.5
    ### Analytic B1 pulse
    Dk = alpha*np.exp(-(B**2)*(kxp**2 + kyp**2)/A**2) 
    ### Calculating gradient waveforms and B1 pulse
    B1 = Dk*gamma*np.sqrt(Gxp**2 + Gyp**2)
    print(T)
    B1f,storearray,storeRMSarray,qarray,darray  = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,alpha)
    B1f,storearray1,storeRMSarray1,qarray,darray  = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp1,kyp1,B1,alpha)
    plt.plot(darray,storeRMSarray,"--",label =' T= {}(s)'.format(T),linewidth = 3)
    plt.plot(darray,storeRMSarray1,label =' T= {}(s)'.format(T),linewidth = 3)
    #plt.plot(darray,storeRMSarray,label =' T= {}(s)'.format(T))
    plt.title("RMS Error as a function of offset frequencey ")
    plt.xlabel("Offset Frequencey")
    plt.ylabel("RMS Error")

    plt.legend()
    a = a + 1
#%%
a = 0
fig, axs = plt.subplots(1,2, figsize=(12, 6))
while a < len(narray)-1:
    
    dt = T/Tn 
    t = np.linspace(0,T,Tn)
    A = z 
    n = narray[a]
    print(n)
    kxp1,kyp1 = krosette(1,n,t,T,phi)
    kxp1 = A*kxp1
    kyp1 = A*kyp1
    kxp = A*(1 - t/T)*np.cos(2*np.pi*n*t/T) #spiral trajectories
    kyp = A*(1 - t/T)*np.sin(2*np.pi*n*t/T)
    Gxp,Gyp = (np.gradient(kxp)/gamma),(np.gradient(kyp)/gamma)
    kt = (kxp**2 + kyp**2)**0.5
    ### Analytic B1 pulse
    Dk = alpha*np.exp(-(B**2)*(kxp**2 + kyp**2)/A**2) 
    ### Calculating gradient waveforms and B1 pulse
    B1 = Dk*gamma*np.sqrt(Gxp**2 + Gyp**2)
    B1f,storearray,storeRMSarray,qarray,darray  = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,alpha)
    B1f,storearray1,storeRMSarray1,qarray,darray  = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp1,kyp1,B1,alpha)
    axs[0].plot(darray,storearray/Norm,"--",label =' n= {}(s)'.format(n),linewidth = 3)
    #axs[0].plot(darray,storearray1/Norm,label =' n= {}(s)'.format(n),linewidth = 3)
    axs[1].plot(darray,storeRMSarray,"--",label =' n= {}(s)'.format(n),linewidth = 3)
    #axs[1].plot(darray,storeRMSarray1,label =' n= {}(s)'.format(n),linewidth = 3)
    axs[0].set_title("Ratio of magnetisation response to target ")
    axs[1].set_title("RMS Error as a function of offset frequencey ")
    axs[1].set_xlabel("Offset Frequencey")
    axs[0].set_ylabel("Fraction of Target excited")
    axs[1].set_ylabel("RMSE")
    plt.legend()
    
    a = a + 1
#%%
a = 0
storeqmid = np.array([])
garray = np.array([10,30,50,70,100,150,200,250,300,350,400,450,500])
while a < len(garray)-1:

    dt = T/Tn 
    t = np.linspace(0,T,Tn)
    A = z 
    FOV,grid_size = 10/z,garray[a]
    grid_spacing = FOV / grid_size
    n = 5
    kxp1,kyp1 = krosette(1,n,t,T,phi)
    kxp1 = A*kxp1
    kyp1 = A*kyp1
    kxp = A*(1 - t/T)*np.cos(2*np.pi*n*t/T) #spiral trajectories
    kyp = A*(1 - t/T)*np.sin(2*np.pi*n*t/T)
    Gxp,Gyp = (np.gradient(kxp)/gamma),(np.gradient(kyp)/gamma)
    kt = (kxp**2 + kyp**2)**0.5
    ### Analytic B1 pulse
    Dk = alpha*np.exp(-(B**2)*(kxp**2 + kyp**2)/A**2) 
    ### Calculating gradient waveforms and B1 pulse
    B1 = Dk*gamma*np.sqrt(Gxp**2 + Gyp**2)
    test = circletarget(R,FOV,grid_size,alpha)
    Norm = np.sum(test[0])
    testRMS = circletarget(R,FOV,grid_size,alpha)
    B1f,storearray,storeRMSarray,qarray,darray  = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,alpha)
    B1f,storearray1,storeRMSarray1,qarray,darray  = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp1,kyp1,B1,alpha)
    qarraymid = qarray[int(dn/2)]/Norm
    storeqmid = np.append(storeqmid,qarraymid)
    #plt.plot(darray,storearray/Norm,"--",label =' grid size= {}'.format(grid_size),linewidth = 1)
    
    #plt.plot(darray,qarray,label =' grid size = {}'.format(grid_size),linewidth = 1)
    #plt.plot(darray,storeRMSarray,label =' T= {}(s)'.format(T))
    #plt.title("Target excitation")
    #plt.xlabel("Grid size")
    #plt.ylabel("Target excitation")

    #plt.legend()
    a = a + 1
#%%
plt.title("Target excitation")
plt.xlabel("Grid_size")
plt.ylabel("Target excitation")
plt.plot(garray[0:len(garray)-1],storeqmid,"x")
plt.plot(garray[0:len(garray)-1],storeqmid,linewidth = 2.5)
#%% Figures for the lab diary


#%%
plt.plot(darray[1:],(storearray[1:])/Norm)
plt.title("Ratio of magnetisation response to target ")
plt.xlabel("Offset Frequencey")
plt.ylabel("Fraction of Target excited")
#%%
plt.plot(darray[1:],storeRMSarray[1:])
plt.title("RMS Error as a function of offset frequencey ")
plt.xlabel("Offset Frequencey")
plt.ylabel("RMS Error")

#%%
plt.plot(t,B1f*10**6)
plt.title("Optimised B1 pulse ")
plt.xlabel("t (s)")
plt.ylabel("B1 ($\mu$T)")




    
#%%
plt.plot(t,M[6]*10**6)
plt.title("Optimised B1 pulse ")
plt.xlabel("t (s)")
plt.ylabel("B1 ($\mu$T)")
B1s = M[6]

#M = Pulsetest(T,Tn,dw,FOV,grid_size,test,kxp,kyp,B1s)
#plt.imshow(np.abs(M[1]))
#%%
plt.title("Interleaved sampling trajectory")  
plt.plot(kxp,kyp)
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
#%%
plt.title("k-space sampling trajectory")
plt.plot(t,kt)
plt.xlabel("t")
plt.ylabel("$k$")
#%%
plt.title("Gradient trajectories")
plt.plot(t,Gxp*10**6,label='$G_x$')
plt.plot(t,Gyp*10**6,label="$G_y$")
plt.xlabel("t")
plt.ylabel("$\mu$T/m")
plt.legend()

#test1 = circletarget(R + 0.25/z,FOV,grid_size,0.22)
#test2 = circletarget(R + 0.5/z,FOV,grid_size,0.22)
#test3 = circletarget(R + 0.75/z,FOV,grid_size,0.22)  
#test4 = circletarget(R + 0.90/z,FOV,grid_size,0.12)
#blurredtarget = test[0] + test1[0] + test2[0] + test3[0] + test4[0]
#blurredtarget1D = test[1] + test1[1] + test2[1] + test3[1] + test4[1]
#test = blurredtarget,blurredtarget1D
#issue with the solver from the looks of it can be fixed hopefully