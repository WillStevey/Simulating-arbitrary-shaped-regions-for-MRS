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
    M0 = B1sum
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
    #x = systemmatrix1[int((dn/4)-1)*(grid_size**2)*Tn:(int(dn/4))*(grid_size**2)*Tn]
    #x1 = systemmatrix1[(int(dn/2))*(grid_size**2)*Tn:(int(dn/2)+1)*(grid_size**2)*Tn]
    #y = systemmatrix1[int((3*dn/4))*(grid_size**2)*Tn:(int(3*dn/4)+1)*(grid_size**2)*Tn]
    x = systemmatrix1[0:(grid_size**2)*Tn]
    x1 = systemmatrix1[(int(dn/2))*(grid_size**2)*Tn:(int(dn/2)+1)*(grid_size**2)*Tn]
    y = systemmatrix1[((dn-1)*(grid_size**2)*Tn):len(systemmatrix1)]
    targettotal = np.append(targettotal,test[1])
    targettotal = np.append(targettotal,test[1])
    x = np.append(x,x1)
    x = np.append(x,y)
    A = np.reshape(x,(3*grid_size**2,Tn)) 
    b = targettotal#
    l1 = 1
    B1f = scipy.sparse.linalg.lsqr(A,b,l1*np.sqrt(gamma))[0]
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
        storearray = np.append(storearray,magnetisation)
        RMSarray = np.append(RMSarray,RMS)
        qarray = np.append(qarray,q)
        darray = np.append(darray,d)
        n = n + 1
        d = d + 2*frequencey_range/dn
    return B1f,storearray,RMSarray,qarray,darray

def Offresonanceunoptimised(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,alpha):
    """
    Packs together many previous functions to investigate the effects of an off resonant pulse,only optimises using central frequencey
    
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
    x1 = systemmatrix1[(int(dn/2))*(grid_size**2)*Tn:(int(dn/2)+1)*(grid_size**2)*Tn]
    A = np.reshape(x1,(grid_size**2,Tn)) 
    b = targettotal
    l1 = 1
    B1f = scipy.sparse.linalg.lsqr(A,b,l1*np.sqrt(gamma))[0]
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
        storearray = np.append(storearray,magnetisation)
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

def segmented_trajectory_spiral_plotting(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,A):
    """
    Parameters
    
    N : number of segments used
    
    Returns :
    magtotal : magnetisation of all off resonance slices with the off resonant pulse applied
    """
    plt.rcParams.update({'font.size': 14})
    test = np.array([test[0]/N,test[1]/N],dtype="object")
    nt = 1
    magtotal,qtotal,RMStotal,kytotal,kxtotal,B1total = 0,0,0,np.array([]),np.array([]),np.array([])
    fig, axs = plt.subplots(N,3, figsize=(9,9),sharex='col')
    plt.subplots_adjust(top = 0.95,bottom = 0.06 ,left = 0.075 ,right = 0.930,hspace = 0.305,wspace = 0.025 )
    #fig.suptitle('Segmented Trajectories')
    while nt < N+1:
        kxp,kyp = kspiral(1,n,t,T,(nt/N)*(2*np.pi/n))
        kxp = A*kxp
        kyp = A*kyp
        B1f,magnetisation,RMSarray,qarray,darray = Offresonance1(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,1/N)
        magtotal = magtotal + magnetisation
        magtotal1 = np.reshape(magtotal,(dn,grid_size,grid_size))
        
        axs[nt-1,0].plot(kxp,kyp)
        axs[nt-1,0].plot(kxp,kyp,"s")
        axs[nt-1,0].set_xlabel("$k_x$ ($m^{-1}$)")
        axs[nt-1,0].set_ylabel("$k_y$ ($m^{-1}$)")
     
        a = axs[nt-1,2].imshow(np.abs(magtotal1[int(dn/2),:,:]))
        axs[nt-1,2].set_title("N= {}".format(nt))
        axs[nt-1,2].set_yticklabels([])
        axs[nt-1,2].set_xticklabels([])
        cbar = fig.colorbar(a, ax=axs[nt-1,2],orientation="vertical")
        axs[nt-1,1].plot(t*1000,B1f*10**6,'--')
        axs[nt-1,1].set_xlabel("t (ms)")
        axs[nt-1,1].set_ylabel("B1 ($\mu$T)")

        nt = nt + 1
    
    magtotal = np.reshape(magtotal,(dn,grid_size,grid_size))
    return 
def segmented_trajectory_spiral_unoptimised(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,A):
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
        B1f,magnetisation,RMSarray,qarray,darray = Offresonanceunoptimised(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,1/N)
        magtotal = magtotal + magnetisation
        qtotal = qtotal + qarray
        RMStotal = RMStotal + RMSarray
        kxtotal = np.append(kxtotal,kxp)
        kytotal = np.append(kytotal,kyp)
        B1total = np.append(B1total,B1f)
        nt = nt + 1
    magtotal = np.reshape(magtotal,(dn,grid_size,grid_size))
    return magtotal,RMStotal,qtotal,darray,B1f,kxtotal,kytotal,B1total

def segmented_trajectory_rossette_unoptimised(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,A):
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
        B1f,magnetisation,RMSarray,qarray,darray = Offresonanceunoptimised(frequencey_range,dn,T,Tn,FOV,grid_size,test,kxp,kyp,B1,1/N)
        magtotal = magtotal + magnetisation
        qtotal = qtotal + qarray
        RMStotal = RMStotal + RMSarray
        kxtotal = np.append(kxtotal,kxp)
        kytotal = np.append(kytotal,kyp)
        B1total = np.append(B1total,B1f)
        nt = nt + 1
    magtotal = np.reshape(magtotal,(dn,grid_size,grid_size))
    return magtotal,RMStotal,qtotal,darray,B1f,kxtotal,kytotal,B1total

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
#%%import imageio as iio
import imageio as iio
img = iio.imread("Picture3.png")
img = img[:,:,1]/255
#img = img[100:550,100:550]
img1 = np.reshape(img,(1200**2))
test = np.array([img,img1])
#%%
### Pulse parameters
T,Tn,dw = 0.001,100,0
dt = T/Tn 
t = np.linspace(0,T,Tn)
#t = np.abs(np.linspace(-T/2,T/2,Tn))
#check k space limits
z = 100
FOV,grid_size,dn = 10/z,100,40
grid_spacing = FOV / grid_size
A = z 
n = 20
frequencey_range = 1000
#At 7Tesla 1ppm is around 300 Hz,most metabolites are around 1-5 Ppm,so frequencey range of 1000
kxp = A*(1 - t/T)*np.cos(2*np.pi*n*t/T) #spiral trajectories
kyp = A*(1 - t/T)*np.sin(2*np.pi*n*t/T)
kt = (kxp**2 + kyp**2)**0.5
###Defining Spatial localisation
B,alpha,A,gamma = 2,1,1,4.257*10**7  
Dk = alpha*np.exp(-(B**2)*(kxp**2 + kyp**2)/A**2) 

### Calculating gradient waveforms and B1 pulse
Gxp,Gyp = (np.gradient(kxp)/gamma),(np.gradient(kyp)/gamma)
B1 = Dk*gamma*np.sqrt(Gxp**2 + Gyp**2)
### Define target
R,alpha = 2/z,1
testRMS = circletarget(R,FOV,grid_size,alpha) 
test = circletarget(R,FOV,grid_size,alpha)
testr = circletarget2(R,FOV,grid_size,alpha)
Norm = np.sum(test[0]) #sum of magnetisation in target
### Producing segmented pulses
N = 1 #number of segments used
#segmented_trajectory_spiral_plotting(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,z)
testr = circletarget2(R,FOV,grid_size,alpha)
#%%
magtotal,RMSarray,qarray,darray,B1f,kx,ky,B1total = segmented_trajectory_spiral(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,z)
magtotal1,RMSarray1,qarray1,darray,B1f1,kx1,ky1,B1total1 = segmented_trajectory_rossette(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,z)
magtotal2,RMSarray2,qarray2,darray,B1f2,kx2,ky2,B1total2  = segmented_trajectory_spiral_unoptimised(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,z)  
magtotal3,RMSarray3,qarray3,darray,B1f3,kx3,ky3,B1total3  = segmented_trajectory_rossette_unoptimised(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,z)
#%%write code to plot all off resonant behaviours
#magtotal = np.reshape(magnetisation1sum,(40,100,100))
n,n1,l,counter = 0,0,int(5),0
d = darray
f,array = int(dn/l),np.arange(0,dn)
fig, axs = plt.subplots(f,l, figsize=(12, 4))
plt.rcParams.update({'font.size': 10})
plt.subplots_adjust(top = 0.95,bottom = 0.06 ,left = 0.075 ,right = 0.930,hspace = 0.305,wspace = 0.025 )
while counter < len(array): 
    if n == l:
        n = 0
        n1 = n1 + 1
    a = axs[n1,n].imshow(np.abs(magtotal2[array[counter],:,:]))
    axs[n1,n].set_title(' $f$= {}Hz'.format(d[counter]))   
    #axs[n1,n].set_xlabel("{}% Excitation".format(np.around(qarray1[counter]/Norm,2)*100))
    #axs[n1,n].set_ylabel("RMSE = {}".format(np.around(RMSarray[counter]/Norm,5)))
    
    #cbar = fig.colorbar(a, ax=axs[n1,n])
    axs[n1,n].set_yticklabels([])
    axs[n1,n].set_xticklabels([])
    n = n + 1
    counter = counter + 1
cbar = fig.colorbar(a, ax=axs[:,:])


#%% Stats
lowerrange = np.ones(len(t))*frequencey_range/4
upperrange = np.ones(len(t))*3*frequencey_range/4
fig, axs = plt.subplots(1,2, figsize=(8, 4))
plt.subplots_adjust(top = 0.948,bottom = 0.088 ,left = 0.066 ,right = 0.980,hspace = 0.330,wspace = 0.196 )
plt.rcParams.update({'font.size': 16})
a = axs[0].plot(darray[1:len(darray)],(qarray[1:len(darray)])/Norm*100,'b-',label="Spiral (full width",linewidth = 3.5)
axs[0].plot(darray[1:len(darray)],(qarray2[1:len(darray)])/Norm*100,'b--',label="Spiral (centre)",linewidth = 3.5)
axs[0].plot(darray[1:len(darray)],(qarray1[1:len(darray)]/Norm)*100,'r-',label="Rosette (full width",linewidth = 3.5)
axs[0].plot(darray[1:len(darray)],(qarray3[1:len(darray)]/Norm)*100,'r--',label="Rosette (centre)",linewidth = 3.5)
axs[0].legend(loc='center')
axs[0].set_title("Target excitation ")
axs[0].set_xlabel("Offset Frequencey $Hz$")
axs[0].set_ylabel("%")



#b = axs[0].inset_axes([0.3, 0.3, 0.47, 0.47])
axs[1].plot(darray[1:len(darray)],RMSarray[1:len(darray)]/100,'b-',label="Spiral (full width)",linewidth = 3.5)
axs[1].plot(darray[1:len(darray)],(RMSarray2[1:len(darray)])/Norm*100,'b--',label="Spiral (centre)",linewidth = 3.5)
axs[1].plot(darray[1:len(darray)],RMSarray1[1:len(darray)]/100,'r-',label="Rosette (full width)",linewidth = 3.5)
axs[1].plot(darray[1:len(darray)],(RMSarray3[1:len(darray)])/Norm*100,'r--',label="Rosette (centre)",linewidth = 3.5)
axs[1].set_title("RMSE as a function of offset frequencey ")
axs[1].set_xlabel("Offset Frequencey $Hz$")
axs[1].set_ylabel("RMSE")
axs[1].legend(loc='best')
axin1 = axs[1].inset_axes([0.2, 0.2, 0.6, 0.30])
axin1.plot(darray[1:len(darray)],RMSarray[1:len(darray)]/100,'b-',label="Spiral (full width)",linewidth = 3.5)
axin1.plot(darray[1:len(darray)],RMSarray1[1:len(darray)]/100,'r-',label="Rosette (full width)",linewidth = 3.5)
#axin1.set_xlabel("Offset Frequencey $Hz$")
#axin1.set_ylabel("RMSE")
#axs[2].plot(t[0:250]*1000,B1f*10**6,'--',label="Spiral")
#axs[2].plot(t*1000,B1fsum*10**6,'-',label="Rossette")
#axs[2].set_title("Optimised B1 pulses ")
#axs[2].set_xlabel("t (ms)")
#axs[2].set_ylabel("B1 ($\mu$T)")
#axs[2].legend(loc='upper right')
#%% 1 slice as opposed to three slices optimisation method
fig, axs = plt.subplots(1,2, figsize=(12, 6))
fig.suptitle('Comparing optimisation methods')
plt.subplots_adjust(top=0.899,bottom=0.088,left=0.058,right=0.986,hspace=0.32,wspace=0.138)
plt.rcParams.update({'font.size': 16})
a = axs[0].plot(darray,(qarray)/Norm,'--',label="Full range (spiral)")
#axs[0].plot(darray,(qarray1)/Norm,'--',label="Full range (rossette)")
axs[0].set_title("Ratio of magnetisation response to target ")
axs[0].set_xlabel("Offset Frequencey $Hz$")
axs[0].set_ylabel("Fraction of Target excited")
axs[0].plot(darray,qarray2/Norm,'--',label="Centre (spiral)")
#axs[0].plot(darray,qarray3/Norm,'--',label="Centre (rossette)")
axs[0].legend(loc='upper right')

a = axs[1].plot(darray,(RMSarray),'--',label="Full range (spiral)")
#axs[1].plot(darray,RMSarray1,'--',label="Full range (rossette)")
axs[1].plot(darray,RMSarray2,'--',label="Centre (spiral)")
#axs[1].plot(darray,RMSarray3,'--',label="Centre (rossette)")
axs[1].set_title("RMS Error as a function of offset frequencey ")
axs[1].set_xlabel("Offset Frequencey $Hz$")
axs[1].set_ylabel("RMS Error")
axs[1].legend(loc='upper right')
#%%
plt.plot(t,M[6]*10**6)
plt.title("Optimised B1 pulse ")
plt.xlabel("t (s)")
plt.ylabel("B1 ($\mu$T)")
B1s = M[6]

#M = Pulsetest(T,Tn,dw,FOV,grid_size,test,kxp,kyp,B1s)
#plt.imshow(np.abs(M[1]))
#%%
n = 1
Gx,Gy = (np.gradient(kx)/gamma),(np.gradient(ky)/gamma) #these are the gradient waveforms for the respective k-space trajectories
Gx1,Gy1 = (np.gradient(kx1)/gamma),(np.gradient(ky1)/gamma) #these are the gradient waveforms for the respective k-space trajectories
tsum = np.linspace(0,N*T,N*Tn)
fig, axs = plt.subplots(2,2, figsize=(12, 6),sharey='row')
plt.subplots_adjust(top=0.958,bottom=0.071,left=0.061,right=0.978,hspace=0.287,wspace=0.025)
axs[0,0].set_title("Spiral sampling trajectory")  
axs[0,0].set_xlabel("$k_x$ ($m^{-1}$)")
axs[0,0].set_ylabel("$k_y$ ($m^{-1}$)")
while n < N+1:
    axs[0,0].plot(kx[(n-1)*Tn:n*Tn],ky[(n-1)*Tn:n*Tn])
    if n ==1:
        axs[0,0].plot(kx[(n-1)*Tn:n*Tn],ky[(n-1)*Tn:n*Tn],"s")
    n = n + 1
axs[0,1].set_title("Rossette sampling trajectory")  
axs[0,1].set_xlabel("$k_x$ ($m^{-1}$)")
#axs[0,1].set_ylabel("$k_x$")
n = 0
while n < N+1:
    axs[0,1].plot(kx1[(n-1)*Tn:n*Tn],ky1[(n-1)*Tn:n*Tn])
    if n ==1:
        axs[0,1].plot(kx1[(n-1)*Tn:n*Tn],ky1[(n-1)*Tn:n*Tn],"s")
    n = n + 1
axs[1,0].plot(tsum*1000,Gx*10**6,label='$G_x$')
axs[1,0].plot(tsum*1000,Gy*10**6,label='$G_y$')             
axs[1,0].set_title("Spiral trajectory gradient waveforms") 
axs[1,0].set_ylabel("$\mu$T/m") 
axs[1,0].set_xlabel("t (ms)")
axs[1,0].legend(loc='upper right')
axs[1,1].plot(tsum*1000,Gx1*10**6,label='$G_x$')
axs[1,1].plot(tsum*1000,Gy1*10**6,label='$G_y$')
axs[1,1].set_xlabel("t (ms)")
axs[1,1].set_title("Rosette trajectory gradient waveforms")  
axs[1,1].legend(loc='upper right')
#%%
plt.title("k-0space sampling trajectory")
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

###Strategy for segmenting trajectoires,produce big trajectory then index the arrays for the spirals then set the target to 1/number of spirals used then
###caluclate the magnetisation each arm produces,store the whole big array and then add them together at the end for plotting and analysis
#offresonance1spiral + offresonance2spiral then calc RMS and fraction excited then plot it
#This strategy has been done next step is to create a segmenting function that automates this process, eg creates k-space trajectory
#gets magnetisation then sums it all up to plot,now rossette is done a similar approach can be done with the spiral
#also want a panel plot of the pulses used along with the trajectory in k-space

##We can now segment the rossette trajectories,yaay!



#%%

#%%
T,Tn,dw = 0.015,N*Tn,0
dt = T/Tn 
t = np.linspace(0,T,Tn)
B1 = np.ones(len(t))
B1fsum,magnetisationsum,RMSarraysum,qarraysum,darraysum = Offresonanceunoptimised(frequencey_range,dn,T,Tn,FOV,grid_size,test,kx2,ky2,B1,1)
B1f1sum,magnetisation1sum,RMSarray1sum,qarray1sum,darraysum = Offresonanceunoptimised(frequencey_range,dn,T,Tn,FOV,grid_size,test,kx3,ky3,B1,1)
#%%
fig, axs = plt.subplots(1,3, figsize=(12, 6))
plt.subplots_adjust(top = 0.948,bottom = 0.088 ,left = 0.066 ,right = 0.980,hspace = 0.330,wspace = 0.196 )
plt.rcParams.update({'font.size': 16})
a = axs[0].plot(darray[1:],(qarray[1:])/Norm,'--',label="Spiral")
axs[0].plot(darray[1:],(qarray1[1:])/Norm,'-',label="Rossette")

axs[0].set_title("Ratio of magnetisation response to target ")
axs[0].set_xlabel("Offset Frequencey $Hz$")
axs[0].set_ylabel("Fraction of Target excited")
#b = axs[0].inset_axes([0.3, 0.3, 0.47, 0.47])
axs[1].plot(darray[1:],RMSarray[1:],'--',label="Spiral")
axs[1].plot(darray[1:],RMSarray1[1:],'-',label="Rossette")
axs[1].set_title("RMS Error as a function of offset frequencey ")
axs[1].set_xlabel("Offset Frequencey $Hz$")
axs[1].set_ylabel("RMS Error")
axs[1].legend(loc='upper right')

axs[2].plot(t*1000,B1f*10**6,'--')
axs[2].plot(t*1000,B1f1*10**6,'-')
axs[2].set_title("Optimised B1 pulses ")
axs[2].set_xlabel("t (ms)")
axs[2].set_ylabel("B1 ($\mu$T)")


#%%
### Pulse parameters
T,Tn,dw = 0.01,500,0
t = np.linspace(0,T,Tn)
#t = np.abs(np.linspace(-T/2,T/2,Tn))
#check k space limits
z = 1
FOV,grid_size,dn = 10/z,100,20
grid_spacing = FOV / grid_size
A = 1
n = 25
kxp = A*(1 - t/T)*np.cos(2*np.pi*n*t/T) 
kyp = A*(1 - t/T)*np.sin(2*np.pi*n*t/T)
kt = (kxp**2 + kyp**2)**0.5
###Defining Spatial localisation
B,alpha,A,gamma = 2,1,1,4.257*10**7  

Dk = alpha*np.exp(-(B**2)*(kxp**2 + kyp**2)/A**2) 

### Calculating gradient waveforms and B1 pulse
Gxp,Gyp = (np.gradient(kxp)/gamma),(np.gradient(kyp)/gamma)
B1 = Dk*gamma*np.sqrt(Gxp**2 + Gyp**2)
#k = 90/gamma*np.sum(B1)
#B1 = k*Dk*gamma*np.sqrt(Gxp**2 + Gyp**2)

M,store,Mcomplex = conversion(kxp,kyp, B1, gamma,FOV, grid_size,T,Tn,dw)
M = np.reshape(Mcomplex,(grid_size,grid_size))
#%% Figure to show how magnetisation is built up over many trajectories

#%%
plt.imshow(np.imag(M))
#%%  
N = 1  
magtotal,RMSarray,qarray,darray,B1f,kx,ky,B1total = segmented_trajectory_spiral(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,z)
N = 4
print(N)
magtotal4,RMSarray4,qarray4,darray,B1f4,kx4,ky4,B1total4 = segmented_trajectory_spiral(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,z)
N = 8
print(N)

magtotal8,RMSarray8,qarray8,darray,B1f8,kx8,ky8,B1total8 = segmented_trajectory_spiral(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,z)
#N = 16
print(N)
#magtotal16,RMSarray16,qarray16,darray,B1f16,kx16,ky16,B1total16 = segmented_trajectory_spiral(N,dn,n,t,frequencey_range,T,Tn,FOV,grid_size,test,B1,z)
#%% Stats for changing number of N
lowerrange = np.ones(len(t))*frequencey_range/4
upperrange = np.ones(len(t))*3*frequencey_range/4
fig, axs = plt.subplots(1,2, figsize=(12, 6))
plt.subplots_adjust(top = 0.948,bottom = 0.088 ,left = 0.066 ,right = 0.980,hspace = 0.330,wspace = 0.196 )
plt.rcParams.update({'font.size': 16})
a = axs[0].plot(darray,(qarray)/Norm,'b-',label="N = 1")
a = axs[0].plot(darray,(qarray4)/Norm,'r-',label="N = 4")
a = axs[0].plot(darray,(qarray8)/Norm,'g-.',label="N = 8")
#a = axs[0].plot(darray,(qarray16)/Norm,':',label="N = 16")
axs[0].legend(loc='upper right')
axs[0].set_title("Ratio of magnetisation response to target ")
axs[0].set_xlabel("Offset Frequencey $Hz$")
axs[0].set_ylabel("Fraction of Target excited")
#b = axs[0].inset_axes([0.3, 0.3, 0.47, 0.47])
axs[1].plot(darray,RMSarray/100,'-',label="N = 1")
axs[1].plot(darray,RMSarray4/100,'--',label="N = 4")
axs[1].plot(darray,RMSarray8/100,'-.',label="N = 8")
#axs[1].plot(darray,RMSarray16,':',label="N = 16")

axs[1].set_title("RMS Error as a function of offset frequencey ")
axs[1].set_xlabel("Offset Frequencey $Hz$")
axs[1].set_ylabel("RMS Error")
axs[1].legend(loc='upper right')

#%% Stats for changing number of N
array = np.arange(0,dn)
fig, axs = plt.subplots(3,2, figsize=(12, 6))
plt.subplots_adjust(top = 0.948,bottom = 0.088 ,left = 0.066 ,right = 0.980,hspace = 0.330,wspace = 0.196 )
plt.rcParams.update({'font.size': 12})
a = axs[0,0].plot(kx,ky,'-')
a1 = axs[1,0].plot(kx4,ky4,'-')
a2 = axs[2,0].plot(kx8,ky8,'-')


axs[0,0].set_title("k-space trajectory")
axs[0,1].set_yticklabels([])
axs[0,1].set_xticklabels([])
axs[1,1].set_yticklabels([])
axs[1,1].set_xticklabels([])
axs[2,1].set_yticklabels([])
axs[2,1].set_xticklabels([])
axs[0,0].set_ylabel("N = 1")
axs[1,0].set_ylabel("N = 4")
axs[2,0].set_ylabel("N = 8")
#b = axs[0].inset_axes([0.3, 0.3, 0.47, 0.47])
axs[0,1].set_title("Magnetisation Response")
axs[0,1].imshow(np.abs(magtotal[array[int(dn/2)],:,:]))

a4 = axs[0,1].imshow(np.abs(testr[0])*np.abs(magtotal[array[int(dn/2)],:,:])+np.abs(magtotal[array[int(dn/2)],:,:]))

axs[0,1].set_xlabel("{}% Excitation".format(np.around(qarray[int(dn/2)]/Norm,5)*100))
axs[0,1].set_ylabel("RMSE = {}".format(np.around(RMSarray[int(dn/2)]/Norm,5)*100))

a4 = axs[1,1].imshow(np.abs(testr[0])*np.abs(magtotal4[array[int(dn/2)],:,:])+np.abs(magtotal4[array[int(dn/2)],:,:]))
axs[1,1].set_ylabel("RMSE = {}".format(np.around(RMSarray4[int(dn/2)]/Norm,5)*100))
axs[1,1].set_xlabel("{}% Excitation".format(np.around(qarray4[int(dn/2)]/Norm,5)*100))
a = axs[2,1].imshow(np.abs(magtotal8[array[int(dn/2)],:,:]))
a4 = axs[2,1].imshow(np.abs(testr[0])*np.abs(magtotal8[array[int(dn/2)],:,:])+np.abs(magtotal8[array[int(dn/2)],:,:]))
axs[2,1].set_ylabel("RMSE = {}".format(np.around(RMSarray8[int(dn/2)]/Norm,5)*100))
axs[2,1].set_xlabel("{}% Excitation".format(np.around(qarray8[int(dn/2)]/Norm,5)*100))
cbar = fig.colorbar(a, ax=axs[:,1],fraction=0.6)
#%%
T =0.001
t = np.linspace(0,T,1000)

n = 10
kx,ky = kspiral(1,n,t,T,0)

kxp,kyp = krosette(1,n,t,T,0)
fig, axs = plt.subplots(1,2, figsize=(12, 6))
axs[0].plot(kx,ky)
axs[1].plot(kxp,kyp)
