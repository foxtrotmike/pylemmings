# -*- coding: utf-8 -*-
"""
Created on Tuesday September 15 02:53:32 2015
@author: Dr. Fayyaz ul Amir Afsar Minhas (afsar <at> pieas dot edu dot pk)
version 2.0
Description:
This module implements the approximate Locality Constrained Linear Coding as described in the 2010 paper by Wang et al. [1]. Given array of datapoints X (N x d) and codebook C (c x d), it returns a vector of approximated points Y = G * C. LLC introduces sparsity by forcing those coefficients of a given data point that correspond to codebook vectors which are not that point's k-nearest neighbors. LLC also uses regularization.
This code has been verified to produce the same LLC coefficients as the original matlab implementation by Wang et al. [1] available at: www.ifp.illinois.edu/~jyang29/LLC.htm
However, this code has a test script which can be used to study the workings of the LLC method on a toy dataset.
Usage: from llc import llc
Running as a script:
When run as a script, this code will plot a toy data and show distortion of each data point (proprtiontal to marker size). There will be arrows indicating the original location of the point and the transformed location.
References:
[1] Wang, Jinjun, Jianchao Yang, Kai Yu, Fengjun Lv, T. Huang, and Yihong Gong. “Locality-Constrained Linear Coding for Image Classification.” In 2010 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3360–67, 2010. doi:10.1109/CVPR.2010.5540018.

Change log:
    Added sparse support
    Improved memory fingerprint

NOTE ON SPARSE DATA USAGE
    There must be memory to store all anchor points in dense form.
    This allows us to improve the time performance of the algorithm.
"""
from scipy.sparse import lil_matrix, issparse
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
#from sklearn.neighbors import BallTree as cKDTree
from scipy.cluster.vq import kmeans as kmeans
import warnings
import numpy as np

def llc(X, C = None, k = None, beta = 1e-6, **kwargs):
    """
    Implements Approximate Locally Linear Coding
    Inputs:
        X: (N x d) numpy array or scipy sparse matrix
        C: (Default: None) 
            integer: number of anchor points (kmeans used to obtain codebook)
            OR
            (c x d) array of anchor points (codebook) or scipy sparse matrix
        k: (Default: None) Number of nearest neighbors for sparsity. 
            If k > c or k<1, then k is changed to c
        beta: regularization parameter (lambda in the paper)
        keyword arguments:
            'sample': (positive integer) When specified, this many points are 
                randomly chosen from the data prior to applying kmeans
                    Either C or sample must be specified if X is sparse
                    When this argument is not specified, all data is used in kmeans.
            'distance': The distance metric used to chose nearest anchor points
                (see help on cdist) to see all options. Default: euclidean
            'reconstruct': (boolean) default: False
                transformed the data. When operating on big data it should be false.
    Outputs:
        (G,C,distortion)
            G: Gamma coefficients (N x c) numpy array (nonsparse)
            C: Codebook (c x d) (nonsparse)
            Y: The transformed points (N x d) Y = G*C (nonsparse)
                None when reconstruct = False
                
    """
    if 'distance' in kwargs:
        distance = kwargs['distance']
        kwargs.pop('distance',None)
        warnings.warn('If using kmeans, euclidean distance will be used.')
    else:
        distance = 'euclidean'
    if 'reconstruct' in kwargs:           
        reconstruct = kwargs['reconstruct']
        kwargs.pop('reconstruct',None)
    else: 
        reconstruct = False
        
    if issparse(C): # should have enought memory to save dense anchor points
        C = C.toarray()
    if type(C)==type(0):
        if 'sample' in kwargs:
            ns = kwargs['sample']
            kwargs.pop('sample',None)
            ridx = np.random.choice(X.shape[0],ns)  #random indices
            if issparse(X):
                Xd = X[ridx,:].toarray()
            else:
                Xd = X[ridx,:]
            C,_ = kmeans(Xd, C, **kwargs)
            Xd = None #release memory
        else:
            if issparse(X):
                raise ValueError(" \' sample \' argument or anchor points \
                must be used when data is sparse.")
            C,_ = kmeans(X, C, **kwargs)  
    else:
        if 'sample' in kwargs:
            warnings.warn('Anchor points have been specified, ignoring \'sample\'.')
            
    assert X.shape[1]==C.shape[1]
    
    N,d = X.shape
    c,_ = C.shape
    if k is None or k < 1 or k>c:        
        warnings.warn('k set to %d.' % c)
        k = c    

    II = np.eye(k)
    G = np.zeros((N, c)) #Gammas
    ones = np.ones((k,1))
    TT = cKDTree(C,leafsize = X.shape[1]+1)
    if issparse(X):
        blocks = min(max((1,(X.shape[0]/C.shape[0])/2)),X.shape[0])
    else:
        blocks = 1        
        Xd = X
    for j in range(blocks):
       begin = j*N/blocks 
       if issparse(X):
           Xd = X[begin:(j+1)*N/blocks,:].toarray()    
       for i in range(Xd.shape[0]):
           xi = Xd[i,:]           
           _,idx = TT.query(xi,k = k)
    
           ci = C[idx,:]
           z = ci - xi #np.tile(xi, (k, 1))  # shift ith pt to origin       
           Z = np.dot(z,z.T)                       # local covariance
           #import pdb; pdb.set_trace()
           Z = Z + II*beta*np.trace(Z);            # regularlization (K>D)
           w = np.linalg.solve(Z,ones) #np.linalg.solve    #np.dot(np.linalg.inv(Z), ones)
           w = w/np.sum(w)                         # enforce sum(w)=1       
           G[begin+i,idx] = w.ravel()
    if reconstruct:
        Y = G.dot(C)        
    else:
        Y = None
    return G, C, Y
    
if __name__=='__main__':
    # Let's generate some random data
#    X = np.vstack((\
#        np.dot(np.random.randn(200,2), [[1, -1],[-1, 0.25]])+[-2,2],\
#        np.dot(np.random.randn(200,2), [[1, 1],[1, 0.25]])+[2,2],\
#        np.dot(np.random.randn(200,2), [[0.25, 0],[0, 1]])+[0,-2]
#        )) 
    X = np.random.rand(50000,420)
    X = (X<0.05)*X
    C = 200
    samples = 10*C #10 times C
    X = lil_matrix(X).tocsr() #sparse testing
    #C = X[list(set(np.random.choice(X.shape[0],C))),:] #user can specify anchor points
    
    G, _, _ = llc(X, C = C, k = 10, beta = 0.002, sample = samples) #, reconstruct = True
    
    # PLOTTING DOESN'T WORK WITH SPARSE
#    if issparse(X):
#        X = X.toarray()    
#        Y = np.array(Y)
#    D = Y-X
#    distortion = np.sqrt(np.sum((D)**2,axis = 1))
#    dm = np.mean(distortion)
#    ds = np.std(distortion)
#    nd = 8+6*((distortion-dm) / ds) # size of point shows relative distortion
#    print 'Mean (std) Distortion: %0.4g (%0.4g)' %(dm,ds)
##    # Plotting
#    import matplotlib.pyplot as plt    
#    for i in range(X.shape[0]):        
#        plt.annotate(
#        '', xy=X[i,:2], xycoords='data',
#        xytext=Y[i,:2], textcoords='data',
#        arrowprops={'arrowstyle': '<-'})
#    plt.scatter(X[:,0],X[:,1],color = 'b', marker = 'o',sizes=nd)
#    plt.scatter(C[:,0],C[:,1],color = 'r',marker = 's',sizes = [50]*X.shape[0])
#    
#    plt.grid()
#    plt.title('Mean (std) Distortion: %0.4g (%0.4g)' %(dm,ds))
#    plt.legend(['data points','anchor points'])
#    plt.show()
