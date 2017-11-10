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
from scipy.cluster.vq import kmeans2
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
            'distance': The distance metric used (see help on cdist) to see 
                all options. Default: euclidean
        
    Outputs:
        (G,C,distortion)
            G: Gamma coefficients (N x c) numpy array (nonsparse)
            C: Codebook (c x d) (nonsparse)
            Y: The transformed points (N x d) Y = G*C (nonsparse)
    """
    if 'distance' in kwargs:
        distance = kwargs['distance']
        kwargs.pop('distance',None)
        warnings.warn('If using kmeans, euclidean distance will be used.')
    else:
        distance = 'euclidean'
    if issparse(C): # should have enought memory to save dense anchor points
        C = C.toarray()
    if type(C)==type(0):
        if 'sample' in kwargs:
            ns = kwargs['sample']
            kwargs.pop('sample',None)
            ridx = np.random.randint(X.shape[0],size=ns)  #random indices
            if issparse(X):
                Xd = X[ridx,:].toarray()
            else:
                Xd = X[ridx,:]
            C,_ = kmeans2(Xd, C, **kwargs)
            Xd = None
        else:
            if issparse(X):
                raise ValueError(" \' sample \' argument or anchor points \
                must be used when data is sparse.")
            C,_ = kmeans2(X, C, **kwargs)  
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
    
    for i in range(N):
       xi = X[i,:]
       if issparse(xi):
           xi = xi.toarray()
       d = cdist(np.atleast_2d(xi), C, distance)[0]       
       idx = np.argsort(d)
       idx = idx[:k]
       ci = C[idx,:]
       z = ci - xi #np.tile(xi, (k, 1))  # shift ith pt to origin       
       Z = np.dot(z,z.T)                       # local covariance
       Z = Z + II*beta*np.trace(Z);            # regularlization (K>D)
       w = np.linalg.solve(Z,ones)     #np.dot(np.linalg.inv(Z), ones)
       w = w/np.sum(w)                         # enforce sum(w)=1       
       G[i,idx] = w.ravel()
    
    Y = G.dot(C)
    return G, C, Y
    
if __name__=='__main__':
    # Let's generate some random data
    X = np.vstack((\
        np.dot(np.random.randn(200,2), [[1, -1],[-1, 0.25]])+[-2,2],\
        np.dot(np.random.randn(200,2), [[1, 1],[1, 0.25]])+[2,2],\
        np.dot(np.random.randn(200,2), [[0.25, 0],[0, 1]])+[0,-2]
        )) 
    C = 10
    X = lil_matrix(X).tocsr() #sparse testing
    #C = X[np.random.randint(X.shape[0],size=C),:] #user can specify anchor points
    
    G, C, Y = llc(X, C = C, k = 3, beta = 0.002, sample = 100 )#,distance = 'cosine'
    
    # PLOTTING DOESN'T WORK WITH SPARSE
    if issparse(X):
        X = X.toarray()    
        Y = np.array(Y)
    D = Y-X
    distortion = np.sqrt(np.sum((D)**2,axis = 1))
    dm = np.mean(distortion)
    ds = np.std(distortion)
    nd = 8+6*((distortion-dm) / ds) # size of point shows relative distortion
    print 'Mean (std) Distortion: %0.4g (%0.4g)' %(dm,ds)
#    # Plotting
    import matplotlib.pyplot as plt    
    for i in range(X.shape[0]):        
        plt.annotate(
        '', xy=X[i,:2], xycoords='data',
        xytext=Y[i,:2], textcoords='data',
        arrowprops={'arrowstyle': '<-'})
    plt.scatter(X[:,0],X[:,1],color = 'b', marker = 'o',sizes=nd)
    plt.scatter(C[:,0],C[:,1],color = 'r',marker = 's',sizes = [50]*X.shape[0])
    
    plt.grid()
    plt.title('Mean (std) Distortion: %0.4g (%0.4g)' %(dm,ds))
    plt.legend(['data points','anchor points'])
    plt.show()
