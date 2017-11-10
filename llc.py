
import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
import pdb
def llc(X, C = None, k = None, beta = 1e-6, **kwargs):
    """
    Implements Approximate Locally Linear Coding
    Inputs:
        X: (N x d) numpy array
        C: (Default: None) 
            integer: number of anchor points (kmeans used to obtain codebook)
            OR
            (c x d) array of anchor points (codebook)
        k: (Default: None) Number of nearest neighbors for sparsity. If k > c or k<1, then k is changed to c
        beta: regularization parameter (lambda in the paper)
    Outputs:
        (G,C,distortion)
            G: Gamma coefficients (N x c) numpy array
            C: Codebook (c x d)
            Y: The transformed points (N x d) Y = G*C
    """
    if type(C)==type(0):
        #C,_ = kmeans(X, C, **kwargs)  
        clf=KMeans(n_clusters=C)
        clf.fit(X)
        C= clf.cluster_centers_
    #pdb.set_trace()   
    X=np.array(X)
    assert X.shape[1]==C.shape[1]
    N,d = X.shape
    c,_ = C.shape
    if k is None or k < 1 or k>c:
        print "Warning: k set to ",c
        k = c    
    D = cdist(X, C, 'euclidean')
    I = np.zeros((N, k),dtype=int)
    for i in range(N):
        d = D[i,:]
        idx = np.argsort(d)
        I[i, :] = idx[:k]
    
    II = np.eye(k)
    G = np.zeros((N, c)) #Gammas
    ones = np.ones((k,1))
    for i in range(N):
       idx = I[i,:]
       z = C[idx,:] - np.tile(X[i,:], (k, 1))  # shift ith pt to origin
       Z = np.dot(z,z.T)                       # local covariance
       Z = Z + II*beta*np.trace(Z);            # regularlization (K>D)
       w = np.linalg.solve(Z,ones)     #np.dot(np.linalg.inv(Z), ones)
       w = w/np.sum(w)                         # enforce sum(w)=1
       
       w=np.ravel(w)
       G[i,idx] = w.T
    Y = np.dot(G,C)
    return G, C, Y
    
if __name__=='__main__':
    # Let's generate some random data
    X = np.vstack((\
        np.dot(np.random.randn(300,2), [[1, -1],[-1, 0.25]])+[-2,2],\
        np.dot(np.random.randn(300,2), [[1, 1],[1, 0.25]])+[2,2],\
        np.dot(np.random.randn(300,2), [[0.25, 0],[0, 1]])+[0,-2]
        )) 

    G, C, Y = llc(X, C = 8, k = 1, beta = 100)
    D = Y-X
    distortion = np.sqrt(np.sum((D)**2,axis = 1))
    dm = np.mean(distortion)
    ds = np.std(distortion)
    nd = 8+6*((distortion-dm) / ds) # size of point shows relative distortion
    print 'Mean (std) Distortion: %0.4g (%0.4g)' %(dm,ds)
    # Plotting
    import matplotlib.pyplot as plt    
#    for i in range(X.shape[0]):        
#        plt.annotate(
#        '', xy=X[i,:2], xycoords='data',
#        xytext=Y[i,:2], textcoords='data',
#        arrowprops={'arrowstyle': '<-'})
    plt.scatter(X[:,0],X[:,1],color = 'r', marker = '.')
#    plt.scatter(C[:,0],C[:,1],color = 'b',marker = 's')
    
    plt.grid()
    plt.title('Mean (std) Distortion: %0.4g (%0.4g)' %(dm,ds))
    plt.legend(['data points','anchor points'])
    plt.show()