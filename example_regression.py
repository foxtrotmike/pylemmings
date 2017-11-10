# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 00:28:39 2015

@author: Amina Asif
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 01:18:46 2015

@author: fayyaz
"""
from sklearn.metrics import mean_squared_error
import re
#from mpi4py import MPI
import numpy as np
from scipy.sparse import lil_matrix, vstack, coo_matrix, issparse
from scipy.sparse import *
from cv import *
from classifiers import *
#from misvmio import * 
import matplotlib.pyplot as plt
from bag import Bag
from llc import *


def create_bags(fname):
    B = []
    bidx = 0
    d = None
    N = None
    rmode = None
    with open(fname,'r') as ifile:
        for ln in ifile:
            lns = ln.strip()
            if lns =='# Number of Dimensions':
                rmode = 'dim'
                continue
            elif lns == '# Number of Examples':
                rmode = 'num'
                continue
            elif re.match('# Number of Instances of molecule \(\d+\)', lns) is not None:
                molid = int(lns.split('(')[1].split(')')[0])            
                rmode = 'bag'
                continue
            elif re.match('# Label of molecule \(\d+\)', lns) is not None:
                molid = int(lns.split('(')[1].split(')')[0])
                rmode = 'lab'
                continue
      
            if rmode == 'dim':
                d = int(lns)
                continue
            if rmode == 'num':
                N = int(lns)
                continue
            if rmode =='bag':
                ni = int(lns)
                rmode = 'fea'
                X = []
                continue
            if rmode == 'lab':
                assert len(X) == ni*d #ensure correct reading
                l = float(lns)
                X = np.array(X)
                b = Bag()
#                b.instances =  X.reshape((ni,d))# 
                b.instances=X.reshape((d,ni)).T # (Amina, look here!)
                for bb in b.instances:
                    np.append(bb,1.0)
                b.label = l
                b.id = molid
                B.append(b)
                b.peta=[1.0]*np.shape(b.instances)[0]
                continue
                
            if rmode == 'fea':
                X.append(float(lns))
            
    return B
def compute_gammas(bags, C=50, beta=0.1, k=50):
    data_points=None
#    for bag_i in (bags):
#        
#        if issparse(bags[0].instances):
#            for example_j in (bag_i.instances).toarray():
#                data_points+=[example_j]
#        else:
        #for example_j in (bag_i.instances):
    for b in bags:
        b.gammas=[]
        if issparse(bags[0].instances):
                
            if data_points==None:
                data_points=b.instances
            else:
                data_points=vstack((data_points, b.instances))
        else:    
            if data_points==None:
                data_points=b.instances
            else:
                data_points=np.vstack((data_points, b.instances))

    G, C, Y=llc(X=data_points,C=C, beta=beta, k=k, sample=100)
    
    gamma_index=0
    for bag_index in range(len(bags)):
        for ins_index in range(np.shape(bags[bag_index].instances)[0]):
            bags[bag_index].gammas+=[G[gamma_index]]
            gamma_index+=1
        bags[bag_index].gammas=np.array(bags[bag_index].gammas)
    if issparse(bags[0].instances):        
        for bag in bags:
            bag.gammas=lil_matrix(bag.gammas).tocsr()

    return G, C, Y      
def set_peta(bags):
    protein_ids=[]
    for b in bags:
        protein_ids+=[b.id]
    protein_ids=list(set(protein_ids))
    num_bags=np.zeros(len(protein_ids))
    idx=-1    
    for ID in protein_ids:
        
        idx+=1
        for b in range(len(bags)):
            if bags[b].id==ID:
                
                num_bags[idx]+=1
    dict1=zip(protein_ids, num_bags)
    dict1=dict(dict1)
    #print dict1
    for key in dict1.keys():
        if dict1[key]==2.0:
            for b in bags:
                if b.id==key and b.label==-1.0:
                    b.peta=[0.5]*(np.shape(b.instances)[0])
                else:
                    b.peta=[1.0]*(np.shape(b.instances)[0])
    #pdb.set_trace()
                    
def eloss_fun(label, scores, **kwargs):
	#Compute the loss function given the score and the data
	#for the epsilon sensitive loss
	# (x,y) passed as data, you may want to pass a number of (score,data) simultaneously for vectorization and speed
    
    if 'epsilon'in kwargs:
        epsilon = kwargs['epsilon']
    else:
        epsilon=0.1    
#    epsilon = kwargs['epsilon']
    return np.max(np.vstack((abs(label-scores.flatten())-epsilon,np.zeros(len(scores)))),axis = 0)
	
def eloss_dfun(x, y, z, **kwargs):
	#Compute the gradient of the loss function given the score and the data
	#for the epsilon sensitive loss
	# (x,y) passed as data, you may want to pass a number of (score,data) simultaneously for vectorization and speed
    if 'epsilon'in kwargs:
        epsilon = kwargs['epsilon']
    else:
        epsilon=0.1
          
    if z-y > epsilon:
        return x
    elif y - z > epsilon:
        return -x
    else:
        return np.zeros(len(x))
		
       
if __name__ == '__main__':
    fname = 'LJ-16.30.2'
    fname_2='LJ-16.30.2.test'
    
    train_bags=create_bags(fname)
    test_bags=create_bags(fname_2)
            
    print "bags created..."
    
    
    #set_peta(bags)
    
    #compute_gammas(bags, C=2, k=20, beta=0.1)
    classifier = linReg(epochs = 100000, Lambda=1.0, lfun=eloss_fun,dlfun=eloss_dfun)
    
#    comm = MPI.COMM_WORLD
#    myid = comm.Get_rank()
#    nprocs = comm.Get_size(); print 'nprocs=', nprocs
#    lopo_parallel(classifier, bags,comm=comm,nprocs=nprocs,myid=myid)
#    x=time.time()
#    result=tenFoldCV(classifier, bags,  parallel=4,    epsilon=0.1)
#    print time.time() -x ,'seconds'
#    print mean_squared_error(result[1], result[0])


    classifier.train(train_bags)
       
    dec_scores= classifier.test(test_bags)
    labels=[]
        
    labels+=[b.label for b in test_bags]   
    print mean_squared_error(labels, dec_scores)
    