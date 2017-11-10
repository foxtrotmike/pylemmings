# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:24:12 2015

@author: Amina Asif
"""
from classifiers import *
import numpy as np
from scipy.io import *
from bag import *
from llc import *


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
        for ins in b.instances:
            if data_points==None:
                data_points=np.array(ins[np.newaxis,:])
            else:
                data_points=np.vstack((data_points, np.array(ins[np.newaxis,:])))
            #data_points+=[example_j] 
   # data_points=vstack(data_points)
#    import pdb; pdb.set_trace()
    G, C, Y=llc(X=data_points,C=C, beta=beta, k=k)
    
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
def readBags(filename_f, filename_l, **kwargs):
    feature_matrix=loadmat(filename_f)['PM'][0]
    labels=loadmat(filename_l)['labels'][0]
    bags=[]
    for i in range(80):
        f=np.array(feature_matrix[i].T)
       # f=np.array(np.hstack((feature_matrix[i][:2].T, feature_matrix[i][2:].T)))
        #import pdb; pdb.set_trace()
        b=Bag()
        b.instances=f
        b.label=labels[i]
        b.peta=[1.0]*(np.shape(f)[0])
        bags.append(b)
    return bags
    
if __name__ == '__main__':
    
    
    bags=readBags('finalfeatures.mat', 'pmlabels.mat')
    print "bags created..."
    compute_gammas(bags, C=40, k=40, beta=100.0)
    classifier=llclass(epochs=500, Lambda=1.0)
    #r=cv(classifier, bags, 10)
    r=LOO(classifier, bags,bias=True, parallel=4)
    
    auc=AUC(r[0], r[1])
    print "AUC =", auc
#    print "Mean AUC=", np.mean(auc)