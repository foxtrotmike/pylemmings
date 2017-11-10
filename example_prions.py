# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 01:18:46 2015

@author: fayyaz
"""
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
    idx = -1
    dim = None
    Bags = []
    with open(fname,'r') as ifile:
        #print ifile
        for ln in ifile:
            
            ln = ln.strip()
            if not ln or ln[0]=='#': #ignore white spaces
                #print 'empty or comment'
                continue
            if ln[0]=='>':
                lns = ln.split()
                ccmd = lns[0].lower()
                text = lns[1:]
                if ccmd == '>dim' and dim is None:
                    dim = int(lns[1])
                    continue
                if ccmd == '>bag':                
                    idx += 1
                    label = int(text[0])
                    comment = ' '.join(text[1:])
                    Bags.append(Bag())
                    cbag = Bags[idx]
                    cbag.label = label
                    cbag.id = comment
                    
            else:
                
                feature_str=ln.split('#')[0]
                #print feature_str
                x=None
                if feature_str.find(':')>=0:
                    non_sparse_vector=np.zeros(dim)
                    feature_str=feature_str.split(',')
                    indices=[]
                    features=[]
                    for feature in feature_str:
                        if feature <> '':
                            indices+=[int(feature.split(':')[0])]
                            features+=[float(feature.split(':')[1])]
                    
                    for l in range(len(indices)):
                        
                        ind=indices[l]
                        non_sparse_vector[ind]=features[l]
                        
                        #import pdb; pdb.set_trace()
                    
                    
                    x=non_sparse_vector
                                       
                    
                else:
                        
                        
                    feature_str=feature_str.split(',')
                    
                    x=[float(i) for i in feature_str if i<>'']
                    #x=[float(i) for i in feature_str]
                    
                    x=np.array(x) 
                    #import pdb; pdb.set_trace()
                    
                if dim is not None:
                        x=lil_matrix(x).tocsr()
                        #pdb.set_trace() 
                #print x
                #import pdb; pdb.set_trace()
                    
    #                
    #                #continue
    #        
    #        x = coo_matrix((1,dim)) #parse the line 
                #pdb.set_trace()
                cbag.addInstance(x)
            cbag.peta=[1.0]*np.shape(cbag.instances)[0]
    return Bags
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
       
if __name__ == '__main__':
    fname = 'prions40_1s.lem'
    
    bags=create_bags(fname)
    print "bags created..."
    #set_peta(bags)
    
    compute_gammas(bags, C=2, k=20, beta=0.1)
    classifier=llclass_rank(epochs=50, Lambda=5e-6)
    
#    comm = MPI.COMM_WORLD
#    myid = comm.Get_rank()
#    nprocs = comm.Get_size(); print 'nprocs=', nprocs
#    lopo_parallel(classifier, bags,comm=comm,nprocs=nprocs,myid=myid)
    x=time.time()
    result=LOPO(classifier, bags, merge=True, bias=True, normalize=True, parallel=4)
    print time.time() -x ,'seconds'
    auc=AUC(r[0], r[1])
    print "AUC =", auc
#    
    
    