# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:00:49 2015

@author: Amina Asif
"""
from scipy import sparse
from cv import *
from classifiers import *
from misvmio import *
from bag import *
from llc import *
def create_bags(dataset='musk1'):
    data_set= parse_c45(dataset)
    bagset = bag_set(data_set)
    bags_1 = [np.array(b.to_float())[:, 2:-1] for b in bagset]
    labels = np.array([b.label for b in bagset], dtype=float)
    no_of_bags=len(bags_1)
    for b_r in range(no_of_bags):
        for b_c in range(len(bags_1[b_r])):
            bags_1[b_r][b_c]=bags_1[b_r][b_c]/np.linalg.norm(bags_1[b_r][b_c])
            
    labels =list( 2 * labels - 1)
    bags= [Bag() for _ in range(len(bags_1))]
    bag_lengths=[] 
    for  i in range(len(bags_1)):
        bags[i].instances=bags_1[i]
        
        
        bag_lengths+=[ len(bags[i].instances)]
        bags[i].instances=np.array(bags[i].instances)

        #bags[i].instances+=[1.0]
        #print len(bags[i].instances)
        
        bags[i].label=labels[i]
        bags[i].peta=[1.0]*len(bags[i].instances)
    
       
        
    
    return bags
    
    
    
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
    
    
    
    
def eloss_fun(y, scores,**kwargs):
    return np.log(1+np.e**(y*scores.flatten()))
def eloss_dfun(x, y, z, **kwargs):
    return -((y/(1+np.e**(y*z)))*x)
    
if __name__ == '__main__':
    bags=create_bags('musk1')
#    compute_gammas(bags, C=50, k=20, beta=0.1)
    classifier=linclass(epochs=500, Lambda=0.00001)
    x=time.time()
    result= tenFoldCV(classifier, bags, parallel=1)
    print time.time() -x ,'seconds'
    
    auc=perFoldAUC(result[0], result[1])
    print "AUCs for ten Folds=", auc
    print "Mean AUC=", np.mean(auc)