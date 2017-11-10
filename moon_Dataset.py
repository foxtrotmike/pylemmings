# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:11:43 2015

@author: Amina Asif
"""
from scipy import sparse
from cv import *
from classifiers import *
import pdb
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets
from scipy import *
import numpy as np
from llc import llc

def compute_gammas(bags, C=50, beta=0.1, k=50):
    data_points=[]
    for bag_i in (bags):
        bag_i.gammas=[]
        for example_j in bag_i.instances:
            data_points+=[example_j]
    #pdb.set_trace()        
    G, C, Y=llc(X=data_points,C=C, beta=beta, k=k)
#    print np.shape(G)
    np.savetxt('Gammas.txt', G )
    
    gamma_index=0
    for bag_index in range(len(bags)):
        for ins_index in range(len(bags[bag_index].instances)):
            bags[bag_index].gammas+=[G[gamma_index]]
            gamma_index+=1
        bags[bag_index].gammas=np.array(bags[bag_index].gammas)

    return G, C, Y
    

        
if __name__ == '__main__':
    noisy_moons = datasets.make_moons(n_samples=2500, noise=0.09)
    #noisy_moons=datasets.make_classification(n_samples=2500, n_features=2,n_clusters_per_class=1,n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)
    #noisy_moons=datasets.make_circles(n_samples=2500)
    #noisy_moons=datasets.make_s_curve(n_samples=2500)
    
    instances=np.array(noisy_moons[0])
    labels=noisy_moons[1]
    pos=[]
    neg=[]
    for j in range(len(instances)):
        if labels[j]==1.0:
            pos+=[instances[j].tolist()]
        else:
            labels[j]=-1.0
            neg+=[instances[j].tolist()]
    pos=np.array(pos)
    neg=np.array(neg)  
    
#    pos=1.5+np.dot(np.random.randn(500,2), [[1,0],[0,0.5]])
#    pos_labels=[1.0]*len(pos)
#    neg=-1.5-np.dot(np.random.randn(500,2), [[1,0],[0,0.5]])
#    neg_labels=[-1.0]*len(neg)
#    instances=np.vstack((pos, neg))
#    labels=pos_labels+neg_labels
    plt.clf()
    
    
    x=np.arange(-2, 3, 0.025)
    y=np.arange(-2, 2, 0.025)
    test_data=[]
    
    
    for i in itertools.product(x,y):
        #print i
        test_data+=[i]
    test_data=np.array(test_data)
    #plt.scatter(test_data[:,0],test_data[:,1], c='green', marker='*', s=40 )
    
    
    train_bags=[Bag() for i in range(len(instances))]
    for b in range(len(train_bags)):
       # import pdb; pdb.set_trace()
        train_bags[b].instances=np.array([instances[b]])
        train_bags[b].label=labels[b]
        train_bags[b].peta=[1.0]
        
    test_bags=[Bag() for i in range(len(test_data))]
    for b in range(len(test_bags)):
        test_bags[b].instances=np.array([test_data[b]])
        test_bags[b].peta=[1.0]
        #test_bags[b].label=labels[b]
    all_bags=train_bags+test_bags 
    cent=compute_gammas(train_bags, C=100, k=30, beta=100.0)
    compute_gammas(test_bags, cent[1])
    #classifier=LLC_MIL_SVM(T=100, lambdaa=0.0001)
    classifier=llclass(epochs=500, Lambda=0.000001)
    classifier.train(train_bags, bias=True)
    labels_t=np.zeros(len(test_bags))
    predictions=[]
    for i in range(len(test_bags)):
        predictions+=[classifier.predict_bag(test_bags[i])]
        if (classifier.predict_bag(test_bags[i])>0):
            test_bags[i].label=1.0
            labels_t[i]=1.0
        else:
            test_bags[i].label=-1.0
            labels_t[i]=-1.0
    predictions=np.array(predictions)
    predictions=np.reshape(predictions, [200,160]).T  
    predictions=np.flipud(predictions)
    plt.imshow(predictions, extent = [-2,+3,-2,2])
    plt.colorbar()
    predictions=np.flipud(predictions)
#    plt.contour(predictions,[-1,0,+1],linewidths = [2],colors=('w','k','w'),extent=[-2,+3,-2,2])
    plt.contour(predictions,[0],linewidths = [2],colors=('k'),extent=[-2,+3,-2,2])
    
#   plt.figure()
    pos_classified=[]
    neg_classified=[]
    
#    for j in range(len(test_bags)):
#        if test_bags[j].label==1.0:
#            pos_classified+=[test_bags[j].instances[l] for l in range(len(test_bags[j].instances))]
#        else:
#            
#            neg_classified+=[test_bags[j].instances[l] for l in range(len(test_bags[j].instances))]
#    pos_c=np.array(pos_classified)
#    neg_c=np.array(neg_classified)
    centres=cent[1]
    
    plt.scatter(pos[:,0],pos[:,1] , c='green', marker='+', s=40)
    plt.scatter(neg[:,0],neg[:,1], c='black', marker='.', s=40 )
#    plt.scatter(centres[:,0],centres[:,1], c='red', marker='s', s=40 )
#    plt.scatter(pos_c[:,0],pos_c[:,1], c='blue', marker='*', s=60 )
#    plt.scatter(neg_c[:,0],neg_c[:,1], c='red', marker='*', s=60 )
    
    plt.show()        
    
        
#    
#        
#    