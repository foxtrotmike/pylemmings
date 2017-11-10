# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 01:51:29 2015

@author: amina
"Cross Validation Module"
contains:
Class definition for 'fold'
create_folds
trainTest
cv
test
"""



import random

import numpy as np
#from sklearn import metrics
from roc import *
from result import Result
import time


def perFoldAUC(dec_scores, labels):
    AUCs=[]
    for i in range(len(dec_scores)):
        fpr, tpr, auc= roc(dec_scores[i], labels[i])
        AUCs+=[auc]
    return AUCs
def AUC(dec_scores, labels):
    dec_scores=list(np.array(dec_scores).flatten())
    labels=list(np.array(labels).flatten())
    
    fpr, tpr, auc= roc(dec_scores, labels)
    return auc
        

class fold:
    
    """
    Contains a list of indices for training and testing bags
    fold.train_bags: list (of indices) of training bags
    fold.test_bags: list (of indices) of testing bags
    """
    def __init__(self):
        self.train_bags=[]
        self.test_bags=[]
        
def separate_bags_multi(bags, classes):
    sep_bags=[]
    for c in classes:
        l_b=[]
        for ind in range (len(bags)):
            if bags[ind].label==c:
                l_b+=[ind]
        sep_bags+=[l_b]
    return sep_bags
      
def separate_bags(bags): 
    """
    Seperates the positive and negative bags
    takes a list of bags as input and returns list of indices for positive and negative bags
    pos, neg=separate_bags(bags)
    """
    #random.shuffle(bags)
    pos_bags=[]
    neg_bags=[]
    for ind in range (len(bags)):
        if bags[ind].label==1.0:
            pos_bags+=[ind]
        else:
            neg_bags+=[ind]
    return pos_bags, neg_bags
 
           
def create_folds(bags, no_of_folds):
    """
    Creates folds from the given data.
    Takes a list of bags and the desired number of folds as input.
    Returns a list of fold objects
    """
    folds=[fold() for _ in range(no_of_folds)]
    
    pos, neg=separate_bags(bags)
    test_ex_n_n=len(neg)/no_of_folds
    test_ex_n_p=len(pos)/no_of_folds    
    for i in range(no_of_folds):
        folds[i].train_bags=[]
        
        if (i>0):
            folds[i].train_bags = pos[0:i*test_ex_n_p]
            folds[i].train_bags=np.append(folds[i].train_bags, neg[0:i*test_ex_n_n])           
            
        if (i<(no_of_folds-1)):
            folds[i].train_bags=np.append(folds[i].train_bags, pos[i*test_ex_n_p+test_ex_n_p:])
            folds[i].train_bags=np.append(folds[i].train_bags, neg[i*test_ex_n_n+test_ex_n_n:])
            
        folds[i].test_bags = pos[i*test_ex_n_p:i*test_ex_n_p+test_ex_n_p]
        folds[i].test_bags = np.append(folds[i].test_bags,neg[i*test_ex_n_n:i*test_ex_n_n+test_ex_n_n])
        
    return folds
 
   



def trainTest(classifier_temp,train_bags,test_bags):
    """
    Trains the classifier over train_bags and returns the decision scores of test_bags
    Takes a classifier object, list of training bags and list of test bags as input
    Returns the list of decision scores    
    """
    classifier=classifier_temp.__class__(classifier_temp)
    classifier.train(train_bags, merge=True)
    return classifier.test(test_bags)
    

#################################################################################################            

    
    
def CV(classifier_temp,bags,folds, **kwargs):
    #import pdb; pdb.set_trace()
    def CVone(f, **kwargs):
        
        train_bags=[];  
        #print 'My id is', myid, 'started processing fold number', fold_number 
        for index in range(len(f.train_bags)):          
            train_bags+=[bags[int(f.train_bags[index])] ]
        test_bags=[]
        m=None
        n=None
        bi=None
        for index in range(len(f.test_bags)):
            test_bags+=[bags[int(f.test_bags[index])] ]
#        if 'merge' in kwargs and kwargs['merge']==True:
#            m=True
#        else:
#            m=False
#            
#        if 'normalize' in kwargs and kwargs['normalize']==True:
#            n=True
#        else:
#            n=False
#            
#        if 'bias' in kwargs and kwargs['bias']==True:
#            bi=True
#        else:
#            bi=False
        
        classifier=classifier_temp.__class__(classifier_temp)
        
        classifier.train(train_bags, **kwargs)
       
        dec_scores= classifier.test(test_bags, **kwargs)
        labels=[]
        
        labels+=[b.label for b in test_bags]
        return dec_scores, labels
        
        
#    import pdb; pdb.set_trace()
    if type(folds) <> type([]): #when there is only one fold, no need to pass a list
        return CVone(folds, **kwargs) 
        
    pred_scores=[]
    true_labels=[]
    if 'parallel' in kwargs:
        
        numproc = kwargs['parallel']
        #kwargs.pop('parallel')        
        if numproc: # if user wants parallelism   
            from joblib import Parallel, delayed #import only when it's needed
            if type(numproc) <> type(0) or numproc < 0: #if user has not specified number of processes
                numproc = 1 #default 4 processors        
            print "Using",numproc,"Processors"            
            result= Parallel(n_jobs=numproc, verbose = True)\
                (delayed(CV)(classifier_temp,bags,f,**kwargs) for f in folds) 
            for i in range(len(result)):
                pred_scores+=[result[i][0]]
                true_labels+=[result[i][1]]
            #import pdb; pdb.set_trace()
            return pred_scores, true_labels
   
            
            
    
    
    for f in folds:
        

        p,l=CVone(f, **kwargs)
        pred_scores+=[p]
        true_labels+=[l]
    return pred_scores, true_labels
    
    
            
             
################################################################################################   
def tenFoldCV(classifier_temp,bags, **kwargs):
    Folds=create_folds(bags,10)
    
    return CV(classifier_temp,bags,Folds, **kwargs)
    
def LOPO(classifier_temp,bags, **kwargs):
    protein_ids=[]
    for b in bags:
        protein_ids+=[b.id]
    protein_ids=list(set(protein_ids))
    folds=[]
    idx=-1
    for ID in protein_ids:
        folds.append(fold())
        idx+=1
        for b in range(len(bags)):
            if bags[b].id==ID:
                
                folds[idx].test_bags+=[b]
            else:
                folds[idx].train_bags+=[b]
    
    return CV(classifier_temp,bags,folds, **kwargs)
    
def LOO(classifier_temp,bags, **kwargs):
    folds=[]
    for i in range(len(bags)):
        f=fold()
        f.test_bags+=[i]
        if i <> 0:
            f.train_bags+=range(0,i)
        
        if i <>(len(bags)-1):
            f.train_bags+=range(i+1, len(bags))
            
        folds.append(f)
    return CV(classifier_temp,bags,folds, **kwargs)
    
###################################################################################################            


def test(classifier,data, **kwargs):
    """
    Test a classifier over a list of bags
    Takes classifier object and list of bags as data
    Returns a list of decision scores
    """
    scores = []
    for instance in data:
        scores.append(classifier.predict_bag(instance, **kwargs))
    return scores
    




