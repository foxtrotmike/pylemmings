# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:43:00 2015

@author: Afsar
"""
import numpy as np
from random import random
import time

class Bag:
    def __init__(self):
        self.val = 2
        
class Classifier:        
    def __init__(self,beginner = None,**kwargs):
        self.name = random()
        
def trainTest(classifier,traindata,testdata,**kwargs):
    return np.ones(len(testdata))*classifier.name, np.zeros(len(testdata))*classifier.name
    
class Fold:
    def __init__(self):
        self.traini = [1]        
        self.testi = [2,3]
        
def cv(classifierTemplate,bags,folds,**kwargs):
    """
    Returns a list with one element for each of the given folds
    """
    def cvOne(f):
        # performs a single fold (using trainTest)
        classifier = Classifier(classifierTemplate,**kwargs)
        time.sleep(2) #just a delay
        # TODO: here generate test and train data for each fold from given indices in f and bags
        # i just past the indices
        r = trainTest(classifier,f.traini,f.testi,**kwargs)
        return r
        
    if type(folds) <> type([]): #when there is only one fold, no need to pass a list
        return cvOne(folds)       
        
    if 'parallel' in kwargs:
        numproc = kwargs['parallel']
        kwargs.pop('parallel')        
        if numproc: # if user wants parallelism   
            from joblib import Parallel, delayed #import only when it's needed
            if type(numproc) <> type(0) or numproc < 0: #if user has not specified number of processes
                numproc = 4 #default 4 processors        
            print "Using",numproc,"Processors"            
            result= Parallel(n_jobs=numproc, verbose = True)\
                (delayed(cv)(classifierTemplate,bags,f,**kwargs) for f in folds)  
            #import pdb; pdb.set_trace()
            return result
            
    return [cvOne(f) for f in folds ] #serial execution

if __name__=='__main__':
    classifierTemplate = Classifier()
    bags = []
    for _ in range(10):
        b = Bag()
        b.val = _
        bags.append(b)
    f0 = Fold()
    f1 = Fold()
    f1.traini = [0,1,2,3]
    f1.testi = [4,5,6]
    Folds = [f0,f1]*2 #total 4 folds
    print "With a single fold"
    print cv(classifierTemplate,bags,f0)
    print "With ",len(Folds)," folds in parallel"
    result= cv(classifierTemplate,bags,Folds,parallel = 2)
    print "With ",len(Folds)," folds in serial"
    print cv(classifierTemplate,bags,Folds,parallel = False)