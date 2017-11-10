# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:59:03 2015

@author: Amina Asif

"""
import numpy as np

from scipy.sparse import *
class Bag:
    def __init__(self):
        self.instances=[]
        self.label=None
        self.gammas=[]
        self.id=None  
        self.peta=[]
        
    def addInstance(self,x):
        if type(self.instances) == type([]) and not len(self.instances):
            
            self.instances = x
            
            if not issparse(x):
                self.instances=np.array(self.instances)[np.newaxis, :]
        else:
            if issparse(x):
                self.instances = vstack([self.instances,x])
            else: 
                self.instances =np.vstack([self.instances,x])
    def addPeta(self,x):# probability of selection
        if type(self.peta) == type([]) and not len(self.peta):
            
            self.peta = x
            
            if not issparse(x):
                self.peta=np.array(self.peta)
        else:
            if issparse(x):
                self.peta = hstack([self.peta,x])
            else: 
                self.peta =np.hstack([self.peta,x])
                
    def addGamma(self,x):
        if type(self.gammas) == type([]) and not len(self.gammas):
            
            self.gammas= x
            
            if not issparse(x):
                self.gammas=np.array(self.gammas)
        else:
            if issparse(x):
                self.gammas = vstack([self.gammas,x])
            else: 
                self.gammas =np.vstack([self.gammas,x])
    def __len__(self):
        if type(self.instances) == type([]):
            return len(self.instances)
        else:
            return int(self.instances.shape[0]) #CHANGED
    def __repr__(self):
        return "Bag ID: "+str(self.id)+",Bag Label: "+str(self.label)+", No. of Instances: "+str(len(self))+"\n" #CHANGED
    
    def __getitem__(self,k): #CHANGED
        return self.instances[k]
        