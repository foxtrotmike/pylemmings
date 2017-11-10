# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:59:45 2015

@author: amina

This module contains the class definitions for the Stochastic subgradient descent based large margin classifiers for Multiple Instance Learning

Following classifiers are implemented here:
linclass-Linear Classifier
linclass_rank- Ranking Implementation of Linear Classifier
llclass- Non Linear Classifier
llclass_rank- Ranking Implementation of Non Linear Classifier 
"""


from scipy.sparse import *
import random
import numpy as np
from cv import *
from bag import Bag


def hloss_fun(y, scores,**kwargs):
    
    return np.max(np.vstack((1-y*scores.flatten(),np.zeros(len(scores)))),axis = 0)
#    reurn 
#    losses=[]
#    for s in scores:
#        if (1-y*s)>0:
#            losses+=[1-y*s]
#        else:
#            losses+=[0.0]
#    return losses
    
    
    
def hloss_dfun(x, y, z, **kwargs):
    if y*z<1:
        return -y*x
    else:
        return 0

    
class ClassifierBase:
    """
    This is the base class for linclass, linclass_rank, llclass and llclass_rank
    """
    CV = CV # These properties are set for all objects of this or derived class even if the base initializer is not called
    test = test
    
    def __init__(self,*args, **kwargs):
        
        if len(args) and isinstance(args[0], self.__class__):            
            self.w=args[0].w
            self.epochs=args[0].epochs
            self.Lambda=args[0].Lambda
            self.b=args[0].b
            self.lfun=args[0].lfun
            self.dlfun=args[0].dlfun
            self.bias=args[0].bias
            
        elif 'filename' in kwargs:
            self.load(kwargs['filename'])
            print self.w
            self.w=np.array(self.w)
        else:
        
            if 'epochs' in kwargs:
                self.epochs=kwargs['epochs']
            else:
                self.epochs=100
            
            if 'Lambda' in kwargs:
                self.Lambda=kwargs['Lambda']
            else:
                self.Lambda=0.00001
            if 'lfun' in kwargs:            
                self.lfun = kwargs['lfun']
                if 'dlfun' not in kwargs:
                    raise ValueError("The gradient function of the given loss must be specified")
                self.dlfun = kwargs['dlfun']
                
                
            else:
                self.lfun=hloss_fun
                self.dlfun=hloss_dfun
               
            self.w=None
            self.bias=0
            self.b=0
        
    def train(self,bags,**kwargs):
        pass
        
    def predict_bag(self,test_bag, **kwargs):
#        w=self.w
#        X=test_bag.instances
        #pred_scores=X.dot(w.T) +self.b
        
        pred_scores=self.predict_instance(test_bag.instances)
        
        pmax_i=np.argmax(pred_scores)
        output=pred_scores[pmax_i]
        return float(output)
        
    def predict_instance(self,test_example):
        
        w=self.w
        pred_score=test_example.dot(w.T) +self.b+self.bias
        return pred_score    
    def normalize_weight(self, bags):
        mn=max(bags[0].instances.dot(self.w.T) )
        mx=max(bags[0].instances.dot(self.w.T) )
        for b in bags:
#            score=max(b.instances.dot(w.T) )
            score=self.predict_bag(b)
            if score>mx:
                mx=score
            elif score<mn:
                mn=score
        self.w=(2.0*self.w)/(mx-mn)
    def add_bias(self, bags):
        for i in range(len(bags)):
            bags[i].instances=[np.append(x,1.0) for x in bags[i].instances]
            bags[i].instances=np.array(bags[i].instances)
#        for b in bags:        
#            b.instances=np.vstack((np.array(b.instances), np.ones(len(b))))
    def set_wbias(self, bags):
        mn=max(bags[0].instances.dot(self.w.T) )
        mx=max(bags[0].instances.dot(self.w.T) )
        for b in bags:
#            score=max(b.instances.dot(w.T) )
            score=self.predict_bag(b)
            if score>mx:
                mx=score
            elif score<mn:
                mn=score
        self.b=(-2.0*mn)/(mx-mn)
        
    def merge_negative_bags(self, neg_bags):
        one_neg_bag=Bag()
                
        for b in neg_bags:
            for n in range((np.shape(b.instances)[0])):
                one_neg_bag.addInstance(b.instances[n]) 
            one_neg_bag.addPeta(b.peta)
        one_neg_bag.label=-1.0
        return one_neg_bag
        
    def save(self,ofname):
        with open(ofname,'w') as fout:
            fout.write(self.toString())
    def load(self,ifname):
        with open(ifname) as fin:
           self.fromString(fin.read())         
           
    def toString(self):
        import json
        s='#Name='+str(self.__class__)
        s+='#w='+str(json.dumps(self.w.tolist()))
        s+='#b='+str(self.b)
        s+='#Epochs='+str(self.epochs)  
        s+='#Lambda='+str(self.Lambda)
        return s
        
    def fromString(self,s):    
        import json
        for token in s.split('#'):
            if token.find('w=')>=0 or token.find('W=')>=0:
                self.w=np.array(json.loads(token.split('=')[1]))
            elif token.find('b=')>=0:
                self.b=token.split('=')[1]
            elif token.find('Epochs=')>=0:
                self.epochs=float(token.split('=')[1]) 
            elif token.find('Lambda=')>=0:
                self.Lambda=float(token.split('=')[1])
                
#############################################################################################
class linReg(ClassifierBase):
    def train(self, bags, **kwargs): 
        bias=0        
        siz1=np.shape(bags[0].instances)[1]
        w=np.zeros((1,siz1)) 
        
        epochs=self.epochs
        T=(len(bags))*epochs
        lambdaa=self.Lambda 
        loss_fun=self.lfun
        d_loss_fun=self.dlfun
        for t in range(1,T):
#            if (t-1)%epochs==0:
#                random.shuffle(bags)
            
            eeta=1.0/(lambdaa*t)
            bag_chosen=bags[(t-1)%len(bags)]
            X=bag_chosen.instances
            scores=X.dot(w.T)# would need loop if use self.predict
            
            losses=loss_fun(bag_chosen.label,scores, **kwargs)
            nan=np.isnan(scores)
            if (nan):
                import pdb; pdb.set_trace()
#            idx=None
#            if bag_chosen.label>0:
#                idx=np.argmin(losses)
#            else:
#                idx=np.argmax(losses)
            idx=np.argmin(losses) 
            
            x=bag_chosen.instances[idx]
            y=bag_chosen.label
            z=scores[idx]
           
             
            
            peta=bag_chosen.peta[idx]
            w=(1-eeta)*w
            bias=(1-eeta)*bias
            
            w=w-eeta*(d_loss_fun(x,y,z, **kwargs))
            bias=bias-eeta*(d_loss_fun(np.array([1]),y,z, **kwargs))
            
            
        self.w=w
        self.bias=0
        
        if 'normalize' in kwargs and kwargs['normalize']:
            self.normalize_weight(bags)
        if 'wbias' in kwargs and kwargs['wbias']:
            self.set_wbias(bags)
            
    def predict_bag(self, test_bag, **kwargs):
        pred_scores=self.predict_instance(test_bag.instances)
        mod=np.abs(pred_scores)
        pmax_i=np.argmin(mod)
        output=pred_scores[pmax_i]
        return float(output)
        
            
        
    
#############################################################################################

class linclass(ClassifierBase):   
    """
    This class defines the stochastic gradient descent based linear large margin classifier for MIL.

    Parent Class: ClassifierBase
    
    Properties:
    epochs: No. of epochs to be run for optimization
    Lambda: The Regularization Parameter
    
    Methods:
    train(bags)
    predict_bag(bag)
    load(filename)
    save(filename)
    
    USAGE
    Class definition:
    clf=linclass() # create a classifier object with default arguments epochs=100, Lambda=0.00001
    clf=linclass(epochs=100, Lambda=0.01) # create a classifier object with customized arguments
    clf=linclass(clf1) # create a classifier object 'clf' with same properties as another object 'clf1'
    
    Training:
    clf.train(bags)
    
    Predict:
    clf.predict_bag(bag)
    
    Load Classifier:
    clf.load(filename)
    
    Save Classifier:
    clf.save(filename)
    """
    
    def train(self, bags,**kwargs):
        
        
        siz1=np.shape(bags[0].instances)[1]
        w=np.array(np.zeros(siz1))  
        w=w[np.newaxis,:]
        epochs=self.epochs
        T=(len(bags))*epochs
        lambdaa=self.Lambda 
        bias=self.bias
        for t in range(1,T):
            if (t)%epochs==0:
                random.shuffle(bags)
            #index=random.randrange(0,len(bags))
            eeta=1.0/(lambdaa*t)
            bag_chosen=bags[(t-1)%len(bags)]
            X=bag_chosen.instances
            scores=X.dot(w.T)+bias
            
            pmax_i=np.argmax(scores)
            z=scores[pmax_i]
            x=bag_chosen.instances[pmax_i]
            y=bag_chosen.label
           
             
            z=x.dot(w.T)+bias
            peta=bag_chosen.peta[pmax_i]
            
            if (y*z<1): 
               
                w=(1-eeta*lambdaa)*w+(peta*eeta*y*x)
#                if 'bias' in kwargs and kwargs['bias']:
                bias=(1-eeta*lambdaa)*bias+(peta*eeta*y)
            else:
                w=(1-eeta*lambdaa)*w
#                if 'bias' in kwargs and kwargs['bias']:
                bias=(1-eeta*lambdaa)*bias
#        import pdb; pdb.set_trace()        
        self.w=w
        self.bias=bias
        
        if 'normalize' in kwargs and kwargs['normalize']:
            self.normalize_weight(bags)
        if 'wbias' in kwargs and kwargs['wbias']:
            self.set_wbias(bags)        
        
  

class linclass_rank(ClassifierBase):
    
    """
    This class defines the stochastic gradient descent based ranking linear large margin classifier for MIL.
    
    Parent Class: ClassifierBase
    
    Properties:
    epochs: No. of epochs to be run for optimization
    Lambda: The Regularization Parameter
    
    Methods:
    train(bags)
    predict_bag(bag)
    load(filename)
    save(filename)
    
    USAGE
    Class definition:
    clf=linclass_rank() # create a classifier object with default arguments epochs=100, Lambda=0.00001
    clf=linclass_rank(epochs=100, Lambda=0.01) # create a classifier object with customized arguments
    clf=linclass_rank(clf1) # create a classifier object 'clf' with same properties as another object 'clf1'
    
    Training:
    clf.train(bags)
    
    Predict:
    clf.predict_bag(bag)
    
    Load Classifier:
    clf.load(filename)
    
    Save Classifier:
    clf.save(filename)
    """
    
    def train(self, bags, **kwargs):
        
        if 'bias' in kwargs and kwargs['bias']:
            self.add_bias(bags)    
        
        class_labels=[]
        for b in bags:
            if b.label not in class_labels:
                class_labels.append(b.label)
        #import pdb; pdb.set_trace()      
                
        
        siz1=np.shape(bags[0].instances)[1]
        
        w=np.array(np.zeros(siz1))  
        w=w[np.newaxis,:]
        epochs=self.epochs
        
        lambdaa=self.Lambda #C=1/2lambda*n
        
#        pos_bags_i, neg_bags_i=separate_bags(bags)
#        pos_bags=[]
#        neg_bags=[]
#        for ind in pos_bags_i:
#            pos_bags+=[bags[ind]]
#        for ind in neg_bags_i:
#            neg_bags+=[bags[ind]]
        
        one_neg_bag=None
        if len(class_labels)==2 and 'merge' in kwargs and kwargs['merge']:
            one_neg_bag=self.merge_negative_bags(neg_bags)
               
        sep_bags=separate_bags_multi(bags, class_labels)
        # identify most negative class
        mst_neg_ind=np.argmin(class_labels)
        mst_neg_label=class_labels[mst_neg_ind]
        most_neg_bags=sep_bags.pop(mst_neg_ind)
        #import pdb; pdb.set_trace()
        
#        T=0
#        for s in range(len(sep_bags)-1):
#            T+=len(s)
#        #T=(len(pos_bags))*epochs
#        T=T*epochs
        
        
        for e in range(epochs):
            iter_s=0.0
            for s in sep_bags:
                iter_b=0.0
                for b in s:
                    eeta=1.0/(lambdaa*(e+1)*(iter_s+1)*(iter_b+1))
                    iter_b+=1
                    bag_chosen_p=bags[b]
                    X_p=bag_chosen_p.instances
                    scores_p=X_p.dot(w.T) 
                    pmax_i=np.argmax(scores_p)
                    z_p=scores_p[pmax_i]
                    x_p=bag_chosen_p.instances[pmax_i]
                    
                    if len(class_labels)==2 and 'merge' in kwargs and kwargs['merge']:
                        bag_chosen_n=one_neg_bag
                        
                    else:
                        index_n=None
                        if (iter_s >= len(sep_bags)-1):
                            index_n=random.choice(most_neg_bags)
                        else:
                            index_n=random.choice(sep_bags)
                        bag_chosen_n=bags[index_n]
                    
                    X_n=bag_chosen_n.instances
                    scores_n=X_n.dot(w.T) 
                    nmax_i=np.argmax(scores_n)
                    z_n=scores_n[nmax_i]
                    x_n=bag_chosen_n.instances[nmax_i]
                    peta=1.0               
                    
                    w=(1-eeta*lambdaa)*w
                    
                    if((z_p)<(z_n+1)):                
                        w=w-(peta*eeta*(x_n-x_p))
                        
                iter_s+=1
        self.w=w
        if 'normalize' in kwargs and kwargs['normalize']:
            self.normalize_weight(bags)
        if 'wbias' in kwargs and kwargs['wbias']:
            self.set_wbias(bags)
                    
                        
                    
                    
                    
                
        

        
        
        
#        for t in range(1,T+1):
#            if (t)%epochs==0:
#                random.shuffle(pos_bags)
#            
#           
#            eeta=1.0/(lambdaa*t)
#            bag_chosen_p=pos_bags[(t-1)%len(pos_bags)]
#            X_p=bag_chosen_p.instances
#            scores_p=X_p.dot(w.T) 
#            pmax_i=np.argmax(scores_p)
#            z_p=scores_p[pmax_i]
#            x_p=bag_chosen_p.instances[pmax_i]
#           
#            
#            if 'merge' in kwargs and kwargs['merge']==True:
#                bag_chosen_n=one_neg_bag
#                
#                
#            else:
#                index_n=random.randrange(0,len(neg_bags))
#                
#                bag_chosen_n=neg_bags[index_n]
#           
#            X_n=bag_chosen_n.instances
#            scores_n=X_n.dot(w.T) 
#            nmax_i=np.argmax(scores_n)
#            z_n=scores_n[nmax_i]
#            x_n=bag_chosen_n.instances[nmax_i]
#            peta=1.0
#            
#            
#            
#            
#            
#            w=(1-eeta*lambdaa)*w
#            
#            if((z_p)<(z_n+1)):                
#                w=w-(peta*eeta*(x_n-x_p))
#        self.w=w
#        if 'normalize' in kwargs and kwargs['normalize']:
#            self.normalize_weight(bags)
#        if 'bias' in kwargs and kwargs['bias']:
#            self.set_bias(bags)     
        
        
    
#class milReg(ClassifierBase):
#    def train(self, bags, **kwargs):
#        epsilon=None # error bound
#        if 'epsilon' in kwargs:
#            epsilon=kwargs['epsilon']
#        else:
#            epsilon=0.00001
#        siz1=np.shape(bags[0].instances)[1]
#        w=np.array(np.zeros(siz1))  
#        w=w[np.newaxis,:]
#        epochs=self.epochs
#        T=(len(bags))*epochs
#        lambdaa=self.Lambda 
#        
#        for t in range(1,T):
#            if (t)%epochs==0:
#                random.shuffle(bags)
#            #index=random.randrange(0,len(bags))
#            eeta=1.0/(lambdaa*t)
#            bag_chosen=bags[(t-1)%len(bags)]
#            X=bag_chosen.instances
#            label_vec=np.array([bag_chosen.label]*len(bag_chosen.instances))
#            label_vec=label_vec[:,np.newaxis]
#            scores=label_vec-X.dot(w.T)
#            #import pdb; pdb.set_trace()
#            pmax_i=np.argmin(scores)
#            z=scores[pmax_i]
#            x=bag_chosen.instances[pmax_i]
#            y=bag_chosen.label
#           
#             
#            z=x.dot(w.T)
#            peta=bag_chosen.peta[pmax_i]
#            
#            if ((z-y)>epsilon): 
#               
#                w=(1-peta*eeta*lambdaa)*w-(eeta*x)
#            elif (z-y)<epsilon:
#                w=(1-peta*eeta*lambdaa)*w+(eeta*x)
#            else:
#                w=(1-eeta*lambdaa)*w
#        
#        
#        if 'normalize' in kwargs and kwargs['normalize']<>False:
#            mn=max(bags[0].instances.dot(w.T) )
#            mx=max(bags[0].instances.dot(w.T) )
#            for b in bags:
#                score=max(b.instances.dot(w.T) )
#                if score>mx:
#                    mx=score
#                elif score<mn:
#                    mn=score
#            w=(2.0*w)/(mx-mn)
#        if 'bias' in kwargs and kwargs['bias']<>False:
#            self.b=(-2.0*mn)/(mx-mn)
#            
#        self.w=w
#    def predict_bag(self,test_bag):
#        w=self.w
#        X=test_bag.instances
#        label_vec=np.array([test_bag.label]*len(test_bag.instances))
#        label_vec=label_vec[:,np.newaxis]
#        pred_scores=label_vec-X.dot(w.T) +self.b
#        pmax_i=np.argmin(pred_scores)
#        output=pred_scores[pmax_i]
#        return float(output)
#            
#            
        
    
  

   
class llclass(ClassifierBase):  
    """
    This class defines the stochastic gradient descent based non(/locally)-linear large margin classifier for MIL.
    
    Parent Class: ClassifierBase
    
    Properties:
    epochs: No. of epochs to be run for optimization
    Lambda: The Regularization Parameter
    
    Methods:
    train(bags)
    predict_bag(bag)
    load(filename)
    save(filename)
    
    USAGE
    Class definition:
    clf=llclass() # create a classifier object with default arguments epochs=100, Lambda=0.00001
    clf=llclass(epochs=100, Lambda=0.01) # create a classifier object with customized arguments
    clf=llclass(clf1) # create a classifier object 'clf' with same properties as another object 'clf1'
    
    Training:
    clf.train(bags)
    
    Predict:
    clf.predict_bag(bag)
    
    Load Classifier:
    clf.load(filename)
    
    Save Classifier:
    clf.save(filename)
    """
    
    def normalize_weight(self, bags):
        W=self.w
        X=bags[0].instances
        G=bags[0].gammas
        
        if issparse(X) and issparse(G):   
            mn = max(np.ravel(np.sum(X.multiply(G.dot(W)),axis=1)))
            mx=max(np.ravel(np.sum(X.multiply(G.dot(W)),axis=1)))
        else:               
            mx= max(np.sum(G.dot(W)*(X),axis=1))
            mn= max(np.sum(G.dot(W)*(X),axis=1))
        
        for b in bags:
            X=b.instances
            G=b.gammas
            score=0
            if issparse(X) and issparse(G):
                score=max(np.ravel(np.sum(X.multiply(G.dot(W)),axis=1)))
            else:
                score=max(np.sum(G.dot(W)*(X),axis=1))
            if score>mx:
                mx=score
            elif score<mn:
                mn=score
        W=(2.0*W)/(mx-mn)
        self.w=W
    def set_wbias(self, bags):
        W=self.w
        X=bags[0].instances
        G=bags[0].gammas
        
        if issparse(X) and issparse(G):   
            mn = max(np.ravel(np.sum(X.multiply(G.dot(W)),axis=1)))
            mx=max(np.ravel(np.sum(X.multiply(G.dot(W)),axis=1)))
        else:               
            mx= max(np.sum(G.dot(W)*(X),axis=1))
            mn= max(np.sum(G.dot(W)*(X),axis=1))
        
        for b in bags:
            X=b.instances
            G=b.gammas
            score=0
            if issparse(X) and issparse(G):
                score=max(np.ravel(np.sum(X.multiply(G.dot(W)),axis=1)))
            else:
                score=max(np.sum(G.dot(W)*(X),axis=1))
            if score>mx:
                mx=score
            elif score<mn:
                mn=score
        self.b=(-2.0*mn)/(mx-mn)  
    def train(self, bags, **kwargs):     
        
#        import pdb; pdb.set_trace()
        bias=np.zeros([1,np.shape(bags[0].gammas)[1]])
        
            
                    
        W=np.zeros([np.shape(bags[0].gammas)[1], np.shape(bags[0].instances)[1]])
        epochs=self.epochs
        T=epochs*len(bags)
        lambdaa=self.Lambda
        for t in range (1,T):      
            if (t)%epochs==0:
                random.shuffle(bags)
            eeta=1.0/(lambdaa*t)            
            bag_chosen=bags[(t-1)%len(bags)]           
            X=bag_chosen.instances
            G=bag_chosen.gammas
            scores=None
            if issparse(X) and issparse(G):   
                scores = np.ravel(np.sum(X.multiply(G.dot(W))+G.dot(bias.T),axis=1))            
            else:               
                scores= np.sum(G.dot(W)*(X)+G.dot(bias.T) ,axis=1)           
            rep_ex_index=np.argmax(scores)            
            rep_score=scores[rep_ex_index]            
            rep_ex=bag_chosen.instances[rep_ex_index]
            rep_gamma=bag_chosen.gammas[rep_ex_index]    
            peta=bag_chosen.peta[rep_ex_index]
            W=(1-eeta*lambdaa)*W
#            if 'bias' in kwargs and kwargs['bias']:
            bias=(1-eeta*lambdaa)*bias
            if not(issparse(X)):
                rep_ex=rep_ex[ np.newaxis,:]                
            if not(issparse(G)):
                rep_gamma=rep_gamma[ np.newaxis,:]
            label=bag_chosen.label 
#            import pdb; pdb.set_trace()
            if (1-label*rep_score)>=0:  
                
                W=W+peta*eeta*label*rep_gamma.T.dot(rep_ex) 
#                if 'bias' in kwargs and kwargs['bias']:
                bias=bias+ eeta*label*rep_gamma  
        self.w=W 
        self.bias=bias
                
        if 'normalize' in kwargs and kwargs['normalize']:
            self.normalize_weight(bags)
        if 'wbias' in kwargs and kwargs['wbias']:
            self.set_wbias(bags)        
        
    def predict_bag(self, bag, **kwargs):
#        if 'bias' in kwargs and kwargs['bias']:
#            self.add_bias([bag])
        scores=np.zeros(np.shape(bag.instances)[0])
        X=bag.instances
        G=bag.gammas
       
        if issparse(X) and issparse(G):    
                
             scores = np.ravel(np.sum(X.multiply(G.dot(self.w))+G.dot(self.bias.T),axis=1))+self.b
#            scores = np.ravel(np.sum(X.multiply(G.dot(self.w)),axis=1))+self.b
        else:
                
             scores= np.sum(G.dot(self.w)*(X)+G.dot(self.bias.T),axis=1)+self.b
#            scores= np.sum(G.dot(self.w)*(X),axis=1)+self.b
        scores=list(scores)
        rep_score=max(scores)
        return rep_score

    def predict_instance(self, instance):
        
        gamma=instance.gammas
        H=np.dot(self.w.T,instance.features)
        H=np.dot(gamma,H)
        
        return H     
       



class llclass_rank(llclass):
    """
    This class defines the stochastic gradient descent based ranking non(/locally)-linear large margin classifier for MIL.
    
    Parent Class: ClassifierBase
    
    Properties:
    epochs: No. of epochs to be run for optimization
    Lambda: The Regularization Parameter
    
    Methods:
    train(bags)
    predict_bag(bag)
    load(filename)
    save(filename)
    
    USAGE
    Class definition:
    clf=linclass_rank() # create a classifier object with default arguments epochs=100, Lambda=0.00001
    clf=linclass_rank(epochs=100, Lambda=0.01) # create a classifier object with customized arguments
    clf=linclass_rank(clf1) # create a classifier object 'clf' with same properties as another object 'clf1'
    
    Training:
    clf.train(bags)
    
    Predict:
    clf.predict_bag(bag)
    
    Load Classifier:
    clf.load(filename)
    
    Save Classifier:
    clf.save(filename)
    """
    
        
    def train(self, bags, **kwargs):
#        if 'bias' in kwargs and kwargs['bias']:
#            self.add_bias(bags)
#        import pdb; pdb.set_trace()
        bias=np.zeros([1,np.shape(bags[0].gammas)[1]])
        class_labels=[]
        for b in bags:
            if b.label not in class_labels:
                class_labels.append(b.label)
        W=np.zeros([np.shape(bags[0].gammas)[1], np.shape(bags[0].instances)[1]])
        epochs=self.epochs       
        lambdaa=self.Lambda
        pos_bags_i, neg_bags_i=separate_bags(bags)
        T=(len(pos_bags_i))*epochs
        pos_bags=[]
        neg_bags=[]
        for ind in pos_bags_i:
            pos_bags+=[bags[ind]]
        for ind in neg_bags_i:
            neg_bags+=[bags[ind]]   
            
            
        if len(class_labels)==2 and 'merge' in kwargs and kwargs['merge']:
            one_neg_bag=self.merge_negative_bags(neg_bags) 
        
        
        sep_bags=separate_bags_multi(bags, class_labels)
        # identify most negative class
        mst_neg_ind=np.argmin(class_labels)
        mst_neg_label=class_labels[mst_neg_ind]
        most_neg_bags=sep_bags.pop(mst_neg_ind)
        #import pdb; pdb.set_trace()
        
#        T=0
#        for s in range(len(sep_bags)-1):
#            T+=len(s)
#        #T=(len(pos_bags))*epochs
#        T=T*epochs
        
        
        for e in range(epochs):
            iter_s=0.0
            for s in sep_bags:
                iter_b=0.0
                random.shuffle(s)
                for b in s:
                    eeta=1.0/(lambdaa*(e+1)*(iter_s+1)*(iter_b+1))
                    iter_b+=1
                    pos_bag_chosen=bags[b]
                    X_p=pos_bag_chosen.instances
                    G_p=pos_bag_chosen.gammas
                    scores_pos=None
                    if issparse(X_p) and issparse(G_p):   
                        scores_pos = np.ravel(np.sum(X_p.multiply(G_p.dot(W)+G_p.dot(bias.T)),axis=1))
                    else:
                        scores_pos= np.sum(G_p.dot(W)*(X_p)+G_p.dot(bias.T),axis=1)
        
                    rep_ex_index_pos=np.argmax(scores_pos)
                    rep_score_pos=scores_pos[rep_ex_index_pos]
                    rep_ex_pos=pos_bag_chosen.instances[rep_ex_index_pos]
                    rep_gamma_pos=pos_bag_chosen.gammas[rep_ex_index_pos]
                    
                    if not(issparse(X_p)):
                        rep_ex_pos=rep_ex_pos[np.newaxis,:]
                    if not(issparse(G_p)):
                        rep_gamma_pos=rep_gamma_pos[np.newaxis,:]
                    
                    if len(class_labels)==2 and 'merge' in kwargs and kwargs['merge']:
                        neg_bag_chosen=one_neg_bag
                        
                    else:
                        index_n=None
                        if (iter_s >= len(sep_bags)-1):
                            index_n=random.choice(most_neg_bags)
                        else:
                            index_n=random.choice(sep_bags)
                        neg_bag_chosen=bags[index_n]
                    G_n=neg_bag_chosen.gammas
                    X_n=neg_bag_chosen.instances
                    scores_neg=None
                    if issparse(X_n) and issparse(G_n):   
                        scores_neg = np.ravel(np.sum(X_n.multiply(G_n.dot(W)+G_n.dot(bias.T)),axis=1))
                    else:
                        scores_neg= np.sum(G_n.dot(W)*(X_n)+G_n.dot(bias.T),axis=1)
                    
        
                    rep_ex_index_neg=np.argmax(scores_neg)
                    rep_score_neg=scores_neg[rep_ex_index_neg]
                    
                    
                    rep_ex_neg=neg_bag_chosen.instances[rep_ex_index_neg]
                    rep_gamma_neg=neg_bag_chosen.gammas[rep_ex_index_neg]
                    W=(1-eeta*lambdaa)*W
                    bias=(1-eeta*lambdaa)*bias
                    if not(issparse(X_n)):
                        rep_ex_neg=rep_ex_neg[np.newaxis,:]
                    if not(issparse(G_n)):
                        rep_gamma_neg=rep_gamma_neg[np.newaxis,:]
                   
                    peta=neg_bag_chosen.peta[rep_ex_index_neg]
                    if (rep_score_pos-rep_score_neg)<1:
                                      
                        W=W-(peta*eeta*(rep_ex_neg.T.dot( rep_gamma_neg))-(rep_ex_pos.T.dot( rep_gamma_pos))).T
#                        if 'bias' in kwargs and kwargs ['bias']:
                        bias=bias+eeta*(rep_gamma_pos-rep_gamma_neg)

                   
                        
                iter_s+=1
        self.bias=bias
        self.w=W
        if 'normalize' in kwargs and kwargs['normalize']:
            self.normalize_weight(bags)
        if 'wbias' in kwargs and kwargs['wbias']:
            self.set_wbias(bags)
        
        
        
        
        ################################################################3
#        for t in range (1,T):
#            eeta=1.0/(lambdaa*t)
#            pos_bag_chosen=pos_bags[(t-1)%len(pos_bags)]
#            G_p=pos_bag_chosen.gammas
#            X_p=pos_bag_chosen.instances
#            scores_pos=None
#            if issparse(X_p) and issparse(G_p):   
#                scores_pos = np.ravel(np.sum(X_p.multiply(G_p.dot(W)),axis=1))
#            else:
#                scores_pos= np.sum(G_p.dot(W)*(X_p),axis=1)
#
#            rep_ex_index_pos=np.argmax(scores_pos)
#            rep_score_pos=scores_pos[rep_ex_index_pos]
#            
#            
#            rep_ex_pos=pos_bag_chosen.instances[rep_ex_index_pos]
#            rep_gamma_pos=pos_bag_chosen.gammas[rep_ex_index_pos]
#            
#            if not(issparse(X_p)):
#                rep_ex_pos=rep_ex_pos[np.newaxis,:]
#            if not(issparse(G_p)):
#                rep_gamma_pos=rep_gamma_pos[np.newaxis,:]
#            
#            if 'merge' in kwargs and kwargs['merge']==True:
#                neg_bag_chosen=one_neg_bag
#            else:
#                rand_ind_neg=random.randrange(0,len(neg_bags))
#                neg_bag_chosen=neg_bags[rand_ind_neg]
#            
#            G_n=neg_bag_chosen.gammas
#            X_n=neg_bag_chosen.instances
#            scores_neg=None
#            if issparse(X_n) and issparse(G_n):   
#                scores_neg = np.ravel(np.sum(X_n.multiply(G_n.dot(W)),axis=1))
#            else:
#                scores_neg= np.sum(G_n.dot(W)*(X_n),axis=1)
#            
#
#            rep_ex_index_neg=np.argmax(scores_neg)
#            rep_score_neg=scores_neg[rep_ex_index_neg]
#            
#            
#            rep_ex_neg=neg_bag_chosen.instances[rep_ex_index_neg]
#            rep_gamma_neg=neg_bag_chosen.gammas[rep_ex_index_neg]
#            W=(1-eeta*lambdaa)*W
#            if not(issparse(X_n)):
#                rep_ex_neg=rep_ex_neg[np.newaxis,:]
#            if not(issparse(G_n)):
#                rep_gamma_neg=rep_gamma_neg[np.newaxis,:]
#           
#            peta=neg_bag_chosen.peta[rep_ex_index_neg]
#            if (rep_score_pos-rep_score_neg)<1:
#                              
#                W=W-(peta*eeta*(rep_ex_neg.T.dot( rep_gamma_neg))-(rep_ex_pos.T.dot( rep_gamma_pos))).T
#        self.w=W
#        if 'normalize' in kwargs and kwargs['normalize']:
#            self.normalize_weight(bags)
#        if 'bias' in kwargs and kwargs['bias']:
#            self.set_bias(bags)
#        