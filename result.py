# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 03:39:07 2015

@author: amina
"""
from bisect import bisect_left # for binary search
import numpy as np
import matplotlib.pyplot as plt
class Result:
    def __init__(self):
        self.prediction_scores=[]
        self.auc_scores=[]
        self.accuracies=[]
        self.times=[]
        self.mean_auc=None
        self.mean_accuracy=None
        self.mean_time=None
        self.true_labels=[]
        self.cv_time=None
        self.tpr=[]
        self.fpr=[]
        self.thresholds=[]
    def __str__(self):
#        print "The Decision scores for the", len(self.accuracies)," folds:\n", self.prediction_scores
#        print "The AUC scores for the", len(self.accuracies)," folds:\n",self.auc_scores,"%"
#        print "Average AUC score=", self.mean_auc,"%"
#        print "The Accuracies for the", len(self.accuracies)," folds:\n",self.auc_scores,"%"
#        print "Average AUC score=", self.mean_auc,"%"
        return "The Decision scores for the "+str(len(self.accuracies))+\
        " folds:\n"+str( self.prediction_scores)+"\nThe AUC scores(%) for the"+\
        str(len(self.accuracies))+" folds:\n"+str(self.auc_scores)+\
        "\nAverage AUC score="+str(self.mean_auc)+"%"+ \
        "\nThe Accuracies(%) for the "+str(len(self.accuracies))+\
        " folds:\n"+str(self.accuracies)+ "\nAverage Accuracy="+\
        str(self.mean_accuracy)+"%"+\
        "\nTime taken (seconds) for each fold: \n"+str(self.times)+\
        "\nAverage time:"+str(self.mean_time)+\
        "\nTime taken (seconds) for complete cross validation run:"+str(self.cv_time)

    def save_result(self, fil):
        fil.write( "The Decision scores for the "+str(len(self.accuracies))+\
        " folds:\n"+str( self.prediction_scores)+"\nThe AUC scores(%) for the"+\
        str(len(self.accuracies))+" folds:\n"+str(self.auc_scores)+\
        "\nAverage AUC score="+str(self.mean_auc)+"%"+ \
        "\nThe Accuracies(%) for the "+str(len(self.accuracies))+\
        " folds:\n"+str(self.accuracies)+ "\nAverage Accuracy="+\
        str(self.mean_accuracy)+"%"+\
        "\nTime taken (seconds) for each fold: \n"+str(self.times)+\
        "\nAverage time:"+str(self.mean_time)+\
        "\nTime taken (seconds) for complete cross validation run:"+str(self.cv_time))
    
    
    def plot_roc(self):
        for i in range(len(self.tpr)):
            plt.plot(self.fpr[i], self.tpr[i])
            
        ax=plt.gca()
        ax.set_ylim([0.0,1.05])
        plt.show()
        
        
    def vertical_average_roc(self):
        folds_1=[]
        for i in range(len(self.accuracies)):
            folds_1+=[(self.prediction_scores[i],self.true_labels[i])]
        fpr, tpr, auc=roc_VA(folds_1)
        plt.plot(fpr, tpr)
       
        ax=plt.gca()
        ax.set_ylim([0.0,1.05])
        
        plt.show()
        
        ################## copied from pyml##############################
    
def roc( dvals, labels, rocN=None, normalize=True ) :
    """
    Compute ROC curve coordinates and area

    - `dvals`  - a list with the decision values of the classifier
    - `labels` - list with class labels, \in {0, 1} 

    returns (FP coordinates, TP coordinates, AUC )
    """
    if rocN is not None and rocN < 1 :
        rocN = int(rocN * np.sum(np.not_equal(labels, 1)))

    TP = 0.0  # current number of true positives
    FP = 0.0  # current number of false positives
    
    fpc = [ 0.0 ]  # fp coordinates
    tpc = [ 0.0 ]  # tp coordinates
    dv_prev = -np.inf # previous decision value
    TP_prev = 0.0
    FP_prev = 0.0
    area = 0.0

    num_pos = labels.count( 1 )  # number of pos labels
    num_neg = labels.count( 0 ) # number of neg labels
    
    if num_pos == 0 or num_pos == len(labels) :
        raise ValueError, "There must be at least one example from each class"

    # sort decision values from highest to lowest
    indices = np.argsort( dvals )[ ::-1 ]
    
    idx_prev = -1
    for idx in indices:
        # increment associated TP/FP count
        if labels[ idx ] == 1:
            TP += 1.
        else:
            FP += 1.
            if rocN is not None and FP == rocN : 
                break
        # Average points with common decision values
        # by not adding a coordinate until all
        # have been processed
        if dvals[ idx ] != dv_prev:
            if len(fpc) > 0 and FP == fpc[-1] :
                tpc[-1] = TP
            else :
                fpc.append( FP  )
                tpc.append( TP  )
            dv_prev = dvals[ idx ]
            area += _trap_area( ( FP_prev, TP_prev ), ( FP, TP ) )
            FP_prev = FP
            TP_prev = TP
            idx_prev = idx

    # Last few decision values were all the same,
    # so must append final points and area
    if idx_prev != indices[-1] :
        tpc.append(num_pos)
        fpc.append(num_neg)
        area += _trap_area( ( FP, TP ), ( FP_prev, TP_prev ) )

    #area += _trap_area( ( FP, TP ), ( FP_prev, TP_prev ) )
    #fpc.append( FP  )
    #tpc.append( TP )

    if normalize :
        fpc = [ float( x ) / FP for x in fpc ]
        if TP > 0:
            tpc = [ float( x ) / TP for x in tpc ]
        if area > 0:
            area /= ( num_pos * FP )

    return fpc, tpc, area



    
    
    
def roc_VA( folds, rocN=None, n_samps=100 ):
    """
    Compute ROC curve using vertical averaging

    `folds` - list of ( labels, dvals ) pairs where labels
              is a list of class labels and dvals are
              the decision values of the classifier
    """
    # return variables
    invl = 1.0 / n_samps # interval to sample FPR
    FPRs = np.arange( 0, (1+invl), invl )
    TPRs = [ ] # will contain assoc TPR avgs for FPRs
    # folds must be listified
    assert type( folds ) == type( [ ] )
    rocs = [ ] # list of roc tuples ( [FPR,TPR] ) for folds
    areas = [ ] # individual AUCs for each fold

    # calculate individual ROC curves for each fold
    for dvals,labels in folds:
        fpc, tpc, area = roc( dvals, labels, rocN )
        rocs.append( (fpc, tpc) )
        areas.append( area )

    for fpr in FPRs:
        # accumulate TPRs for current fpr over all folds
        tpr_folds = [ ] 
        # fix FPR and accumulate (interpolated) TPRs
        for fpc, tpc in rocs:
            tpr_folds.append( _tpr_for_fpr( fpc, tpc, fpr ))
        # average tprs and append
        TPRs.append( np.mean( tpr_folds ) )
    
    return FPRs, np.array( TPRs ), np.mean( areas )


def _tpr_for_fpr( fpc, tpc, fpr ):
    """
    Returns the (estimated) tpr for the given fpr for 
    the given false positive/true positive coordinates
    from an ROC curve.

    `fpc` - False positive coordinates from ROC curve
    `tpc` - True positive coordinates from ROC curve
    """
    
    # take advantage of monotonic property of ROC curves
    # and search for fpr in O( log n ) time
    idx = bisect_left( fpc, fpr, 0, len(fpc)-1 )
    
    # if exact match, then return
    if fpc[ idx ] == fpr:
        return tpc[ idx ]

    else:
        # check if idx is last index of fpc
        #if idx == len( fpc ) - 1:
        #    return tpc[ idx ]

        # check if the neighboring fprs are identical
        #elif fpc[ idx ] == fpc[ idx + 1 ]:
            # return the average of the tprs 
        #    return ( tpc[ idx ] + tpc[ idx+1 ] ) / 2.0
        # otherwise, interpolate the tpr
        return _interpolate( ( fpc[ idx-1 ], tpc[ idx-1 ] ),
                             ( fpc[ idx ], tpc[ idx ] ),
                             fpr )

def _interpolate( p1, p2, x ):
    """
    Interpolate the value of f( x ).

    `p1` - 1st interpolation point (x1, y1)
    `p1` - 2nd interpolation point (x2, y2)
    `x`  - the value to interpolate
    """
    return p1[ 1 ] + _slope( p1, p2 ) * ( x - p1[ 0 ] )

def _trap_area( p1, p2 ):
    """
    Calculate the area of the trapezoid defined by points
    p1 and p2
    
    `p1` - left side of the trapezoid
    `p2` - right side of the trapezoid
    """
    base = abs( p2[ 0 ] - p1[ 0 ] )
    avg_ht = ( p1[ 1 ] + p2[ 1 ] ) / 2.0

    return base * avg_ht

def _slope( p1, p2 ):
    """
    Calculates the slope of the line defined by
    points p1 and p2
    """
    delta_x = p2[ 0 ] - p1[ 0 ]
    delta_y = p2[ 1 ] - p1[ 1 ]
    
    # if infinite slope, scream
    if delta_x == 0: raise( "Infinite slope" )

    return float( delta_y ) / delta_x
