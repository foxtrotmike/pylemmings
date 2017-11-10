# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:52:33 2015

@author: Afsar
"""
import re
import numpy as np
from bag import *


fname = 'LJ-160.166.1'

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
            b.instances =  X.reshape((ni,d))# X.reshape((d,ni)).T # (Amina, look here!)
            b.label = l
            b.id = molid
            B.append(b)
            continue
            
        if rmode == 'fea':
            X.append(float(lns))
        