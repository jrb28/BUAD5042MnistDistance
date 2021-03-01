# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:23:47 2021

@author: jrbrad
"""

import numpy as np
import time
#import multiprocessing as mp
#from numba import njit
import math

if __name__ == '__main__':
    p = np.load('data/mnist1_1000.npy').astype(np.float32)
    q = np.load('data/mnist2_1000.npy').astype(np.float32)
    assert p.shape[0] == q.shape[0]
    assert p.shape[1] ==784
    assert q.shape[1] == 784
    
    n = p.shape[0]
    pixels = p.shape[1]
    
    start = time.time()
    result = np.zeros((n,n)).astype(np.float32)
    for i in range(n):
        for j in range(n):
            sum_sq = 0.0
            for k in range(pixels):
                sum_sq += (q[i][k] - p[j][k])**2
            result[i][j] = math.sqrt(sum_sq)
    
            
        print('Avg. time per loop: %s sec.' % (float(time.time() - start)/float(i+1)))        
    print('Exec. time: %s for %sx%s' % (str(float(time.time() - start)), str(n), str(n)))