#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: -----, 2020
"""
# pylint: disable=C0103

import matplotlib.pyplot as plt
import numpy as np

class FPS:
    # https://stackoverflow.com/questions/2350072/custom-data-types-in-numpy-arrays
    def __init__(self, explb, base, prec):
        self.explb = explb # k, lower bound for exponent
        self.base = base # b
        self.prec = prec # t, precision
        
        self.array = None # contains tuples (m, e) or 0 to denote values
        self.shape = None
        self.asize = 0
        
    def get_fpsparams(self):
        return self.explb, self.base, self.prec
    
    def __str__(self):
        return 'F_%d(%d,p=%d)...'%(self.explb, self.base, self.prec)

    def visualize(self):
        explb, base, p = self.get_fpsparams()
        
        print('Visualizing', self)
        denom = base ** (p-1)
        numbers = []
        bars = [base ** (-i) for i in range(1, explb+1)]
    
        plt.plot(bars, np.ones(len(bars)),
                     linestyle='None', marker='|', c='black', ms = 25)
        plt.plot([0,1], np.ones(2),
                     linestyle='None', marker='|', c='black', ms = 30)
    
        texts = []
        fps = []

        for e in range(0, explb+1):
            fac = base ** (-e-p)
            texts.append(r'\frac{1}{%d}'%fac)
            numbers = [m * fac for m in range(denom, base * denom)]
            plt.plot(numbers, np.ones(len(numbers)),
                     linestyle='None', marker='o', ms=5, label=r'e=-%d'%e)
            print(numbers, 'for exp=%d'%e)
            fps += numbers
        plt.plot([0], np.ones(1),
                     linestyle='None', marker='s', ms=5, label=r'e=0')
        
        print('The FPS contains %d+1 numbers'%len(fps))
        
        fsize = 20
        plt.text(-.02, .965, '0', {'size':fsize})
        plt.text(1-.02, .965, '1', {'size':fsize})
        plt.hlines(1,0,1)  # Draw a horizontal line
        plt.xlim(0,1)
        plt.ylim(0.9,1.1)
        plt.axis('off')
#        plt.legend(prop={'size': 12}, loc=9, ncol=4, borderaxespad=0.) # upper center
        
        plt.savefig('F_%db%dp%d.pdf'%(explb, base, p), bbox_inches='tight')
        plt.show()
    
    def round_down(self, array):
        """store array and return double precision approximation"""
        self.store_array(array)
        return self.get_nparray()
    
    def store_array(self, array):
        """round down elements in array to nearest element from FPS 
        specified by the tuple fps"""
        assert (0 <= array).all() and (array <=1).all()
        
        _, base, p = self.get_fpsparams()
    
        self.shape = array.shape
        self.asize = np.prod(self.shape)
        max_mantissa = base ** p - 1

        fpsarray = np.empty(self.asize, dtype=object)
        for i, a in enumerate(np.nditer(array)):
            if a == 0:
                fpsarray[i] = 0
            elif a == 1:
                fpsarray[i] = max_mantissa, 0
            else:
                tmp = np.log(a) / np.log(base)
                e = np.ceil(np.log(a) / np.log(base))
                if tmp == e and e < 0: # a is an exponential of base
                    e += 1
                if e < -self.explb:
                    fpsarray[i] = 0
                    continue
                m = np.floor(a / base**(e - p))
                
#                print(a, tmp, 'exp', e, 'mant', m)
                fpsarray[i] = int(m), int(e)

        self.fpsarray = fpsarray
#        print(self.fpsarray)
#        return self.get_nparray()
    
    def get_nparray(self, fpsarray=None):
        """return numpy array"""
        if fpsarray is None:
            assert not self.fpsarray is None
            fpsarray = self.fpsarray

        _, base, prec = self.get_fpsparams()            
        nparray = np.empty(self.asize, dtype=np.float64)
        for i, a in enumerate(fpsarray):
            if a in [0, 1]:
                nparray[i] = a
            else:
                m, e = a
                nparray[i] = m * base ** (e - prec)
        
        return nparray.reshape(self.shape)


    def get_ub(self):
        """find smallest greater elements from FPS, compared to the ones in array
        if no greater element exists, use 1
        FPS is specified by the tuple fps"""
        assert not self.fpsarray is None
        
        e, base, prec = self.get_fpsparams()
        ub_array = np.empty(self.asize, dtype=object)

        ub_mantissa = base ** prec - 2
#        ufl = base ** (self.explb - 1)
        ufl = base ** (prec - 1), -self.explb
#        print('ufl', ufl)
        for i, a in enumerate(self.fpsarray):
            if a == 0:
                ub_array[i] = ufl
            else:
                m, e = a
#                print(m, e, ub_mantissa)
                if m <= ub_mantissa:
                    ub_array[i] = m + 1, e
                else:
                    ub_array[i] = base ** (prec - 1), e + 1

        return self.get_nparray(fpsarray=ub_array)
    
    
    
    
if __name__ == '__main__':
    
    fps = FPS(1, 4, 2)
    fps = FPS(1, 2, 2)
#    fps = FPS(0, 2, 2)
#    fps = FPS(5, 2, 3)
    
    fps.visualize()
    
    a = np.array([0, 0.2, 0.3, 0.5, .7, 0.8, 1])
    ra = fps.round_down(a)
    print('exact:  ', a)
    print('rounded:', ra)
    
    uba = fps.get_ub()
    
    print('ubarray:', uba)

   