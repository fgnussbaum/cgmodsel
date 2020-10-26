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

    def visualize(self, plot_bound=False):
        explb, base, p = self.get_fpsparams()


        print('Visualizing', self)
        denom = base ** (p-1)
        numbers = []
        bars = [base ** (-i) for i in range(1, explb+2)]
        print(bars, explb)
        ms = 50

        fig = plt.figure(figsize=(15,2))
#        fig = plt.figure(figsize=(15,.5))

        if plot_bound or 1:
            plt.ylim(-.03,.4)
            nn = 1000
#            xs = np.array(list(range(int(nn/4), nn)))
            xs = np.array(list(range(nn)))
            ts = [self.round_down(np.array([i/nn])) for i in xs]
            ts2 = []
            for i in xs:
                xl = self.store_array(np.array([i/nn]))[0]
                try:
                    tmp =xl[1]
                except:
                    tmp = xl
                ts2.append(base ** (tmp-p))
            ts = np.array(ts).flatten()
#            print(xs.shape, ts.shape)
            ts = ts * base ** (1 - p)
#            print(ts)
#            ts2 = 250 * [0,] + 250 * [.125, ] + 500 * [.25,]
            ts2 = np.array(ts2).flatten()
            plt.plot(xs/nn, ts2, linestyle='--', label='Lemma 2')
#            plt.plot(xs/nn, ts, linestyle='-', label='Lemma 1')
        else:
            plt.plot(bars, np.zeros(len(bars)),
                         linestyle='None', marker='|', c='black', ms = .6*ms)
            plt.plot([0,1], np.zeros(2),
                         linestyle='None', marker='|', c='black', ms = ms)
        
        texts = []
        fps = []

        for e in range(0, explb+1):
            fac = base ** (-e-p)
            texts.append(r'\frac{1}{%d}'%fac)
            numbers = [m * fac for m in range(denom, base * denom)]
            color = 'C%d'%(e % 10)
            plt.plot(numbers, np.zeros(len(numbers)),
                     linestyle='None', marker='o', ms=8, c=color, label=r'e=-%d'%e)
            print(numbers, 'for exp=%d'%e)
            fps += numbers
        plt.plot([0], np.zeros(1),
                     linestyle='None', marker='o', ms=8, c='black', label=r'e=0')
        
        print('The FPS contains %d+1 numbers'%len(fps))
        
        fsize = 20

#        plt.text(-.008, .75-1, '0', {'size':fsize})
#        plt.text(1-.008, .75-1, '1', {'size':fsize})
        plt.hlines(0,0,1)  # Draw a horizontal line
        plt.xlim(-.01,1.01)
#        plt.ylim(-0.4,0.4)
#        plt.axis('off')
#        plt.legend(prop={'size': 12}, loc=9, ncol=4, borderaxespad=0.) # upper center
        


        
        plt.savefig('F_%db%dp%d.png'%(explb, base, p), bbox_inches='tight')
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
        return self.fpsarray
    
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
    

    fps = FPS(2, 2, 2)
#    fps = FPS(0, 2, 2)
#    fps = FPS(1, 4, 2)
#    fps = FPS(5, 2, 3)
    
    fps.visualize()
    
#    a = np.array([0, 0.2, 0.3, 0.5, .7, 0.8, 1])
#    ra = fps.round_down(a)
#    print('exact:  ', a)
#    print('rounded:', ra)
#    
#    uba = fps.get_ub()
#    
#    print('ubarray:', uba)

   