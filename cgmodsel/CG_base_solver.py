# Copyright (c) 2019 Frank Nussbaum (frank.nussbaum@uni-jena.de)
"""
@author: Frank Nussbaum

base class for CG models and solvers
"""

import numpy as np

from cgmodsel.models.model_base import get_modeltype

######### base class for all model solvers ########################

class CG_base_solver:
    def __init__(self, meta, useweights, reduced_levels = False):
        """must call method drop_data after initialization"""

        self.useweights = useweights # determines usage of sparse norm calibration scheme
        
        self.D = None # discrete data, dropped later
        self.Y = None # continuous data, dropped later
        
        self.currentsolution = None

        self.reduced_levels = reduced_levels
        # if True, parameters set to zero due to identifiability constraints are removed
        # if False, these parameters are included but are box-constrained to [0,0]
        
        if reduced_levels:
            self.sizes = [size - 1 for size in meta['sizes']] # do not modify meta
        else:
            self.sizes = meta['sizes']

        self.dg =  meta['dg']# number of Gaussian variables
        self.dc = len(self.sizes) # number of discrete variables

        self.Lsum = np.cumsum([0] + self.sizes) # cumulative # of levels
        self.Ltot = self.Lsum[-1] # total number of discrete levels (-dc if reduced_levels)
        
        self.d = self.Ltot + self.dg # size of one axis of the interaction parameter matrix
        
        # variables that need to be overwritten by derived classes
        self.shapes = []
        self.totalnumberofparams = -1

        self.name = 'base'
        
        self.modeltype = get_modeltype(self.dc, self.dg, self.sizes)

    def drop_data(self, discretedata, continuousdata):
        """drop data, derived classes may perform additional computations"""
#        print(self.dg, continuousdata.shape[1])
        assert self.dg == continuousdata.shape[1]
#        print(self.Ltot, discretedata.shape[1])
        assert self.Ltot == discretedata.shape[1]

        self.D = discretedata
        self.Y = continuousdata
        self.n = continuousdata.shape[0]

        # here is a prescaling factor for the sparse norm regularization
        # from the original of Chandrasekaran (it is motivated by consistency analysis):
        self.unscaledlbda = np.sqrt(np.log(self.dg+self.dc)/self.n) 
        
        self.set_sparsity_weights() # weighting scheme in sparse regularization

    def get_name(self):
        return self.name

    def get_shapes(self):
        return self.shapes

    def set_sparsity_weights(self):
        """  use adjusted weights for all groups as suggested by LST2015
        (may be essential for "good", "consistent" results)"""
       
        # precompute weights - we need the empiric standard deviations
        if self.useweights:
            # Gaussians
            self.mus = self.Y.sum(axis=0) / self.n
            self.sigmas = np.sqrt( (self.Y ** 2).sum(axis=0) / self.n - self.mus **2 )
            # categoricals
            sigma_r = np.empty(self.dc)
            freqs = self.D.sum(axis=0) / self.n
            for r in range(self.dc):
                sig_r = 0
                for k in range(self.sizes[r]):
                    p_xr_k = freqs[self.Lsum[r] + k] # relative probability that x_r has value k
                    sig_r += p_xr_k * (1 - p_xr_k) 
                sigma_r[r] = np.sqrt(sig_r)
                
            sigma_s = self.sigmas
        else: 
            sigma_r = np.ones(self.dc)
            sigma_s = np.ones(self.dg)
  
#        print(sigma_r)
        self.weights = {}
        for j in range(self.dc):
            for r in range(j):
                self.weights[('Q', r, j)] = sigma_r[r] * sigma_r[j]
            for s in range(self.dg):
                self.weights[('R', s, j)] = sigma_r[j] * sigma_s[s]
        for t in range(self.dg):
            for s in range(t):
                self.weights[('B', s, t)] = sigma_s[s] * sigma_s[t]

        # print weights
#        for key in sorted([a for a in self.weights]):
#            print(key, self.weights[key])

    def unpack(self, x):
        """unpack model parameters from vector x, save: returns copy"""
        offset = 0
        params = []
        xcopy = x.copy() # allows modifying the copy without modifying x
        for _, shapedim in self.shapes:
            h = np.prod(shapedim)
            params.append(xcopy[offset: offset+h].reshape(shapedim))
            offset += h
        
        return params
    
    def pack(self, components): 
        """pack (typically) gradients into vector x"""
        g = np.empty(self.totalnumberofparams)
        offset = 0
        for i, component in enumerate(components):
            size = np.prod(self.shapes[i][1]) 
#            print(self.shapes[i][0], size, np.prod(component.shape))
            assert size == np.prod(component.shape)
            g[offset: offset + size] = component.flatten() # row-wise, same as .ravel()
            offset += size
        return g

    def get_canonicalparams(self, x): 
        """a function to display the problem parameters
        
        overwritten in derived classes to return a specific model class
        
        """
        params = self.unpack(x)
        for i, p in enumerate(params):
            print('%s:\n'% self.shapes[i][0], p)
        
        return params

    def get_regularization_params(self):
        raise NotImplementedError # overwritten in derived classes
