# Copyright (c) 2019 Frank Nussbaum (frank.nussbaum@uni-jena.de)
"""
@author: Frank Nussbaum

base class for CG models and solvers
"""
import abc

import numpy as np

from cgmodsel.models.model_base import get_modeltype

from cgmodsel.models.model_pwsl import ModelPairwiseSL

######### base class for all model solvers ########################

class BaseSolver(abc.ABC):
    def __init__(self, meta, useweights=False, reduced_levels = False):
        """must call method drop_data after initialization"""

#        print('Init BaseSolver')
        super().__init__()
        self.useweights = useweights # determines usage of sparse norm calibration scheme
        
        self.cat_data = None # discrete data, dropped later
        self.cont_data = None # continuous data, dropped later
        
        self.problem_vars = None

        self.reduced_levels = reduced_levels
        # if True, parameters set to zero due to identifiability constraints are removed
        # if False, these parameters are included but are box-constrained to [0,0]
        
        if reduced_levels:
            self.sizes = [size - 1 for size in meta['sizes']] # do not modify meta
        else:
            self.sizes = meta['sizes']

        self.n_cgvars = meta['dg']# TODO: number of Gaussian variables
        self.n_catvars = len(self.sizes) # number of discrete variables

        self.ndata = 0 # number of data points

        self.cat_glims = np.cumsum([0] + self.sizes) # cumulative # of levels
        self.Ltot = self.cat_glims[-1] # total number of discrete levels (-dc if reduced_levels)
        
        self.d = self.Ltot + self.n_cgvars # size of one axis of the interaction parameter matrix
        
        # variables that need to be overwritten by derived classes
        self.shapes = []
        self.n_params = -1

        self.ndataame = 'base'
        
        self.modeltype = get_modeltype(self.n_catvars, self.n_cgvars, self.sizes)


        off = self.opts['off']
        if len(self.sizes) > 0 and max(self.sizes) > 1: # use group norms (reduced sizes are used)
            groupdelimiters = self.sizes + self.n_cgvars * [1]
            cumulative = np.cumsum([0] + groupdelimiters)
        else:
            groupdelimiters = None
            cumulative = None
            
        self.func_shrink = lambda S, tau: grp_soft_shrink(S, tau,
                                        groupdelimiters, cumulative, off=off)
        self.sparsenorm = lambda S: l21norm(S, groupdelimiters,
                                            cumulative, off=off)




    def drop_data(self, discrete_data, continuous_data):
        """drop data, derived classes may perform additional computations"""
#        print(self.n_cgvars, continuousdata.shape[1])
        assert self.n_cgvars == continuous_data.shape[1]
#        print(self.Ltot, discretedata.shape[1])
        assert self.Ltot == discrete_data.shape[1]

        self.cat_data = discrete_data
        self.cont_data = continuous_data
        self.ndata = continuous_data.shape[0]

        # here is a prescaling factor for the sparse norm regularization
        # from the original of Chandrasekaran (it is motivated by consistency analysis):
        self.unscaledlbda = np.sqrt(np.log(self.n_cgvars+self.n_catvars)/self.ndata) 
        
        self.set_sparsity_weights() # weighting scheme in sparse regularization

    def get_name(self):
        return self.ndataame

    def get_shapes(self):
        return self.shapes

    def set_sparsity_weights(self):
        """  use adjusted weights for all groups as suggested by LST2015
        (may be essential for "good", "consistent" results)"""
       
        # precompute weights - we need the empiric standard deviations
        if self.useweights:
            # Gaussians
            self.mus = self.cont_data.sum(axis=0) / self.ndata
            self.sigmas = np.sqrt( (self.cont_data ** 2).sum(axis=0) / self.ndata - self.mus **2 )
            # categoricals
            sigma_r = np.empty(self.n_catvars)
            freqs = self.cat_data.sum(axis=0) / self.ndata
            for r in range(self.n_catvars):
                sig_r = 0
                for k in range(self.sizes[r]):
                    p_xr_k = freqs[self.cat_glims[r] + k] # relative probability that x_r has value k
                    sig_r += p_xr_k * (1 - p_xr_k) 
                sigma_r[r] = np.sqrt(sig_r)
                
            sigma_s = self.sigmas
        else: 
            sigma_r = np.ones(self.n_catvars)
            sigma_s = np.ones(self.n_cgvars)
  
#        print(sigma_r)
        self.weights = {}
        for j in range(self.n_catvars):
            for r in range(j):
                self.weights[('Q', r, j)] = sigma_r[r] * sigma_r[j]
            for s in range(self.n_cgvars):
                self.weights[('R', s, j)] = sigma_r[j] * sigma_s[s]
        for t in range(self.n_cgvars):
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
        g = np.empty(self.n_params)
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


class BaseSolverSL(BaseSolver):
    """
    base class for S+L model solvers
    """
    
    def __init__(self, *args, **kwargs):
        print('Init BaseSolverSL')
        super().__init__(*args, **kwargs)

        self.alpha, self.beta = None, None
        self.lbda, self.rho = None, None

        self.opts.setdefault('off', 0) # if 1 regularize only off-diagonal
        
        self.problem_vars = None
        
    def __str__(self):
        s='<ADMMsolver> la=%s'%(self.lbda) + ', rho=%s'%(self.rho)
        s+=', alpha=%s'%(self.alpha) + ', beta=%s'%(self.beta)
        s+=', sc_a=%s'%(self.scale_l1) + ', sc_b=%s'%(self.scale_nuc) # TODO
        
        return s

#  def set_regularization_params(self, hyperparams, set_direct=False,
#                                  ptype='std', useunscaledlbda=True):
#        """set regularization parameters        
#        
#        hyperparams ... pair of regularization parameters
#        
#        ptype ... if 'std', then lambda, rho = hyperparams
#                    min l(S+L) + la * ||S||_{2,1} + rho * tr(L)
#                    s.t. Lambda[S+L]>0, L>=0
#                  else assume that alpha, beta = hyperparams and
#                  alpha, beta are weights in [0,1] and the problem is
#                   min (1-alpha-beta) * l(S+L) + alpha * ||S||_{2,1} + beta * tr(L)
#                   s.t. Lambda[S+L]>0, L>=0
#        
#        In addition to the specified regularization parameters, 
#        the regularization parameters can be scaled by a fixed value (depending
#        on the number of data points and variables):
#            
#        set_direct ... if False, then do not use scaling
#        
#        scales ... if None, use standard scaling np.sqrt(log(dg)/n)
#                   else  scales must be a two-tuple, and lambda and rho are 
#                   scaled according to the elements of this two-tuple
#        """
#        if ptype == 'std': # standard regularization parameters, first for l21, second for nuclear norm
#            self.lbda, self.rho = hyperparams
#            
#            if not set_direct:
#                assert self.n_cgvars > 0
#
#        else: # convex hyperparams are assumed
#            alpha, beta = hyperparams
#            assert alpha + beta < 1, "must contain likelihood part"
#            denom = 1 - alpha - beta
#            self.lbda = alpha / denom
#            self.rho = beta / denom
#
#        if useunscaledlbda:
#            if type(self.unscaledlbda) is type(()): # individual scaling of lambda and rho
#                self.lbda *= self.unscaledlbda[0]
#                self.rho *= self.unscaledlbda[1]
#            else:
#                self.lbda *= self.unscaledlbda
#                self.rho *= self.unscaledlbda
                
    

    def get_canonicalparams(self):
        """Retrieves the PW S+L CG model parameters from flat parameter vector.
        
        output: Model_PWSL instance"""
        
        mat_s, mat_l, alpha = self.problem_vars
        
        mat_lambda = -mat_s[self.Ltot:, self.Ltot:] # cts-cts parameters 
        # they have negative sign in CG pairwise interaction parameter matrix

        if self.n_cgvars > 0:
            mat_q = mat_s[:self.Ltot, :self.Ltot]
            mat_r = mat_s[self.Ltot:, :self.Ltot]        
            vec_u =  0.5 * np.diag(mat_q).copy().reshape(self.Ltot)
            for r in range(self.n_catvars): # set block-diagonal to zero
                mat_q[self.cat_glims[r]:self.cat_glims[r+1], 
                      self.cat_glims[r]:self.cat_glims[r+1]] = \
                      np.zeros((self.sizes[r], self.sizes[r]))
        else:
            mat_q = np.empty(0)
            mat_r = np.empty(0)
            vec_u = np.empty(0)
        
        can_pwsl = vec_u, mat_q, mat_r, alpha, mat_lambda, mat_l
        
        annotations = {'n': self.ndata, 'la': self.lbda, 'rho': self.rho}

        if self.reduced_levels:
            fullsizes = [size + 1 for size in self.sizes]
        else:
            fullsizes = self.sizes

        meta = {'dc': self.n_catvars, 'dg': self.n_cgvars,'sizes': fullsizes}
        return ModelPairwiseSL(can_pwsl, meta,
                          annotations=annotations, in_padded=False)


    def get_regularization_params(self):
        """get regularization parameters"""
        return self.lbda, self.rho

    def set_regularization_params(self, hyperparams, set_direct: bool=False,
                                  scales=None, ptype='std'):
        """set regularization parameters        
        
        hyperparams ... pair of regularization parameters
        
        ptype ... if 'std', then lambda, rho = hyperparams
                    min l(S-L) + la * ||S||_1 + rho * tr(L)
                    s.t. S-L>0, L>=0
                  else assume that alpha, beta = hyperparams and
                  alpha, beta are weights in [0,1] and the problem is
                   min (1-alpha-beta) * l(S-L) + alpha * ||S||_1 + beta * tr(L)
                   s.t. S-L>0, L>=0
        
        In addition to the specified regularization parameters, 
        the regularization parameters can be scaled by a fixed value (depending
        on the number of data points and variables):
            
        set_direct ... if False, then do not use scaling
        
        scales ... if None, use standard scaling np.sqrt(log(dg)/n)
                   else  scales must be a two-tuple, and lambda and rho are 
                   scaled according to the elements of this two-tuple
        """
        if scales is None:
            self.scale_l1, self.scale_nuc = self.scales
        else:
            self.scale_l1, self.scale_nuc = scales
        if ptype == 'std': # standard regularization parameters, first for l21, second for nuclear norm
            self.lbda, self.rho = hyperparams
            
            if not set_direct:
                assert self.ndata > 0, "No data provided"
                self.lbda *= self.scale_l1
                self.rho *= self.scale_nuc
        else: # convex hyperparams are assumed
            alpha, beta = hyperparams
            self.alpha = alpha
            self.beta = beta
#            assert alpha+beta < 1, "must contain likelihood part"
            denom = 1 - alpha - beta

            if denom != 0:
                self.lbda = self.scale_l1 * alpha / denom
                self.rho = self.scale_nuc * beta  /denom
            else:
                self.lbda, self.rho = 0, 0