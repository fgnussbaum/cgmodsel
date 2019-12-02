# Copyright (c) 2019 Frank Nussbaum (frank.nussbaum@uni-jena.de)
"""
@author: Frank Nussbaum

base class for CG models and solvers
"""
import abc
import numpy as np

#from cgmodsel.models.model_base import get_modeltype
from cgmodsel.models.model_pwsl import ModelPairwiseSL

######### base class for all model solvers ########################

class BaseSolver(abc.ABC):
    def __init__(self, useweights=False):
        """must call method drop_data after initialization"""

#        print('Init BaseSolver')
        super().__init__()
        self.useweights = useweights # determines usage of sparse norm calibration scheme
        
        self.cat_data = None # discrete data, dropped later
        self.cont_data = None # continuous data, dropped later
        
        self.problem_vars = None
        self.meta = {'n_data': 0}

        # variables that need to be overwritten by derived classes
#        self.shapes = [] # TODO: Huber only
#        self.n_params = -1

        self.name = 'base'

    def _postsetup_data(self):
        """called after drop_data """
#        self.set_sparsity_weights() # weighting scheme in sparse regularization
        pass # may be overwriten in derived classes

    def drop_data(self,
                  data, 
                  meta: dict) -> None:
        """drop data, derived classes may perform additional computations"""

        # process argument data
        counter = 0
        if 'n_cat' in meta and meta['n_cat'] > 0:
            counter += 1
            cat_data = data
            assert 'sizes' in meta
            assert len(meta['sizes']) == meta['n_cat']
        if 'n_cg' in meta and meta['n_cg'] > 0:
            counter += 1
            cont_data = data
        if counter == 0:
            raise Exception('Meta information states no variables')
        elif counter == 2:
            assert len(data) == 2
            cat_data, cont_data = data
        
        self.meta = meta.copy()
        # check validity of dictionary meta

        if 'n_cg' in meta:
            if meta['n_cg'] > 0:
                assert not np.any(np.isnan(cont_data))
                assert meta['n_cg'] == cont_data.shape[1]
                self.cont_data = cont_data
                self.meta['n_data'] = cont_data.shape[0]
        else:
            self.meta['n_cg'] = 0
        if 'n_cat' in meta:
            if meta['n_cat'] > 0:

                self.cat_data = cat_data
                if 'n_data' in self.meta:
                    assert self.meta['n_data'] == cat_data.shape[0]
                
                ltot = np.sum(meta['sizes'])
                if ltot == cat_data.shape[1]:
                    # 0-th levels of the discrete data are contained
                    # for identifiability, assume that corresponding 
                    # parameters are constrained to zero
                    self.meta['red_levels'] = False
                elif ltot - meta['n_cat'] == cat_data.shape[1]:
                    # assume that 0-th levels are left out in discrete data
                    # assures identifiability of the model
                    self.meta['red_levels'] = True
                    self.meta['sizes'] = [size - 1 for size in meta['sizes']]
                else:
                    raise Exception('Dimension of discrete_data do not fit')
                self.meta['ltot'] = cat_data.shape[1]
                
                # calculate cumulative # of levels/ group delimiters
                self.cat_glims = np.cumsum([0] + self.sizes)
            else:
                self.meta['ltot'] = 0
        else:
            self.meta['n_cat'] = 0
            self.meta['ltot'] = 0

        self.meta['dim'] = self.meta['ltot'] + self.meta['n_cg']
        
#        self.meta['type'] = get_modeltype(self.n_cat, self.n_cg, self.sizes)        

        fac = np.log(self.meta['n_cg']+self.meta['n_cat'])
        fac = np.sqrt(fac /self.meta['n_data'])
        self.meta['reg_fac'] = fac # potentially used as prescaling factor 
        # for regularization parameters

        self._postsetup_data()

    def get_name(self):
        return self.name

    def set_sparsity_weights(self):
        """  use adjusted weights for all groups as suggested by LST2015
        (may be essential for "good", "consistent" results)"""
       
        # precompute weights - we need the empiric standard deviations
        if self.useweights:
            # Gaussians
            self.mus = self.cont_data.sum(axis=0) / self.n_data
            self.sigmas = np.sqrt( (self.cont_data ** 2).sum(axis=0) / self.n_data - self.mus **2 )
            # categoricals
            sigma_r = np.empty(self.n_cat)
            freqs = self.cat_data.sum(axis=0) / self.n_data
            for r in range(self.n_cat):
                sig_r = 0
                for k in range(self.sizes[r]):
                    p_xr_k = freqs[self.cat_glims[r] + k] # relative probability that x_r has value k
                    sig_r += p_xr_k * (1 - p_xr_k) 
                sigma_r[r] = np.sqrt(sig_r)
                
            sigma_s = self.sigmas
        else: 
            sigma_r = np.ones(self.n_cat)
            sigma_s = np.ones(self.n_cg)
  
#        print(sigma_r)
        self.weights = {}
        for j in range(self.n_cat):
            for r in range(j):
                self.weights[('Q', r, j)] = sigma_r[r] * sigma_r[j]
            for s in range(self.n_cg):
                self.weights[('R', s, j)] = sigma_r[j] * sigma_s[s]
        for t in range(self.n_cg):
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
        string='<ADMMsolver> la=%s'%(self.lbda) + ', rho=%s'%(self.rho)
        string+=', alpha=%s'%(self.alpha) + ', beta=%s'%(self.beta)
        # string+=', sc_a=%s'%(self.scale_l1) + ', sc_b=%s'%(self.scale_nuc)
        
        return string
    

    def get_canonicalparams(self):
        """Retrieves the PW S+L CG model parameters from flat parameter vector.
        
        output: Model_PWSL instance"""
        
        mat_s, mat_l, alpha = self.problem_vars

        ltot = self.meta['ltot']
        
        mat_lambda = -mat_s[ltot:, ltot:] # cts-cts parameters 
        # have negative sign in CG pairwise interaction parameter matrix

        if self.n_cat > 0:

            glims = self.meta['cat_glims']
            sizes = self.meta['sizes']
            
            mat_q = mat_s[:ltot, :ltot]
            mat_r = mat_s[ltot:, :ltot]        
            vec_u =  0.5 * np.diag(mat_q).copy().reshape(ltot)
            for r in range(self.n_cat): # set block-diagonal to zero
                mat_q[glims[r]:glims[r+1], 
                      glims[r]:glims[r+1]] = \
                      np.zeros((sizes[r], sizes[r]))
        else:
            mat_q = np.empty(0)
            mat_r = np.empty(0)
            vec_u = np.empty(0)
        
        can_pwsl = vec_u, mat_q, mat_r, alpha, mat_lambda, mat_l
        
        annotations = {'n': self.meta['n_data'],
                       'lambda': self.lbda,
                       'rho': self.rho}

        if self.meta['red_levels']:
            fullsizes = [size + 1 for size in sizes]
        else:
            fullsizes = sizes

        meta = {'n_cat': self.meta['n_cat'],
                'n_cg': self.meta['n_cg'],
                'sizes': fullsizes}
        return ModelPairwiseSL(can_pwsl, meta,
                          annotations=annotations, in_padded=False)


    def get_regularization_params(self):
        """get regularization parameters"""
        return self.lbda, self.rho

    def set_regularization_params(self,
                                  hyperparams, 
                                  scales=None,
                                  set_direct = False,
                                  ptype: str='std') -> None:
        """set regularization parameters        
        
        hyperparams ... pair of regularization parameters
        
        ptype ... if 'std',
                    set lambda, rho = hyperparams * scaling(n, nvars), where 
                    the parameters are for the problem
                        min l(S-L) + lambda * ||S||_1 + rho * tr(L)
                        s.t. S-L>0, L>=0
                    Here, scaling(n, nvars) is a scaling suggested by
                    consistency results
                    Argument <scales> is not used in this case!
                  if 'direct', directly set lambda, rho = hyperparams
                  if 'convex' assume that alpha, beta = hyperparams and
                  alpha, beta are weights in [0,1] and the problem is
                   min (1-alpha-beta) * l(S-L) + alpha * ||S||_1 + beta * tr(L)
                   s.t. S-L>0, L>=0
        
        In addition to the specified regularization parameters, 
        the regularization parameters can be scaled by a fixed value (depending
        on the number of data points and variables):
        
        scales ... if None, use standard scaling np.sqrt(log(dg)/n)
                   else  scales must be a two-tuple, and lambda and rho are 
                   scaled according to the elements of this two-tuple
        """
        assert len(hyperparams) == 2
        assert hyperparams[0] >= 0 and hyperparams[1] >= 0
        
        if not set_direct:
            if not scales is None:
                scale_lbda, scale_rho = scales
            else:
                assert self.meta['n_data'] > 0, \
                    "data-dependent scaling, drop data first"
                # calculate prescaling factor for the regularization parameters
                # based on consistency analysis by Chandrasekaran et. al (2010)
    
#                assert 'reg_fac' in self.meta
                scale_lbda = self.meta['reg_fac']
                scale_rho = self.meta['reg_fac']
            
        if ptype == 'std':
            # standard regularization parameters
            # first for l21, second for nuclear norm
            self.lbda, self.rho = hyperparams
            if not set_direct:
                self.lbda *= scale_lbda
                self.rho *= scale_rho

        elif ptype == 'convex':
            alpha, beta = hyperparams
#            assert alpha + beta <= 1
            assert alpha+beta < 1, "must contain likelihood part"
            
            self.alpha = alpha
            self.beta = beta
            
            denom = 1 - alpha - beta

            if denom != 0:
                self.lbda = self.scale_l1 * alpha / denom
                self.rho = self.scale_nuc * beta  /denom
#            else:
#                # no likelihood part
#                self.lbda, self.rho = 0, 0
            
            if not set_direct:
                self.lbda *= scale_lbda
                self.rho *= scale_rho

        else:
            raise Exception('unknown ptype')