# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
from cgmodsel.models.model_base import _invert_indices
from cgmodsel.models.model_base import Model_PW_Base
#from cgmodsel.utils import _logsumexp_and_conditionalprobs
#import cgmodsel.constants as constants

import numpy as np

##################################
class Model_PW(Model_PW_Base):
    """
    class for parameters of distribution
    p(x,y) ~ exp(1/2 (C_x, y)^T Theta (C_x y) + u^T C_x + alpha^T y)
    
    this class uses padded parameters (that is, parameters for 0-th levels are
    included, however, one might want them to be constrained to 0 for 
    identifiability reasons)
    
    here:
    [C_x .. dummy representation of x]
    u .. univariate discrete parameters
    alpha .. univariate cont. parameters
    Theta ... pairwise interaction parameter matrix given by
        Theta = (Q & R^T \\ R & -Lambda)
    with 
    Q .. discrete-discrete interactions
    R .. discrete-cont. interactions
    Lambda .. cont-cont interactions
    
    initialize with tuple pw = (u, Q, R, alpha, Lambda)
    
    Attention to the special case of only Gaussians observed variables:
    uses as above
        p(y) ~ exp(1/2  y^T Theta y + alpha^T y),
    i.e. S has inverted sign
    instead of the representation in Chandrasekaran et al. (2011)
        p(y) ~ exp( -1/2 y^T Theta y + alpha^T y)
    
    """
    
    
    def __init__(self, pw, meta, **kwargs):
        Model_PW_Base.__init__(self, pw, meta, **kwargs)
        # possible extension: pw params in dict, allow passing of pwmat
         
        self.name = 'PW'

    def __str__(self): 
        """a string representation of the model"""
        s = ''
        if np.prod(self.Q.shape)>0: s += '\nQ:\n' + str(self.Q)
        if np.prod(self.u.shape)>0: s += '\nu:' + str(self.u.T)
        if np.prod(self.R.shape)>0: s += '\nR:\n' + str(self.R)
        if np.prod(self.alpha.shape)>0: s += '\nalpha:' + str(self.alpha)
        if np.prod(self.Lambda.shape)>0: s += '\nLambda:\n' + str(self.Lambda)
        
        return s[1:]
    


    def add_latent_CG(self, dl=1, seed=-1, makePD=False, 
                      connectionprob=.95, strength=.5,
                      disscale=.3, ctsscale=.2, marginalize=True):
        """
        augment dl latent variables to given pairwise model pwmodel
        
        connectionprob ... probability of interaction between latent var
        
        marginalize ... if True, return S+L model (class Model_PWSL)
                        with the latent variables marginalized out
        
        experimental code,
        various other parameters to determine interactions strengths etc.
        """
        if seed != -1:
            print('Model_PW.add_latent_CG: Set seed to %d'%(seed))
            np.random.seed(seed)
        dc = self.dc
        dg = self.dg
        
        LambdaHO = np.zeros((dl, dg))
        alphaH = np.zeros(dl)
        RH = np.zeros((dl, self.Ltot))
        
        def get_entry(dim, prob =.8, offset=.7, scale=.3, samesign=True):
            """sample one interaction parameter"""
            if np.random.random() > prob:                
                return np.zeros(dim)
            if samesign:
                sign = -1
                if np.random.random() > .5:
                    sign = 1
            a = np.empty(dim)
            for i in range(dim):
                if not samesign:
                    sign = -1
                    if np.random.random()>0.5:
                        sign = 1
                a[i] = sign * (offset + scale * np.random.random())

            return a
        
        ## add edges with specified probability
        for l in range(dl):
            for r in range(dc): 
                RH[l, self.Lcum[r]+1:self.Lcum[r+1]] = \
                   get_entry(self.sizes[r]-1, prob=connectionprob,
                             offset=strength, scale=disscale)
            
            for s in range(dg): # dg
                LambdaHO[l, s] = get_entry(1, prob=connectionprob,
                        offset=strength + .3, scale=ctsscale)
        
        ## update model parameters
        
        c = 1.0
        if self.dg>0: # ensure positive definiteness of precision matrix
            tmp = np.dot(LambdaHO.T, LambdaHO) # dg by dg
            # calculate smallest eigenvalue of A=Lambda-c*tmp
            # choose c s.t. smallest eigenvalue is positive for a valid distribution
            # could use power method on A^(-1)
            while True:
                lamin = np.min(np.linalg.eigvals(self.Lambda - c * tmp))
                if lamin <.2:
                    c *= 2 / 3
                else:
                    break
            
        if c != 1.0:
            print('Warning(model_pw.add_latent_gaussians): made precision matrix of full model PD, c=%f'%(c))

        fac = np.sqrt(c)
        B = np.empty((self.dg + dl, self.dg + dl)) # construct new precision matrix of full model
        B[:-dl, :-dl] = self.Lambda
        B[:-dl, -dl:] = fac * LambdaHO.T # dg by dl
        B[-dl:, :-dl] = fac * LambdaHO
        B[-dl:, -dl:] = np.eye(dl) # orthogonal latent variables
        
        if self.dg>0:
            if self.dc>0:
                self.R = np.concatenate((self.R, RH), axis=0)
#            print(self.alpha, alphaH)
            self.alpha = np.concatenate((self.alpha, alphaH))
        else:
            self.alpha = alphaH
            self.R = RH
        
        self.dg += dl
        
        self.Lambda = B
        
        if marginalize: # return marginal model
            return self.marginalize(drop_idx=[self.dg-i-1 for i in range(dl)], verb = 0)

    def get_meanparams(self):
        """
        wrapper for _get_meanparams from base class Model_PW_Base
        """
        pwparams = self.u, self.Q, self.R, self.alpha, self.Lambda
        
        return self._get_meanparams(pwparams)
    
#    def get_regressionparams(self, r):
#        assert self.modeltype == constants.BINARY
#        
##        print(self.Q)
#        sliceQ = self.Q[self.Lcum[r]+1, :]
##        print(sliceQ)
#        from cgmodsel.utils import dummypadded_to_unpadded
#        params = dummypadded_to_unpadded(sliceQ, self.dc)
#        params = np.concatenate((params[:r],params[r+1:]))
##        print(params)
#        intercept = self.u[self.Lcum[r]+1]
#        
#        return intercept, params
#
#    def get_ncprobs(self, Dx, y):
#        ## ** discrete node conditionals ** 
#        W = np.dot(y, self.R) + np.dot(Dx, self.Q) + self.u.flatten() # n by Ltot
#        print(W.shape, self.Lcum)
#        for r in range(self.dc):
#            Wr = W[self.Lcum[r]:self.Lcum[r+1]] # view of W
#            print(Wr)
#            conditionalprobs = np.exp(Wr-np.amax(W))
#            conditionalprobs /= np.sum(conditionalprobs)
#            print('R', r,  conditionalprobs)
#        
#        return conditionalprobs
        
    def get_params(self):
        """return model parameters """
        return self.u, self.Q, self.R, self.alpha, self.Lambda

    def marginalize(self, drop_idx=[], verb=False):
        """ marginalize out CG variables
        
        drop_idx ... indices of (conditional) Gaussians to be marginalized out
        
        returns Model_PWSL instance

        marginalization follows a Schur-complement formula given in the Dissertation script
        (see section marginalization of pairwise CG distributions)
        the decomposition is of the form S + L
        """
        # import here to avoid files for both classes importing each other
        from cgmodsel.model_pwsl import Model_PWSL
        
        H = drop_idx
        dl = len(H)
        assert dl > 0, "Supply with indices for marginalization"
        #print(self.dg, dl)
        do = self.dg - dl # number of observed (remaining) Gaussians

        V = _invert_indices(H, self.dg) # indices of Gaussians to keep (visible)
       
        ## observed direct interactions (together they form S)
        SQ = self.Q
        SR = self.R[V, :]
        SLa = self.Lambda[np.ix_(V, V)]
        
        ## latent interactions (together they form L)   
        RH = self.R[H, :]
        LambdaHH = self.Lambda[np.ix_(H, H)]
        Si_HH = np.linalg.inv(LambdaHH)
        LambdaHV = self.Lambda[np.ix_(H, V)]

        ## calculate L from latent interactions
        tmp_expr1 = np.dot(LambdaHV.T, Si_HH)
     
        LQ = np.dot(np.dot(RH.T, Si_HH), RH) # has univariate effects on its diagonal
        LR = - np.dot(tmp_expr1, RH)
        LLa= np.dot(tmp_expr1, LambdaHV) # see Diss(3.9), Schur complement

        L = np.empty((self.Ltot + do, self.Ltot + do))
        L[:self.Ltot, :self.Ltot] = LQ
        L[self.Ltot:, :self.Ltot] = LR
        L[:self.Ltot, self.Ltot:] = LR.T
        L[self.Ltot:, self.Ltot:] = LLa
        
#        Thmarg = SLa - LLa
#        minla = np.min(np.linalg.eigvals(Thmarg))
#        print('marg-minla', minla)
#        assert  minla > 0

        ## the observed alpha is also changed
        tialpha = self.alpha[V] - np.dot(tmp_expr1, self.alpha[H]) # see (3.8)
        
        # note that Theta_Marg = S + L (where Lambda has neg sign in Theta_Marg)
            
        ## construct a PW-SL model and return it
        sl_params = self.u, SQ, SR, tialpha, SLa, L
#        print(sl_params)
        meta = {'dc':self.dc, 'dg':do, 'sizes':self.sizes}
        
        sl_params_class = Model_PWSL(sl_params, meta)
        
        if verb:
            eigvals = np.linalg.eigvals(L)
            print('eigvals L:', eigvals)

        return sl_params_class
    
#    def normalize(self):
#        """ scale all pairwise parameters"""
#        raise NotImplementedError
