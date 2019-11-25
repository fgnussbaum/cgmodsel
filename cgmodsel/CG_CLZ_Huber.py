# Copyright (c) 2017-2019 Frank Nussbaum (frank.nussbaum@uni-jena.de)
"""
@author: Frank Nussbaum

A class to fit CLZ CG models

this code is experimental
"""

from cgmodsel.CG_base_Huber import CG_base_Huber, _huberapprox
from cgmodsel.models.model_clz import Model_CLZ

import numpy as np
from math import inf # used in derived classes


############################################################## FUll Pseudo LH
class CG_CLZ_Huber(CG_base_Huber): 
    """
    A class that provides with methods (model selection, crossvalidation) associated with CLZ CG models
    
    code in this class is experimental    
    """
    def __init__(self, meta, **kwargs):
        """pass a dictionary that provides with keys dg, dc, and L"""
        
        self._set_defaults(**kwargs)
        CG_base_Huber.__init__(self, meta, self.opts['useweights'])

        # dimensions of parameters Q, u, R, B, B^d, alpha
        self.shapes = [('Q', (self.Ltot, self.Ltot)), ('u', (self.Ltot,1)), \
                        ('R', (self.dg, self.Ltot)),
                        ('B0', (self.dg, self.dg)), ('beta0', (self.dg, 1)), \
                        ('B', (self.dg, self.dg, self.Ltot)), ('Bdiag', (self.dg, self.Ltot)), \
                        ('alpha', (self.dg, 1))]
        self.totalnumberofparams = sum( [np.prod(shape[1]) for shape in self.shapes] )

        self.name = 'CLZ'
        
    def drop_data(self, discretedata, continuousdata):
        """drop data and 
        
        code does not check input dimensions"""        
        CG_base_Huber.drop_data(self, discretedata, continuousdata) # self.Y, self.D, self.n, self.dg ...
        
        # precompute array of size n * dg ** 2
        self.YsYt = np.empty((self.n, self.dg ** 2)) # store values of squared cts empirical sufficient statistics
        for s in range(self.dg):
            for t in range(self.dg):
                self.YsYt[:, s*self.dg + t] = np.multiply(self.Y[:, s], self.Y[:, t])
        self.YsYs = np.empty((self.n, self.dg))
        for s in range(self.dg):
            self.YsYs[:, s] = self.YsYt[:, s*(self.dg+1)]

    def set_regularization_params(self, kS):
        """set regularization parameters
        kS is a scaling parameter for the factor lambda for the group-sparsity norm
        
        min l(Theta) + kS*la* (sum_g ||Theta_g||_{2}), where the sum is over the groups
        """
        self.lbda = kS * np.sqrt(np.log(self.dc + self.dg) / self.n)

        self.set_sparsity_weights() # weighting scheme for sparse regularization
        
############ ** model specific functions required for solving** ################################################

    def get_bounds(self): # model specific, only difference to pw are factors Ltot in B, Bd
        """return bounds for l-bfgs-b solver"""
        # some of the parameters are constrained to be zero as a consequence of identifiability assumptions
        bnds = []

        lLtot_ident = [] # bounds for a row B_{\cdot, s, t} of flattened B
        lLtot_diag = []
        for r in range(self.dc): # TODO: zero bounds for diagonal elements of matrices B_rk
            lLtot_ident += [(0,0)]
            lLtot_ident+=  (self.sizes[r] - 1) * [(-inf, inf)] 
            lLtot_diag += [(0,0)]
            lLtot_diag+=  (self.sizes[r] - 1) * [(10E-6, inf)]

        # Q
        for r in range(self.dc):
            bnds += self.Ltot * [(0,0)]
            bnds += (self.sizes[r]-1) * lLtot_ident
        
        bnds+= lLtot_ident # u
        bnds+= self.dg * lLtot_ident # R
        
        for s in range(self.dg - 1): # B0 with zero bounds on diagonal
            bnds+=[(0,0)]
            bnds+= self.dg * [(-inf, inf)]
        bnds += [(0,0)]      
        
        bnds += self.dg * [(10E-6, inf)] # beta0 diagonal

        
        for s in range(self.dg - 1): # B with zero bounds on diagonal
            bnds+=self.Ltot * [(0,0)]
            bnds+= self.dg * lLtot_ident
        bnds += self.Ltot * [(0,0)]
       
        bnds+= self.dg * lLtot_diag # B diagonal
        
        bnds+= self.dg * [(-inf, inf)] # alpha

        return bnds

    def get_starting_point(self, random=False, seed=10): # model independent given # params
        """return a starting point containing zeros"""
        x0 = np.zeros(self.totalnumberofparams)
        # TODO: zero precmat diagonal entries? LBFGSB is able to handle infeasible starting point (by first projecting to a point that satisfies the box constraints)
        if random:
            np.random.seed(seed)
            x0 = np.random.random(self.totalnumberofparams)
            x0 = self.pack(self.preprocess(x0)) # preprocess ensures symmetry
#            print('Starting value x0=')
#            self.print_params(x0)
        else:
            self.currentsolution = x0
        
        return x0

    def set_sparsity_weights(self, useweights = True):
        """  use adjusted weights for all groups as suggested by LH
        (this is essential for good results)"""
        # precompute weights - we need the empiric standard deviations

        if useweights: # TODO: come up with a weighting scheme for CLZ models
            self.mus = self.Y.sum(axis=0) / self.n 
            self.sigmas = np.sqrt( (self.Y **2).sum(axis=0) / self.n - self.mus **2 )
            
            # categoricals
            sigma_r = np.empty(self.dc)
            freqs = self.D.sum(axis = 0) / self.n
            for r in range(self.dc):
                sig_r = 0 # TODO: sigma?
                for k in range(self.sizes[r]):
                    p_xr_k = freqs[self.Lsum[r]+k] # relative probability that x_r has value k
                    sig_r += p_xr_k * (1 - p_xr_k) 
                sigma_r[r] = np.sqrt(sig_r)
                
            sigma_s = self.sigmas
        else: # do not use weights, i.e. weights equal to one
            sigma_r = np.ones(self.dc)
            sigma_s = np.ones(self.dg)
        
        self.weights = {}
        for r in range(self.dc):
            for j in range(r):
                self.weights[('dis_dis', r, j)] = sigma_r[r] * sigma_r[j]
            for s in range(self.dg):
                self.weights[('dis_cts', s, r)] = sigma_r[r] * sigma_s[s] #* 0.5
        for t in range(self.dg):
            for s in range(t):
                self.weights[('cts_cts', s, t)] = sigma_s[s] * sigma_s[t]

#        for key in sorted([a for a in self.weights]):
#            print(key, self.weights[key])
        

##### ** graph and representation **############################
    
    def get_canonicalparams(self, x, verb=False): # overwritten from base class
        """retrieves the CLZ-CG model parameters from flat parameter vector.
        output: CLZ params class"""
        Q, u, R, Lambda0, beta0, Lambdas, beta, alpha = self.preprocess(x) # return copy (since unpack does)

        Lambda0 += np.diag(beta0.reshape(-1))
        for r in range(self.dc):
            for k in range(self.sizes[r]):
                rk = self.Lsum[r] + k
                Lambdas[:, :, rk] += np.diag(beta[:, rk].reshape(-1))

        canparams = (u, Q, R, alpha, Lambda0, Lambdas)
        can_clz_class = Model_CLZ(canparams, {'dc':self.dc, 'dg':self.dg, 'sizes':self.sizes})

        if verb:
            print('Learned parameters:')
            print(can_clz_class)
        
        return can_clz_class
    
#    def disp_params_by_group(self, x):
#        # TODO: remove ? (probably only used for debugging)
#        Q, u, R, B0, beta0, B, beta, alpha = self.preprocess(x)
#
#        # discrete - discrete, as in pairwise model
#        print('dis-dis')
#        for r in range(self.dc): # Phis
#            for j in range(r):
#                wrj = self.weights[('dis_dis', r, j)] * self.lbda # TODO weighting scheme
#                tmp_group = Q[self.Lsum[r]:self.Lsum[r+1],
#                              self.Lsum[j]:self.Lsum[j+1]]
#                print(r, j, tmp_group)
#
#        # continuous - discrete
#        print('dis-cts')        
#        for r in range(self.dc): # Rhos
#            tmp_group = np.empty(self.sizes[r] *(1 + self.dg))
#            for s in range(self.dg):
#                tmp_group[:self.sizes[r]] = R[s, self.Lsum[r]:self.Lsum[r+1]]
#                tmp_group[self.sizes[r] :] = B[s*self.dg:(s+1)*self.dg , self.Lsum[r]:self.Lsum[r+1]].flatten() # params la_{st}^{r:k} for t\in [d_g], k\in[L_r] 
#                wrs = self.weights[('dis_cts', s, r)] * self.lbda # 
#                print(r, s, tmp_group)
#        
#        # continuous - continuous
#        print('cts-cts')
#        tmp_group = np.empty(self.Ltot + 1)
#        for t in range(self.dg): # upper triangle s<t
#            for s in range(t):
#                tmp_group[:self.Ltot] = B[s*self.dg+t, :]
#                tmp_group[self.Ltot] = B0[s, t]
#                wst = self.weights[('cts_cts', s, t)] * self.lbda
#                print(t, s, tmp_group)

        
#################################################################################

    def preprocess(self,x):
        """ unpack parameters from vector x and preprocess"""   
        Q, u, R, B0, beta0, B, beta, alpha = self.unpack(x) # modifying returned params does not modify x

        # preprocess - zero out diagonals (required e.g. for grad approximation)
        for rk in range(self.Ltot):
#            print(rk, B[:, :, rk])
            B[:, :, rk] -= np.diag(np.diag(B[:, :, rk])) # TODO: remove & LBFGSB bounds instead ? does not work with gradapproximation, because this adds eps also to the diagonal 
            B[:, :, rk] = np.triu(B[:, :, rk]) # use only upper triangle -> gradient approximation works only for upper triangle
            B[:, :, rk] = B[:, :, rk] + B[:, :, rk].T
        B0 -= np.diag(np.diag(B0)) # TODO: s.o.
        B0 = np.triu(B0)
        B0 = B0 + B0.T
        
        for r in range(self.dc): # set block-diagonal to zero
            Q[self.Lsum[r]:self.Lsum[r+1], self.Lsum[r]:self.Lsum[r+1]] = np.zeros((self.sizes[r], self.sizes[r])) # TODO: same here: LBFGSB bounds?
        Q = np.triu(Q)
        Q = Q + Q.T

        return (Q, u, R, B0, beta0, B, beta, alpha)
        
    def get_fval_and_grad(self, x, delta=None, sparse=False, smooth=True, verb='-'): 
        """calculate function value f and gradient g of CLZ model
        x    vector of parameters
        increases self.fcalls by 1, no other class variables are modified"""
        
        self.fcalls += 1 # tracks number of function value and gradient evaluations
        
        Q, u, R, B0, beta0, B, beta, alpha = self.preprocess(x) # unpack, zero out diagonals and symmetrize,...
        B = B.reshape((self.dg ** 2, self.Ltot)) # this is \tilde{B} (B tilde) from the doc

        # intitialize f and gradients
        f = 0
        grad = np.zeros(self.totalnumberofparams)
        gradQ, gradu, gradR, gradB0, gradbeta0, \
            gradB, gradbeta, gradalpha = self.unpack(grad) 

        if smooth:
            e = np.ones((self.n, 1))
            
            #### ** Gaussian node conditionals ** ####
            b = np.dot(self.D, beta.T) + np.dot(e, beta0.T)# n by dg
    
            M = np.dot(e, alpha.T) + np.dot(self.D, R.T) - np.dot(self.Y, B0)# n by dg, part as in pairwise model
            for s in range(self.dg): # new in CLZ
                M[:, s] -= np.sum(np.multiply(self.Y, np.dot(self.D, B[s*self.dg:(s+1)*self.dg, :].T)), axis = 1)
            
            Delta = np.divide(M, b) - self.Y # regression residual
            

            lG = - 0.5*np.sum(np.log(b)) \
                + 0.5*np.linalg.norm(np.multiply(Delta, np.sqrt(b)), 'fro')**2 
            lG /= self.n
            f+= lG

            # gradients as in pw model
            gradalpha = np.sum(Delta, 0).T # dg by 1
            gradR = np.dot(Delta.T, self.D) # dg by Ltot
            gradB0 = -np.dot(self.Y.T, Delta) # zero out diagonal and add transpose later
    
            # new gradients in CLZ model 
            for i in range(self.n):
                yiTdi = np.dot(self.Y[i, :].reshape((self.dg, 1)), Delta[i, :].reshape(1,self.dg)) # need to cast into matrix objects first
                for rk in np.where(self.D[i, :] == 1)[0]:
                    gradB[:, :, rk] -= yiTdi
            # later add transpose and zero out diagonal
    
            # gradients of diagonals 
            for s in range(self.dg):
                gradbeta[s, :] = np.dot(self.D.T, \
                - 0.5*np.divide(np.ones(self.n), b[:, s]) \
                + 0.5*np.multiply(Delta[:, s], Delta[:, s]) \
                - np.divide(np.multiply(Delta[:, s], M[:, s]), b[:, s])) # Ltot x 1 slice
#                print(gradbeta.shape)
#                gradbeta0[s] = np.sum(gradbeta[s, 0:self.sizes[0]]) # sum of the gradient of beta_{s,s, r:k} over k
                gradbeta0[s] = np.dot(e.T, \
                - 0.5*np.divide(np.ones(self.n), b[:, s]) \
                + 0.5*np.multiply(Delta[:, s], Delta[:, s]) \
                - np.divide(np.multiply(Delta[:, s], M[:, s]), b[:, s]))
            
            #### ** discrete node conditionals ** ####
            W = np.dot(self.Y, R) + np.dot(self.D, Q) + np.dot(np.ones((self.n, 1)), u.T) # n by Ltot, as in pw model      
#            asd = np.dot(self.YsYt, B)
            W += -0.5 * np.dot(self.YsYt, B) -0.5 * np.dot(self.YsYs, beta) # new in CLZ
      
            A = np.empty((self.n, self.Ltot))  
#            print(W[6, :])
            lD = 0
            for r in range(self.dc): # as in pw model, see doc
                Lr = self.sizes[r]
#                Wr = W[:, self.Lsum[r]:self.Lsum[r+1]]
                tmpexpWr = np.exp(W[:, self.Lsum[r]:self.Lsum[r+1]])
                Ar= np.divide(tmpexpWr, np.dot(tmpexpWr, np.ones((Lr, Lr))))
                A[:, self.Lsum[r]:self.Lsum[r+1] ]= Ar
#                ass=np.sum(np.multiply(Ar, self.D[:, self.Lsum[r]:self.Lsum[r+1]]), 1)
#                print(Wr[:5, :])
#                print(np.log(ass)[:5])
                lr = - np.sum(np.log(np.sum(np.multiply(Ar, self.D[:, self.Lsum[r]:self.Lsum[r+1]]), 1)))
                lD += lr
            lD /= self.n
            f += lD
            
#            print('lD:', lD, 'lG', lG)
                
            A = A - self.D
            # gradients as in pw model
            gradu = np.sum(A, 0) # Ltot by 1
            gradR += np.dot(self.Y.T, A) # dg by Ltot
            gradQ = np.dot(self.D.T, A) # this is Phihat from the doc, zero out diagonal and add transpose later
            
            # new gradients in CLZ model
            tmp_grad = 0.5 * np.dot(self.YsYt.T, A).reshape((self.dg, self.dg, self.Ltot))
            gradB -= tmp_grad # d_g^2 by Ltot, do not add transpose of this - appears only in 1 discrete node conditional
            for s in range(self.dg):
                gradbeta[s, :] -= tmp_grad[s, s, :]

            # scale gradients as likelihood
            gradQ *= 1/self.n 
            gradu *= 1/self.n 
            gradR *= 1/self.n 
            gradB *= 1/self.n 
            gradbeta *= 1/self.n 
            gradB0  *= 1/self.n 
            gradbeta0 *= 1/self.n            
            gradalpha *= 1/self.n 
    
        ## *** l1/l2 regularization ***
        if sparse: # iterate over groups
            dis_dis = 0; dis_cts=0; cts_cts=0
            # discrete - discrete, as in pairwise model
            for r in range(self.dc): # Phis
                for j in range(r):
#                    print(np.linalg.norm(Q[self.Lsum[r]:self.Lsum[r+1], self.Lsum[j]:self.Lsum[j+1]]))
                    wrj = self.weights[('dis_dis', r, j)] * self.lbda
                    fval, tmp_grad = _huberapprox(Q[self.Lsum[r]:self.Lsum[r+1], self.Lsum[j]:self.Lsum[j+1]], delta)
                    dis_dis += wrj * fval
                    gradQ[self.Lsum[r]:self.Lsum[r+1], self.Lsum[j]:self.Lsum[j+1]] += wrj * tmp_grad

            # continuous - discrete            
            for r in range(self.dc): # Rhos
                tmp_group = np.empty(self.sizes[r] *(1 + self.dg))
                for s in range(self.dg):
                    tmp_group[:self.sizes[r]] = R[s, self.Lsum[r]:self.Lsum[r+1]]
                    tmp_group[self.sizes[r] :] = B[s*self.dg:(s+1)*self.dg , self.Lsum[r]:self.Lsum[r+1]].flatten() # params la_{st}^{r:k} for t\in [d_g], k\in[L_r] 
                    wrs = self.weights[('dis_cts', s, r)] * self.lbda
                    fval, tmp_grad = _huberapprox(tmp_group, delta)
                    dis_cts += wrs * fval
                    gradR[s,self.Lsum[r]:self.Lsum[r+1]] += wrs * tmp_grad[:self.sizes[r]]
                    gradB[s, :, self.Lsum[r]:self.Lsum[r+1] ] += wrs * tmp_grad[self.sizes[r]:].reshape((self.dg,self.sizes[r]))

            # continuous - continuous
            gradB = gradB.reshape((self.dg**2, self.Ltot))
            
            tmp_group = np.empty(self.Ltot + 1)

            for t in range(self.dg): # upper triangle s<t
                for s in range(t):
                    tmp_group[:self.Ltot] = B[s*self.dg+t, :]
                    tmp_group[self.Ltot] = B0[s, t]
                    wst = self.weights[('cts_cts', s, t)] * self.lbda
                    fval, tmp_grad = _huberapprox(tmp_group, delta)
                    cts_cts += wst * fval
                    gradB[s*self.dg+t, :] += wst * tmp_grad[:self.Ltot]
                    gradB0[s, t] += wst * tmp_grad[self.Ltot]
#            print('dis_dis', dis_dis, 'dis_cts', dis_cts, 'cts_cts', cts_cts) # note different # of edges

#            if self.fcalls > 40:
#                print('f=%f, reg=%f, logsum=%f'%(f, rsum, sum([np.log(self.sigmas[s]) for s in range(self.dg)])))
            f += dis_dis + dis_cts + cts_cts

        # zero out diagonals and add transposes
        gradB = gradB.reshape((self.dg, self.dg, self.Ltot))
        for rk in range(self.Ltot):
            gradB[:, :, rk] -= np.diag(np.diag(gradB[:, :, rk]))
            gradB[:, :, rk] = np.triu(gradB[:, :, rk])+np.tril(gradB[:, :, rk]).T
        gradB0 -= np.diag(np.diag(gradB0))
        gradB0 = np.triu(gradB0)+np.tril(gradB0).T

        for r in range(self.dc): # set block-diagonal to zero
            gradQ[self.Lsum[r]:self.Lsum[r+1], self.Lsum[r]:self.Lsum[r+1]] = np.zeros((self.sizes[r], self.sizes[r]))
        gradQ = np.triu(gradQ) + np.tril(gradQ).T #np.triu(tmpPhihat) + np.tril(tmpPhihat).T to compare with gradient approximation


        grad = self.pack((gradQ, gradu, gradR, gradB0, gradbeta0, gradB, gradbeta, gradalpha))
#        print('f', f)
        return f, grad.reshape(-1)

    def crossvalidate(self, x):
        """crossvalidation of model with parameters in x using current data"""
        Q, u, R, B0, beta0, B, beta, alpha = self.preprocess(x)
        B = B.reshape((self.dg ** 2, self.Ltot)) # this is \tilde{B} (B tilde) from the doc

        dis_errors = np.zeros(self.dc)
        cts_errors = np.zeros(self.dg)

        e =  np.ones((self.n, 1))
        # TODO: the following is the same code as in get_fval_and_g. Smart way to remove redundancy? - extra func
        W = np.dot(self.Y, R) + np.dot(self.D, Q) + np.dot(np.ones((self.n, 1)), u.T) # n by Ltot, as in pw model      
        W += -0.5 * np.dot(self.YsYt, B) - 0.5 * np.dot(self.YsYs, beta) # new in CLZ  

        for r in range(self.dc): # as in pw model, see doc
            Lr = self.sizes[r]
            tmpexpWr = np.exp(W[:, self.Lsum[r]:self.Lsum[r+1]])
            Ar= np.divide(tmpexpWr, np.dot(tmpexpWr, np.ones((Lr, Lr)))) # matrix of conditional probabilities
            dis_errors[r] = self.n - np.sum(np.multiply(self.D[:, self.Lsum[r]:self.Lsum[r+1]], Ar) )
        b = np.dot(self.D, beta.T) + np.dot(e, beta0.T)# n by dg

        M = np.dot(e, alpha.T) + np.dot(self.D, R.T) - np.dot(self.Y, B0)# n by dg, part as in pairwise model
        for s in range(self.dg): # new in CLZ
            M[:, s] -= np.sum(np.multiply(self.Y, np.dot(self.D, B[s*self.dg:(s+1)*self.dg, :].T)), axis = 1)
        
        Delta = np.divide(M, b) - self.Y # regression residual

        cts_errors = np.sum(np.multiply(Delta, Delta), axis = 0)

        dis_errors /= self.n
        cts_errors /= self.n
        lval_testdata, grad = self.get_fval_and_grad(x, smooth = True, sparse = False)
        
        return dis_errors, cts_errors, lval_testdata
  