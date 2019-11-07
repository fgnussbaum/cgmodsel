# -*- coding: utf-8 -*-
"""
Copyright: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""

from cgmodsel.models.model_base import Model_Base, canon_to_meanparams

import numpy as np


###############################################################################

class Model_CLZ(Model_Base):
    def __init__(self, clz, meta):
        # meta must provided with dg, dc
        Model_Base.__init__(self)
        self.dg = meta['dg']
        self.dc = meta['dc']

        self.sizes = meta['sizes']
        self.Lcum = np.cumsum([0] + self.sizes)
        self.Ltot = np.sum(self.sizes)
        
        self.name = 'CLZ'

        self.u, self.Q, self.R, self.alpha, self.Lambda0, self.Lambdas = clz
    
    def __str__(self): 
        """string representation of the model"""
        s ='u:' + str(self.u) + '\nQ:\n' + str(self.Q) + '\nR:\n' + str(self.R) +\
         '\nalpha:' + str(self.alpha) + '\nLambda0:\n' + str(self.Lambda0)
        for r in range(self.dc):
            for k in range(1, self.sizes[r]): # for k=0 self.Lambdas[:, :, rk] is zero (identifiability constraint)
                rk = self.Lcum[r] + k
                s += '\nLambda_%d:%d'%(r, k) + str(self.Lambdas[:, :, rk])
        return s 

    def get_group_mat(self, diagonal=False, norm=True, aggr=True): # calibration? optional class param?
        # ~ functionValandGrad(smooth = False, sparse = True), includes calibration, gradient, accumulates..
        
        assert aggr==True, "only implemented for doing aggregation"
        assert norm==True, "l2-norm is the only implemented aggregation function"
        
        d = self.dc + self.dg
        grpnormmat = np.zeros((d, d))

        for r in range(self.dc): # dis-dis
            for j in range(r):
                grpnormmat[r,j] = np.linalg.norm(self.Q[self.Lcum[r]:self.Lcum[r+1], self.Lcum[j]:self.Lcum[j+1]])
        self.Lambdas = self.Lambdas.reshape((self.dg*self.dg, self.Ltot))
        
        for r in range(self.dc):
            tmp_group = np.empty(self.sizes[r] *self.dg)
            for s in range(self.dg):
                offset = s*self.dg
#                print(s, self.sizes[r])
                tmp_group[:self.sizes[r]] = self.R[s, self.Lcum[r]:self.Lcum[r+1]]
                tmp_group[self.sizes[r] : (s+1)*self.sizes[r]] = self.Lambdas[offset:offset+s , self.Lcum[r]:self.Lcum[r+1]].flatten() 
                tmp_group[(s+1)*self.sizes[r] :] = self.Lambdas[offset+s+1:offset+self.dg, self.Lcum[r]:self.Lcum[r+1]].flatten() 
#                print(tmp_group, np.linalg.norm(tmp_group))
                grpnormmat[self.dc+s, r] = np.linalg.norm(tmp_group)

        tmp_group = np.empty(self.Ltot + 1)
        for t in range(self.dg): # upper triangle s<t
            for s in range(t):
                tmp_group[:self.Ltot] = self.Lambdas[s*self.dg+t, :]
                tmp_group[self.Ltot] = self.Lambda0[s, t]
                grpnormmat[self.dc+s, self.dc+t] = np.linalg.norm(tmp_group)

        grpnormmat += grpnormmat.T
        
        if not diagonal:
            grpnormmat -= np.diag(np.diag(grpnormmat))
        else:
            for s in range(self.dg): # add diagonal of cts-cts interactions
                tmp_group[:self.Ltot] = self.Lambdas[s*self.dg+s, :]
                tmp_group[self.Ltot] = self.Lambda0[s, s]
                grpnormmat[self.dc+s, self.dc+s] = np.linalg.norm(tmp_group)
        
        self.Lambdas = self.Lambdas.reshape((self.dg, self.dg, self.Ltot))

#        print('CLZ.get_group_norm >>')
#        print(self.Lambda0)
#        print(self.Lambdas)
            
        return grpnormmat

    def get_params(self):
        return self.u, self.Q, self.R, self.alpha, self.Lambda0, self.Lambdas

    def get_meanparams(self):
        """convert CLZ parameters into mean parameter representation
           (p(x)_x, mu(x)_x, Sigma(x)_x)
        Note: Some arrays might be empty
              (if not both discrete and continuous variables are present)
              
        ** conversion formulas to mean parameters ** 
        p(x) ~ (2pi)^{n/2}|La(x)^{-1}|^{1/2}exp(q(x) + 1/2 nu(x)^T La(x)^{-1} nu(x) )
        mu(x) = La(x)^{-1}nu(x)
        Sigma(x) = La(x)^{-1}
        
        with nu(x) = alpha + R D_x and q(x) = u^T D_x + 1/2 D_x^T Q D_x
        and La(x) = Lambda0 + sum_r Lambda_r D_{x_r} = Lambda0 + Lambdas*D_x.
        Here D_x is the dummy representation of the categorical values in x.
        """
        assert self.dc + self.dg > 0
        
        if self.dc == 0:
            Sigma = np.linalg.inv(self.Lambda0)
            return np.empty(0), np.dot(Sigma, self.alpha), Sigma

        ## initialize mean params (reshape later)
        n_discrete_states = np.prod(self.sizes)
        q = np.zeros(n_discrete_states)
        
        ## discrete variables only
        if self.dg == 0:
            for x in range(n_discrete_states):
                unrvld_ind = np.unravel_index([x], self.sizes)
                # TODO: perhaps iter more systematically over dummy representations Dx
                Dx = np.zeros((self.Ltot,1))
                for r in range(self.dc): # construct dummy repr Dx of x
                    Dx[self.Lcum[r]+unrvld_ind[r][0], 0] = 1 
                q[x] = np.dot(self.u.T, Dx) + 0.5 * np.dot(Dx.T, np.dot(self.Q, Dx))
                canparams = q.reshape(self.sizes), np.empty(0), np.empty(0)
        else:
            precmatshape = (self.dg, self.dg)
            nus = np.empty((n_discrete_states, self.dg))
            Lambdas = np.empty((n_discrete_states, self.dg, self.dg))
    
            for x in range(n_discrete_states):
                unrvld_ind = np.unravel_index([x], self.sizes)
    
                Dx = np.zeros((self.Ltot,1))
                for r in range(self.dc): # construct full dummy repr Dx of x
                    Dx[self.Lcum[r]+unrvld_ind[r][0], 0] = 1
                q[x] = np.dot(self.u.T, Dx) + 0.5 * np.dot(Dx.T, np.dot(self.Q, Dx))
                nus[x, :] = (self.alpha + np.dot(self.R, Dx)).reshape(-1)
                Lambdas[x, :, :] = self.Lambda0 + np.dot(self.Lambdas, Dx).reshape(precmatshape)  
    
                # the precmat components are "cols" in <Lambdas>
                # (3rd component of the tensor),
                # the correct cols are selected by multiplying Dx
                # Lambdas is a 3D tensor.
                # np.dot with a vector from the right maps to last component

                tmp_eigvals = np.linalg.eigvals(Lambdas[x, :, :])
                tmp_min = np.min(tmp_eigvals)
    #            assert tmp_min > 0, 'Non-PD covariance of (flat) discrete state %d with la_min=%f. Pseudolikelihood estimation makes this possible. Note that nodewise prediction using this model still works (however it does not represent a valid joint distribution).'%(x, tmp_min)
                if tmp_min < 0:
                    print(x, Lambdas[x, :, :], tmp_min)
                    continue
            
            ## reshape to original forms
            q = q.reshape(self.sizes)
            nus = nus.reshape(self.sizes + [self.dg])
            Lambdas = Lambdas.reshape(self.sizes + [self.dg, self.dg])
            canparams = q, nus, Lambdas

        return canon_to_meanparams(canparams)
    
