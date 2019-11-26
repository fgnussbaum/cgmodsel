# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

###################################### model types

GAUSSIAN = 'Gaussian'
BINARY = 'Binary'
DISCRETE = 'Discrete'
CG = 'CG'

def get_modeltype(dc, dg, sizes):
    if dc == 0:
        modeltype = GAUSSIAN
    elif dg == 0:
        binary = True
        for no_labels in sizes:
            if no_labels != 2:
                binary = False
        if binary:
            modeltype = BINARY
        else:
            modeltype = DISCRETE
    else:
        modeltype = CG
    return modeltype


#import itertools

def _invert_indices(l, d):
    """return all indices from d that are not in l as a list """
    inv = []
    for j in range(d):
        if j not in l:
            inv.append(j)
    return inv

def _schur_complement_SL(M, keep_idx=None, drop_idx=None):
    """Returns S and L from upper Schur complement S-L of array_like M
    where the 'upper block' is indexed by keep_idx and the lower block by drop_idx
    (only one list should be given).
    """
    ## derive index lists
    if keep_idx is None and drop_idx is None: 
        raise
    if not keep_idx is None:
        i = keep_idx
        j = _invert_indices(i, M.shape[0])
    else:
        j = drop_idx
        i = _invert_indices(j, M.shape[0])
#        print('hidden indices',drop_idx, j)
#    print(M[np.ix_(j, j)])
    
    S = M[np.ix_(i, i)]
    L = np.dot(M[np.ix_(i, j)], 
                 np.dot(np.linalg.inv(M[np.ix_(j, j)]), M[np.ix_(j, i)]) )

    return S, L

######################################
    
def _split_Theta(Theta, Ltot):
    """split pairwise interaction parameter matrix into components"""
    Q = Theta[:Ltot, :Ltot]
    R = Theta[Ltot:, :Ltot]
    Lambda = -Theta[Ltot:, Ltot:]
    return Q, R, Lambda

def _Theta_from_components(Q, R, Lambda):
    """
    stack components together into pairwise interaction matrix of CG model
    
    Q ... discrete - discrete interactions
    R ... continuous - discrete interactions
    Lambda ... continuous - continuous interactions
    """
    Ltot = Q.shape[0]
    dg = Lambda.shape[0]
    d = Ltot + dg
    Theta = np.empty((d, d))
    if Ltot > 0:
        Theta[:Ltot, :Ltot] = Q
        Theta[Ltot:, :Ltot] = R
        Theta[:Ltot, Ltot:] = R.T
    Theta[Ltot:, Ltot:] = -Lambda
    return Theta

def pad(arr, sizes, rowsonly=False):
    """ padding for pairwise parameter matrix or univariate parameter vector u
    (adds rows/columns of zeros for the 0-th levels, 
    they correspond to interactions parameters set to 0 for identifiability
    reasons)
    
    sizes ... list of number of levels for each discrete variable
    rowsonly ... if True, pad only rows"""
    Ltot = sum(sizes)
    dc = len([x for x in sizes if x > 0])
    sizes_cum = np.cumsum([0] + sizes)

    d_unpadded = arr.shape[0]
    dg = d_unpadded - Ltot + dc
    
    rows_discrete_indices = _invert_indices(sizes_cum[:-1], Ltot)
    rows_gauss_indices = list(range(Ltot, Ltot + dg))
    row_indices = rows_discrete_indices + rows_gauss_indices
#    print(rows_discrete_indices)
#    print(rows_gauss_indices)
    
    if len(arr.shape) == 2:
        if rowsonly:
            col_indices = list(range(Ltot - dc + dg))
#            cols_discrete_indices = list(range(Ltot-dc))
#            cols_gauss_indices = list(range(Ltot-dc, Ltot-dc+dg))
            newarr = np.zeros((d_unpadded + dc, d_unpadded))
        else:
            col_indices = row_indices
#            cols_discrete_indices = rows_discrete_indices
#            cols_gauss_indices = rows_gauss_indices
            newarr = np.zeros((d_unpadded + dc, d_unpadded + dc))
    #    print(insert_indices)
        
        newarr[np.ix_(row_indices,col_indices)] = arr

        return newarr
    else:
        newu = np.zeros(d_unpadded + dc)
#        print(newu[np.ix_(rows_discrete_indices)])
#        print('arr', arr, rows_discrete_indices)
        newu[np.ix_(rows_discrete_indices)] = arr
        return newu

def unpad(Theta, sizes):
    """unpad pairwise parameter matrix
    sizes ... list of number of levels for each discrete variable"""
    sizes_cum = np.cumsum([0] + sizes)

    if len(Theta.shape) == 2:
        Theta = np.delete(Theta, sizes_cum[:-1], axis=1)  # delete rows
    return np.delete(Theta, sizes_cum[:-1], axis=0) # delete columns


###############################################################################
# Parameter Conversion
###############################################################################
def mean_to_canonparams(meanparams):
    """
    convert mean parameters to canonical parameters
    
    meanparameters is a tupe (p, mus, Sigmas), 
    where Sigmas is the covariance for all conditional distributions p(y|x) 
    for pairwise model -> 2D array, and Sigmas is a 3D array if the covariances
    are different for p(y|x), in this case Sigmas(x, :, :) is the covariance of
    p(y|x)
    """
    p, mus, Sigmas = meanparams

    sizes = p.shape
    if len(sizes) == 1 and sizes[0] == 0:
        sizes = []
    dc = len(sizes)
    dg = Sigmas.shape[-1]

    if dg == 0:
        q = np.log(p) - dg / 2 * np.log(2 * np.pi)
        return q.reshape(sizes), np.empty(0), np.empty(0)
    if dc == 0:
        Lambdas = np.linalg.inv(Sigmas)
        nus = np.dot(Lambdas, mus)
        return np.empty(0), nus, Lambdas
    
    if len(Sigmas.shape) == 2:
        cov_variable = 0 # use zero to scale indices and toggle between modes of covariance
    else:
        cov_variable = 1
        
    n_discrete_states = int(np.prod(sizes)) # prod([]) is 1.0
    n_covmats = (n_discrete_states - 1 ) * cov_variable + 1 # number of different covariance matrices

    shapes = mus.shape, Sigmas.shape # store shapes
    p = p.reshape(-1)
    mus = mus.reshape((n_discrete_states, dg))
    Sigmas = Sigmas.reshape((n_covmats, dg, dg))

    ## initialization    
    q = np.empty(n_discrete_states)    
    nus = np.empty(mus.shape)
    Lambdas = np.empty(Sigmas.shape)

    ## precompute determinants from normalization and Lambdas
    dets = np.empty(n_covmats)
    for x in range(n_covmats):
        Lambdas[x, :, :] = np.linalg.inv(Sigmas[x, :, : ])
        dets[x] = np.linalg.det(Sigmas[x, :, :])
    
    for x in range(n_discrete_states):
        covindex = x * cov_variable
        mu_x = mus[x, :]
        q[x] = np.log(p[x]) - dg / 2 * np.log(2 * np.pi) \
            - 0.5 * np.log(dets[covindex]) \
            - 0.5 * np.dot(np.dot(mu_x.T, Lambdas[covindex, :, :]), mu_x)

        nus[x] = np.dot(Lambdas[covindex, :, :], mu_x)
        
    q = q.reshape(sizes)
    nus = nus.reshape(shapes[0])
    Lambdas = Lambdas.reshape(shapes[1])

    return q, nus, Lambdas
            
def canon_to_meanparams(canparams):
    """
    convert canonical parameters to mean parameters
    
    canparams is a tupe (q, nus, Lambdas), 
    where Lambdas is the precision for all conditional distributions p(y|x) 
    for pairwise model -> 2D array, and Lambdas is a 3D array if the precision matrices
    are different for p(y|x), in this case Lambdas(x, :, :) is the precision of
    p(y|x)
    """
    q, nus, Lambdas = canparams
    
    sizes = q.shape
    if len(sizes) == 1 and sizes[0] == 0:
        sizes = []
    dc = len(sizes)
    dg = Lambdas.shape[-1]

    if dg == 0:
        p = np.exp(q - np.max(q))
        p /= np.sum(p)
        return p.reshape(sizes), np.empty(0), np.empty(0)
    if dc == 0:
        Sigmas = np.linalg.inv(Lambdas)
        mus = np.dot(Lambdas, nus)
        return np.empty(0), nus, Sigmas
    
    if len(Lambdas.shape) == 2:
        cov_variable = 0 # use zero to scale indices and toggle between modes of covariance
    else:
        cov_variable = 1
        
    n_discrete_states = int(np.prod(sizes)) # prod([]) is 1.0
    n_covmats = (n_discrete_states - 1 ) * cov_variable + 1 # number of different covariance matrices

    ## reshape by collapsing discrete dimensions
    shapes = q.shape, nus.shape, Lambdas.shape # store shapes
    q= q.reshape(-1)
    nus = nus.reshape((n_discrete_states, dg))
    Lambdas = Lambdas.reshape((n_covmats, dg, dg))
  
    p = np.zeros(n_discrete_states)
    mus = np.zeros(nus.shape)
    Sigmas = np.empty(Lambdas.shape)
    
    ## precompute determinants from normalization and Sigmas
    dets = np.empty(n_covmats)  
    for x in range(n_covmats):
        Sigmas[x, :, :] = np.linalg.inv(Lambdas[x, :, : ])
        dets[x] = np.linalg.det(Sigmas[x, :, :])

    Qxstore = np.empty(n_discrete_states)
    for x in range(n_discrete_states):
        covindex = x * cov_variable
        nu_x = nus[x, :]
        q_x = q[x]
        mu_x = np.dot(Sigmas[covindex, :, :], nu_x)
        mus[x, :] = mu_x.reshape(-1)

        Qxstore[x] = q_x + 0.5 * np.einsum('i,i', nu_x, mu_x)
        p[x] =  dets[covindex] ** 0.5 # omitted constant part (later normalize anyway)

    Qxstore = np.exp(Qxstore - np.max(Qxstore)) # save exp
    # note: normalization is done later
    p = np.multiply(p, Qxstore)
    p /= np.sum(p)
    
    p = p.reshape(shapes[0])
    mus = mus.reshape(shapes[1])
    Sigmas = Sigmas.reshape(shapes[2])
    
    return p, mus, Sigmas

###############################################################################

class Model_Base:
    def __init__(self):
        self.dl = 0 # number of latent variables
        self.annotations = {} # textual information about the model (e.g. how it was learned)
    
    def get_numberofedges(self, threshold=10E-2):
        """ return number of edges in the graph"""
        graph = self.get_graph(threshold=threshold)
        return int(np.sum(graph) / 2)

    def get_group_mat(self):
        raise NotImplementedError # Implement in derived class
        
    def get_meta(self):
        """get dictionary of meta data"""
        return {'dg': self.dg, 'dc': self.dc, 'dl': self.dl, 'Ltot': self.Ltot,
                'sizes': self.sizes, 'Lcum': self.Lcum}
    
    def get_params(self):
        raise NotImplementedError # Implement in derived class

    def repr_graphical(self, diagonal=True, graph=False,
                       threshold=None, caption='', norm=True,
                       samecolorbar=True, save=False, vmax=None):
        """ a graphical representation of the groupnorm for PW and CLZ models
        (this method is overwritten by the Model_PWSL model class)
        
        experimental code"""
        grpnormat = self.get_group_mat(diagonal=diagonal, norm=norm)
        vmin = np.min(grpnormat);
        if vmax is None:
            vmax = np.max(grpnormat)

        if not norm:
            cmap = cm.coolwarm
            # center colormap at zero
            m = max((np.abs(vmin), vmax))
            vmin = -m
            vmax = m
        else:
            cmap = cm.viridis
            cmap = cm.Greys
        if graph:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
            if not threshold is None:
                modelgraph = self.get_graph(threshold=threshold)
            else:
                modelgraph = self.get_graph()
            axes.flat[0].matshow(modelgraph * np.max(grpnormat), cmap=cmap,
                                 interpolation='nearest')
            axes.flat[0].set_title('%s graph'%(self.name), y=1.10)

            im0 = axes.flat[1].matshow(grpnormat, interpolation='nearest',
                           cmap=cmap, vmin=vmin, vmax=vmax)
            axes.flat[1].set_title('%s (%s) edge weights'%(self.name, caption), y=1.10)
            fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=.7)
        else:
            fig = plt.figure(figsize=(6, 5))
            axes = fig.gca()
    
            im0 = axes.matshow(grpnormat, interpolation='nearest',
                               vmin=vmin, vmax=vmax, cmap=cmap)
            axes.set_title('%s pairwise parameters %s' % (self.name, caption), y=1.10)

            fig.colorbar(im0, ax=axes, shrink=.7)
        if save:
            fig.savefig('savedmodel.pdf', bbox_inches='tight')
        
        plt.show()


###############################################################################
       
class Model_PW_Base(Model_Base):
    """
    this class specializes to standard PW model and S+L model,
    see the classes Model_PW and Model_PWSL
    """
    def __init__(self, pw, meta, annotations={}, in_padded=True):
        # TODO: pw params in dict, allow passing of pwmat
        Model_Base.__init__(self)
        
        self.annotations.update(annotations)
        # meta must provide with dg, dc, sizes
        self.dg = meta['dg']
        self.dc = meta['dc']

        self.sizes = meta['sizes']
        self.Ltot = int(np.sum(self.sizes)) # np.sum([]) is 0.0
        self.Lcum = np.cumsum([0] + self.sizes)
            
        self.modeltype = get_modeltype(self.dc, self.dg, self.sizes)
        
        if in_padded:
            self.u, self.Q, self.R, self.alpha, self.Lambda = pw
            self.u = self.u.flatten()     
        else: # do padding
            u, Q, R, self.alpha, Lambda = pw
            
            self.u = pad(u.flatten(), self.sizes)

            Theta = _Theta_from_components(Q, R, Lambda) # Lambda with neg sign
            Theta = pad(Theta, self.sizes)
            self.Q, self.R, self.Lambda = _split_Theta(Theta, self.Ltot)

        self.alpha = self.alpha.flatten()

    def _get_group_mat(self, Q, R, Lambda,
                           diagonal=False, aggr=True, norm=True):
        """
        calculate group aggregations, e.g., the l_2 norms of all groups 
        
        diagonal ... if False and aggr=True, then exclude groups
                     of univariate parameters (that are on the diagonal)
        
        aggr ... if True, aggregate values of group into single value
        
        norm ... if True, use np.linalg.norm for aggregation, else use np.sum
        """
        if not aggr: # do not use an aggregation functions for the groups
#            i_old = 0
#            for r, i in enumerate(self.Lcum[1:]): # dis-dis
#                j_old = 0
#                for j in self.Lcum[1:]:
#                    aggrmat[i_old:i, j_old:j] = Q[i_old:i, j_old:j]
#                    j_old = j
#
#                for s in range(self.dg):
#                    aggrmat[self.Ltot+s, i_old:i] = aggrmat[r, self.dc+s] \
#                        = R[s, i_old:i]
#                i_old = i
            return _Theta_from_components(Q, R, Lambda)
        else:
            if norm:
                f = np.linalg.norm
            else:
                f = np.sum
                
            d = self.dc + self.dg
            aggrmat = np.zeros((d, d))
            
            for r in range(self.dc): # dis-dis
                for j in range(self.dc):
                    aggrmat[r,j] = f(Q[self.Lcum[r]:self.Lcum[r+1], self.Lcum[j]:self.Lcum[j+1]])

                for s in range(self.dg):
                    aggrmat[self.dc+s, r] = aggrmat[r, self.dc+s] \
                        = f(R[s, self.Lcum[r]:self.Lcum[r+1]])
            if norm:
                aggrmat[self.dc:, self.dc:] = np.abs(Lambda)
            else:
                aggrmat[self.dc:, self.dc:] = -Lambda
        
        if not diagonal:
            aggrmat -= np.diag(np.diag(aggrmat))
        
        return aggrmat

    def get_group_mat(self, **kwargs):
        """ returns matrix of group norms of the direct interactions"""
        return self._get_group_mat(self.Q, self.R, self.Lambda, **kwargs)

    def get_graph(self, threshold=1e-1):
        """ calculate group norms of the parameters associated with each edge and threshold
        
        plot graph if disp is True"""
        # perhaps do variable threshold for different group types
        grpnormmat = self._get_group_mat(self.Q, self.R, self.Lambda,
                                         diagonal=False, norm=True)
        graph = grpnormmat > threshold
        
        for i in range(graph.shape[0]):
            graph[i, i] = False # no edges on diagonal

        return graph

    def _get_meanparams(self, pwparams):
        """
        convert pairwise parameters to mean parameters
        p(x) ~ (2pi)^{n/2}|La^{-1}|^{1/2}exp(q(x) + 1/2 nu(x)^T La^{-1} nu(x) )
        mu(x) = La^{-1}nu(x)
        Sigma = La^{-1}
        
        with nu(x) = alpha + R D_x and q(x) = u^T D_x + 1/2 D_x^T Q D_x.
        Here D_x is the dummy representation of the categorical values in x.
        """
        u, Q, R, alpha, Lambda = pwparams
        assert u.shape == (self.Ltot, )
        assert alpha.shape == (self.dg, )

        if self.dc == 0 and self.dg == 0:
            raise "Empty model"
        
        if self.dc == 0:
            Sigma = np.linalg.inv(Lambda)
            return np.empty(0), np.dot(Sigma, alpha), Sigma # TODO: types

        n_discrete_states = np.prod(self.sizes) # np.prod([]) would be 1.0, but dc>0 here

        q = np.empty(n_discrete_states)
        nus = np.zeros((n_discrete_states, self.dg) )

        for x in range(n_discrete_states):
            unrvld_ind = np.unravel_index([x], self.sizes)

            Dx = np.zeros(self.Ltot)
            for r in range(self.dc):
                Dx[self.Lcum[r]+unrvld_ind[r][0]] =1 # dummy repr
            nus[x, :] = alpha + np.dot(R, Dx)
            q[x] = np.inner(u, Dx) + 0.5 * np.inner(Dx, np.dot(Q, Dx)) 

        return canon_to_meanparams((q.reshape(self.sizes), nus, Lambda))

    def get_Thetaalpha(self, **kwargs):
        """return compound pairwise interaction parameter matrix and alpha,
        that is, univariate continuous parameter"""
        Theta = self.get_pw_mat(**kwargs)
        return Theta, self.alpha

    def get_pw_mat(self, padded=True, addunivariate=False):
        """
        returns the pairwise matrix of direct interaction parameters.
        in the case of the standard pairwise model this is the matrix of all
        pairwise interaction parameters
        
        padded ... Boolean to control whether return matrix is padded with zero rows/cols (corresponding to first levels of discrete variables)
        
        precmatsign ... sign of the cts-cts interaction, use negative sign to obtain valid representation of Schur complement
        """

        X = _Theta_from_components(self.Q, self.R, self.Lambda) # inverts sign of Lambda

        if addunivariate:
            assert len(self.u.shape) <= 1
            X[:self.Ltot, :self.Ltot] += 2 * np.diag(self.u)
        if not padded:
            return unpad(X, self.sizes)

        return X