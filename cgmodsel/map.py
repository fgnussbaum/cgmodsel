# Copyright (c) 2017-2019 Frank Nussbaum (frank.nussbaum@uni-jena.de)
"""
@author: Frank Nussbaum

CG model selection via MAP-estimation using mean parameters.
"""

import numpy as np
from numpy import ix_

class CG_MAP:
    """
    this class can be used to estimate maximum a-posteriori models
    for CG distributions that are parametrized using mean parameters
    """
    def __init__(self):
        """pass a dictionary that provides with keys dg, dc, and L"""
        
        self.meta = {}
        
        self.name = 'MAP'
    
    def drop_data(self, data, meta):
        """ drop data into the class
        note: discretedata is required in index form
        (i.e., convert labels to indices in data previously)"""
        
        if isinstance(data, tuple):
            assert len(data) == 2
            cat_data, cont_data = data
        else:
            counter = 0
            if 'n_cat' in meta and meta['n_cat'] > 0:
                counter += 1
                cat_data = data
                cont_data = np.empty((data.shape[0],0))
                assert 'sizes' in meta
                assert len(meta['sizes']) == meta['n_cat']
            if 'n_cg' in meta and meta['n_cg'] > 0:
                counter += 1
                cont_data = data
                cat_data = np.empty((data.shape[0],0))
            assert counter == 1, 'dictionary meta incompatible with provided data'

        self.meta = meta.copy()
        # check validity of dictionary meta
        if not 'n_cg' in meta:
            self.meta['n_cg'] = 0
        if not 'n_cat' in meta:
            self.meta['n_cat'] = 0
            
        self.cont_data = cont_data
        self.cat_data = cat_data

        assert self.dg == continuousdata.shape[1]
        assert self.dc == 0 or 1 == len(discretedata.shape), 'Is the discrete data in index Form?'

        self.n = 0
        self.sizes = meta['sizes']
        self.dg =  meta['dg'] # number of Gaussian variables
        self.dc = len(self.sizes) # number of discrete variables

        self.n = continuousdata.shape[0]
        
    def get_binomialprobs(self, mean_params):  # TODO: move to mean param class method
        """for binary-only models: calculate probabilities k outcomes are 1"""
        assert self.dg == 0, "Use on multivariate Bernoulli data only"
        for size in self.sizes:
            assert size == 2, "Use binary variables only"
        
        p, _, _ = mean_params        
        probs = np.zeros(self.dc + 1) # buckets for the probabilities of # ones that appear
        it = np.nditer(p, flags=['multi_index'])
        while not it.finished:
            probs[np.sum(it.multi_index)] += p[it.multi_index]
            it.iternext()
        
        return probs

    def get_name(self):
        return self.name
    
    def _fit_Gaussian(self, Vinv, nu, mu0, k0):
        """ fit Gaussian MAP-estimate
        
        Wishart prior for precision matrix
        nu    ... degrees of freedom
        Vinv  ... inverse of V where the prior is given by W(Lambda| V, nu)
        
        Gaussian prior for mean
        k0    ... number of artificial observations
        mu0   ... mean of prior N(mu | mu0, (k0 * Lambda)^{-1})
        
        Note: setting k0=0 (and nu = #Gaussians) produces ML estimate
        
        returns MAP estimates (mu, mat_sigma)
        """
        assert self.dc == 0, 'do not use this method in the presence of discrete variables'
        assert self.dg > 0 
        
        mu = np.sum(self.cont_data, axis=0) # sum over rows (axis = 0)
        mu = (k0 * mu0 + mu) / (k0 + self.n)

        mat_sigma = Vinv # this is V^{-1} from the doc
        for i in range(self.n): # add 'scatter matrix' of the evidence
            diffyi_muMAP = self.cont_data[i, :] - mu
            mat_sigma += np.outer(diffyi_muMAP, diffyi_muMAP)

        mudiff = mu - mu0
        mat_sigma +=  k0 * np.outer(mudiff, mudiff)
        mat_sigma /= self.n + nu - self.dg

        return mu, mat_sigma
                                  
    def fit_fixed_covariance(self, k=1, k0=1, nu=None, mu0=None, mat_sigma0=None):
        """fit MAP-CG model with a unique single covariance matrix and
            individual means for all conditional gaussians.

        ** components of the Dirichlet-Normal-Wishart prior **
        Dirichlet prior parameters (prior for discrete distribution)
        k     ... Laplacian smoothing parameter = artificial observations per discrete variable
                  (this is alpha in the doc)
        
        Gaussian prior parameters (prior for means of conditional Gaussians)
        k0    ... number of 'artificial' data points per conditional Gaussian 
                  (this is kpa in the doc)
        mu0   ... value of artificial data points
    
        Wishart prior parameters (prior for shared precision matrix of conditional Gaussians)
        nu     .. degrees of freedom
        mat_sigma0 .. initial guess for the covariance matrix
        
        returns MAP-estimate (p(x)_x, mu(x)_x, mat_sigma) where x are the discrete outcomes
        
        A (computational) warning: This method of model estimation uses
        sums over the whole discrete state space.
        """
        # future ideas:
        # (1) iter only over observed discrete examples, all other outcomes default to 'prior'
        #     (often times much less than the whole discrete state space)
        #     use dictionary + counts?
        # (2) use flag to indicate if cov is fixed or variable (avoid code redundancy)
        
        assert self.n > 0, 'No data loaded.. use method dropdata'

        ## defaults for smoothing parameters
        if mu0 is None:
            mu0 = np.zeros(self.dg) # reasonable when using standardized data Y
        assert mu0.shape == (self.dg, )
        if nu is None:
            nu = self.dg # least informative non-degenerate prior
        assert nu >= self.dg, 'degrees of freedom nu >= dg required to achieve non-degenerate prior' 
        if mat_sigma0 is None:
            mat_sigma0 = np.eye(self.dg) # standardized data --> variances are 1
        assert mat_sigma0.shape == (self.dg, self.dg)
        # choose V = 1/nu * mat_sigma0 as parameter for the Wishart prior ..
        Vinv = nu * np.linalg.inv(mat_sigma0) # formerly used self.dg instead of nu here
    
        ## MAP-estimate Gaussians only (with unknown mean and covariance)
        if self.dc == 0: 
            mu, mat_sigma = self._fit_Gaussian(Vinv, nu, mu0, k0)
            return np.array([]), mu, mat_sigma

        ## MAP-estimation in the presence of discrete variables
        n_discrete_states = int(np.prod(self.sizes))

        p = np.zeros(n_discrete_states) 
        mus = np.zeros((n_discrete_states, self.dg))
        mat_sigma = np.zeros((self.dg, self.dg))
    
        ## mu and p
        for i, x in enumerate(self.cat_data):
            p[x] += 1
            mus[x, :] += self.cont_data[i, :]
    
        ## MAP-estimates of mu(x)
        for x in range(n_discrete_states):
            mus[x, :] = (k0 * mu0 + mus[x, :]) / (k0 + p[x]) # MAP estimator for mu(x)

        ## MAP-estimate of mat_sigma
        mat_sigma = Vinv # this is V^{-1} from the doc
        for i, x in enumerate(self.cat_data): # add 'scatter matrix' of the evidence
            diffyi_muMAP = self.cont_data[i, :] - mus[x,:]
            mat_sigma += np.outer(diffyi_muMAP, diffyi_muMAP) 
    
        for x in range(n_discrete_states): # add scatter part of artificial observations mu0
            mudiff = mus[x, :] - mu0
            mat_sigma +=  k0 * np.outer(mudiff, mudiff)
        mat_sigma /= nu + self.n - self.dg - 1 + n_discrete_states
    
        ##  MAP-estimate of p
        p = (p + k) / (p.sum() + k * p.size) # wo. smoothing would be pML = p/n  

        ## reshape to the correct shapes
        p = p.reshape(self.sizes)
        mus = mus.reshape(self.sizes+[self.dg])
            
        return p, mus, mat_sigma

    def fit_variable_covariance(self, k=1, k0=1, nu=None, mu0=None, mat_sigma0=None):
        """fit MAP-CG model with  individual covariance matrices
        and means for all conditional gaussians.
        
        ** Components of the Dirichlet-Normal-Wishart Prior **
        Dirichlet prior parameters (prior for discrete distribution)
        k     ... Laplacian smoothing parameter = artificial observations per discrete variable
                  (this is alpha in the doc)
        
        Gaussian prior parameters (prior for means of conditional Gaussians)
        k0    ... number of 'artificial' data points per conditional Gaussian 
                  (this is kpa in the doc)
        mu0   ... value of artificial data points
    
        Wishart prior parameters (prior for shared precision matrix of conditional Gaussians)
        nu     .. degrees of freedom
        mat_sigma0 .. initial guess for the covariance matrices mat_sigma(x)
        
        returns MAP-estimate (p(x)_x, mu(x)_x, mat_sigma(x)_x) where x are the discrete outcomes
        
        A (computational) warning: This method of model estimation uses
        sums over the whole discrete state space.
        """
        assert self.n > 0, 'No data loaded.. use method dropdata'

        ## defaults for smoothing parameters
        if mu0 is None:
            mu0 = np.zeros(self.dg) # reasonable when using standardized data Y
        assert mu0.shape == (self.dg, )
        if nu is None:
            nu = self.dg + 1 # yields mat_sigma(x)=mat_sigma_0 if x not observed
        assert nu >= self.dg+1, 'degrees of freedom nu >= dg+1 required to achieve non-degenerate prior and deal with unobserved discrete outcomes' 
        if mat_sigma0 is None:
            mat_sigma0 = np.eye(self.dg)
        assert mat_sigma0.shape == (self.dg, self.dg)
        # choose V = 1/nu * mat_sigma0 as parameter for the Wishart prior
        # then prior mean of W(Lambda(x)|V, nu) is nu*V= mat_sigma0
        Vinv = nu * np.linalg.inv(mat_sigma0) # formerly used self.dg instead of nu here

        ## MAP-estimate Gaussians only (with unknown mean and covariance)
        if self.dc == 0: 
            mu, mat_sigma = self._fit_Gaussian(Vinv, nu, mu0, k0)
            return np.array([]), mu, mat_sigma
        
        ## initialization
        n_discrete_states = int(np.prod(self.sizes)) 
        p = np.zeros(n_discrete_states) 
        mus = np.zeros((n_discrete_states, self.dg))
        sigmas = np.zeros((n_discrete_states, self.dg, self.dg))
    
        ## mu and p
        for i, x in enumerate(self.cat_data):
            p[x] += 1
            mus[x, :] += self.cont_data[i, :]
    
        ## MAP-estimates of mu(x)
        for x in range(n_discrete_states):
            mus[x, :] = (k0 * mu0 + mus[x, :]) / (k0 + p[x]) # MAP estimator for mu(x)

        ## MAP-estimate of mat_sigma(x)
        for i, x in enumerate(self.cat_data):
            diffyi_muMap = self.cont_data[i, :] - mus[x,:]
            sigmas[x, :, :] += np.outer(diffyi_muMap, diffyi_muMap) # scatter matrix of the evidence
    
        for x in range(n_discrete_states): 
            mudiff = mus[x, :] - mu0
            sigmas[x, :, :] += Vinv + k0 * np.outer(mudiff, mudiff)
            sigmas[x, :, :] /= p[x] - self.dg + nu # > 0 since nu > self.dg
    
        ## MAP-estimate of p
        p = (p + k) / (p.sum() + k * p.size)
        
        ## reshape to the correct shapes
        p = p.reshape(self.sizes)
        mus = mus.reshape(self.sizes+[self.dg])
        sigmas = sigmas.reshape(self.sizes + [self.dg, self.dg])
        
        return p, mus, sigmas

    def get_PLHvalue(self, mean_params):
        """ returns pseudo-likelihood value of current data set """
        e1, e2, lval = self.crossvalidate(mean_params) # few redundant computations
        return lval

    def crossvalidate(self, mean_params):
        """
        perform crossvalidation
        mean_params ... tuple (p, mus, sigmas)
        p:      has shape sizes
        mus:    has shape sizes +(dg)
        sigmas: has shape (dg,dg) if independent of discrete variables x
                else has shape sizes + (dg, dg) if dependent on x

        """
        p, mus, sigmas = mean_params
        if len(sigmas.shape)==2:
            cov_variable = 0 # use zero to scale indices and toggle between modes of covariance
        else:
            cov_variable = 1
        
        ## initialization
        n_discrete_states = int(np.prod(self.sizes))  # this is 1 if dc==0 since np.prod([])=1
        n_covmats = (n_discrete_states - 1 ) * cov_variable + 1 # number of different covariance matrices
        dis_errors = np.zeros(self.dc)
        cts_errors = np.zeros(self.dg)
        
        lval_testdata = 0 # likelihood value
    
        ## reshape by collapsing discrete dimensions
        shapes = (p.shape, mus.shape, sigmas.shape) # store shapes
        p= p.reshape(-1)
        mus = mus.reshape((n_discrete_states, self.dg))
        sigmas = sigmas.reshape((n_covmats, self.dg, self.dg))

        ## discrete only models
        if self.dg == 0:
            assert self.dc > 0
            for i in range(self.n):
                x = self.cat_data[i] # flat index, or empty list if dc==0
                cat = list(np.unravel_index(x, self.sizes)) # multiindex of discrete outcome
    
                for r in range(self.dc):
                    probs = np.empty(self.sizes[r])
                    h = cat[r]
                    for k in range(self.sizes[r]): # calculate conditional probs
                        cat[r] = k
                        ind = np.ravel_multi_index(tuple(cat), self.sizes)
                        probs[k] = p[ind] # this unnormalized at this point 
                    cat[r] = h
                    prob_xr = probs[h]/np.sum(probs) # normalize
                    dis_errors[r] += 1 - prob_xr
                    lval_testdata -= np.log(prob_xr) # note that log(x) ~ x-1 around 1

            return dis_errors, cts_errors, lval_testdata
        
        ## setting with Gaussian variables
        ## precalculate determinants, inverse matrices (full, and with dim recuced by 1)
        dets = np.empty(n_covmats)
        sigmas_inv = np.empty((n_covmats, self.dg, self.dg))
        sigmas_red_inv = np.empty((n_covmats, self.dg, self.dg - 1, self.dg - 1)) 
        
        for x in range(n_covmats):
            dets[x] = np.linalg.det(sigmas[x, :, :]) ** (-0.5)
            sigmas_inv[x, :, :] = np.linalg.inv(sigmas[x, :, :])

            # for each x, s: store mat_sigma[x]_{-s, -s}^{-1}
            cond_inds = list(range(1,self.dg)) # indices to keep
            for s in range(self.dg): # reduced det of cov with rows and col s deleted
                S = sigmas[x, :, :][ix_(cond_inds, cond_inds)] # this is not a view
                sigmas_red_inv[x, s, :, :] = np.linalg.inv(S)
                if s <self.dg - 1:
                    cond_inds[s] -= 1 # include index s and remove index s+1

        ## cross validation
        for i in range(self.n):
            yi = self.cont_data[i, :]
            
            ## discrete ##
            if self.dc > 0:
                x = self.cat_data[i] # flat index, or empty list if dc==0
                covindex = x * cov_variable
                cat = list(np.unravel_index(x, self.sizes)) # multiindex of discrete outcome
 
                for r in range(self.dc):
                    probs = np.empty(self.sizes[r])
                    exps = np.empty(self.sizes[r])
                    h = cat[r]
                    for k in range(self.sizes[r]): # calculate all conditional probs
                        cat[r] = k
                        ind = np.ravel_multi_index(tuple(cat), self.sizes)

                        y_mu = yi - mus[ind]
                        exps[k] = -0.5 * np.dot(np.dot(y_mu.T, sigmas_inv[ind*cov_variable, :, :]), y_mu)
                        probs[k] = p[ind] * dets[ind*cov_variable]
                    cat[r] = h
                    exps = np.exp(exps - np.max(exps)) # stable exp (!)
                    probs = np.multiply(probs, exps) # unnormalized (!)
                    
                    prob_xr = probs[h] / np.sum(probs)
                    dis_errors[r] += 1 - prob_xr
                    lval_testdata -= np.log(prob_xr) # note that log(x) ~ x-1 around 1
            else: # no discrete variables: set dimension to zeros
                x = 0
                covindex = 0
            ## continuous ##
            cond_inds = list(range(1,self.dg)) # list of indices to keep for Schur complement
            for s in range(self.dg):
                # p(y_s|..) = sqrt(1/(2pi))*|Schur|^{-1/2}*exp(-.5*(y_s-mu)var_s^{-1}(y_s-mu))
                # since by fixing x this is a node conditional of a purely Gaussian model
                y_s_mu_s = yi[ix_(cond_inds)] - mus[x][ix_(cond_inds)]
                mat_sigma_x_OH = sigmas[covindex, :, :][ix_([s], cond_inds)] # O = s, H = -s, 1 by dg-1
                tmp_expr = np.dot(mat_sigma_x_OH, sigmas_red_inv[covindex, s, :, :])
                mu_hat = mus[x][s] + np.dot(tmp_expr, y_s_mu_s)
                if s < self.dg - 1:
                    cond_inds[s] -= 1 # include index s and remove index s+1
                
                residual = yi[s] - mu_hat
    
                cts_errors[s] += (residual) ** 2 # squared residual
                
                var_s = sigmas[covindex, :, :][s, s] - np.dot(tmp_expr, mat_sigma_x_OH.T) # precalculate schur complements?
                lval_testdata +=  0.5 * residual ** 2 / var_s + 0.5 * np.log(var_s) 
                # TODO: what about the constant pi part? -- ok iff left out everywhere
            
        ## reshape parameters to original form
        p.reshape(shapes[0])
        mus.reshape(shapes[1])
        sigmas.reshape(shapes[2])
                
        dis_errors /= self.n
        cts_errors /= self.n
        lval_testdata /= self.n
    
        return dis_errors, cts_errors, lval_testdata
    