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
    def __init__(self, meta):
        """pass a dictionary that provides with keys dg, dc, and L"""
        self.n = 0
        self.sizes = meta['sizes']
        self.dg =  meta['dg'] # number of Gaussian variables
        self.dc = len(self.sizes) # number of discrete variables
        
        self.name = 'MAP'
    
    def drop_data(self, discretedata, continuousdata):
        """ drop data into the class
        note: discretedata is required in index form
        (i.e., convert labels to indices in data previously)"""
        assert self.dg == continuousdata.shape[1]
        assert self.dc == 0 or 1 == len(discretedata.shape), 'Is the discrete data in index Form?'
        
        self.D = discretedata
        self.Y = continuousdata
        self.n = continuousdata.shape[0]
        
    def get_binomialprobs(self, meanParams):  # TODO: move to mean param class method
        """for binary-only models: calculate probabilities k outcomes are 1"""
        assert self.dg == 0, "Use on multivariate Bernoulli data only"
        for size in self.sizes:
            assert size == 2, "Use binary variables only"
        
        p, _, _ = meanParams        
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
        
        returns MAP estimates (mu, Sigma)
        """
        assert self.dc == 0, 'do not use this method in the presence of discrete variables'
        assert self.dg > 0 
        
        mu = np.sum(self.Y, axis=0) # sum over rows (axis = 0)
        mu = (k0 * mu0 + mu) / (k0 + self.n)

        Sigma = Vinv # this is V^{-1} from the doc
        for i in range(self.n): # add 'scatter matrix' of the evidence
            diffyi_muMAP = self.Y[i, :] - mu
            Sigma += np.outer(diffyi_muMAP, diffyi_muMAP)

        mudiff = mu - mu0
        Sigma +=  k0 * np.outer(mudiff, mudiff)
        Sigma /= self.n + nu - self.dg

        return mu, Sigma
                                  
    def fit_fixed_covariance(self, k=1, k0=1, nu=None, mu0=None, Sigma0=None):
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
        Sigma0 .. initial guess for the covariance matrix
        
        returns MAP-estimate (p(x)_x, mu(x)_x, Sigma) where x are the discrete outcomes
        
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
        if Sigma0 is None:
            Sigma0 = np.eye(self.dg) # standardized data --> variances are 1
        assert Sigma0.shape == (self.dg, self.dg)
        # choose V = 1/nu * Sigma0 as parameter for the Wishart prior ..
        Vinv = nu * np.linalg.inv(Sigma0) # formerly used self.dg instead of nu here
    
        ## MAP-estimate Gaussians only (with unknown mean and covariance)
        if self.dc == 0: 
            mu, Sigma = self._fit_Gaussian(Vinv, nu, mu0, k0)
            return np.array([]), mu, Sigma

        ## MAP-estimation in the presence of discrete variables
        n_discrete_states = int(np.prod(self.sizes))

        p = np.zeros(n_discrete_states) 
        mus = np.zeros((n_discrete_states, self.dg))
        Sigma = np.zeros((self.dg, self.dg))
    
        ## mu and p
        for i, x in enumerate(self.D):
            p[x] += 1
            mus[x, :] += self.Y[i, :]
    
        ## MAP-estimates of mu(x)
        for x in range(n_discrete_states):
            mus[x, :] = (k0 * mu0 + mus[x, :]) / (k0 + p[x]) # MAP estimator for mu(x)

        ## MAP-estimate of Sigma
        Sigma = Vinv # this is V^{-1} from the doc
        for i, x in enumerate(self.D): # add 'scatter matrix' of the evidence
            diffyi_muMAP = self.Y[i, :] - mus[x,:]
            Sigma += np.outer(diffyi_muMAP, diffyi_muMAP) 
    
        for x in range(n_discrete_states): # add scatter part of artificial observations mu0
            mudiff = mus[x, :] - mu0
            Sigma +=  k0 * np.outer(mudiff, mudiff)
        Sigma /= nu + self.n - self.dg - 1 + n_discrete_states
    
        ##  MAP-estimate of p
        p = (p + k) / (p.sum() + k * p.size) # wo. smoothing would be pML = p/n  

        ## reshape to the correct shapes
        p = p.reshape(self.sizes)
        mus = mus.reshape(self.sizes+[self.dg])
            
        return p, mus, Sigma

    def fit_variable_covariance(self, k=1, k0=1, nu=None, mu0=None, Sigma0=None):
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
        Sigma0 .. initial guess for the covariance matrices Sigma(x)
        
        returns MAP-estimate (p(x)_x, mu(x)_x, Sigma(x)_x) where x are the discrete outcomes
        
        A (computational) warning: This method of model estimation uses
        sums over the whole discrete state space.
        """
        assert self.n > 0, 'No data loaded.. use method dropdata'

        ## defaults for smoothing parameters
        if mu0 is None:
            mu0 = np.zeros(self.dg) # reasonable when using standardized data Y
        assert mu0.shape == (self.dg, )
        if nu is None:
            nu = self.dg + 1 # yields Sigma(x)=Sigma_0 if x not observed
        assert nu >= self.dg+1, 'degrees of freedom nu >= dg+1 required to achieve non-degenerate prior and deal with unobserved discrete outcomes' 
        if Sigma0 is None:
            Sigma0 = np.eye(self.dg)
        assert Sigma0.shape == (self.dg, self.dg)
        # choose V = 1/nu * Sigma0 as parameter for the Wishart prior
        # then prior mean of W(Lambda(x)|V, nu) is nu*V= Sigma0
        Vinv = nu * np.linalg.inv(Sigma0) # formerly used self.dg instead of nu here

        ## MAP-estimate Gaussians only (with unknown mean and covariance)
        if self.dc == 0: 
            mu, Sigma = self._fit_Gaussian(Vinv, nu, mu0, k0)
            return np.array([]), mu, Sigma
        
        ## initialization
        n_discrete_states = int(np.prod(self.sizes)) 
        p = np.zeros(n_discrete_states) 
        mus = np.zeros((n_discrete_states, self.dg))
        Sigmas = np.zeros((n_discrete_states, self.dg, self.dg))
    
        ## mu and p
        for i, x in enumerate(self.D):
            p[x] += 1
            mus[x, :] += self.Y[i, :]
    
        ## MAP-estimates of mu(x)
        for x in range(n_discrete_states):
            mus[x, :] = (k0 * mu0 + mus[x, :]) / (k0 + p[x]) # MAP estimator for mu(x)

        ## MAP-estimate of Sigma(x)
        for i, x in enumerate(self.D):
            diffyi_muMap = self.Y[i, :] - mus[x,:]
            Sigmas[x, :, :] += np.outer(diffyi_muMap, diffyi_muMap) # scatter matrix of the evidence
    
        for x in range(n_discrete_states): 
            mudiff = mus[x, :] - mu0
            Sigmas[x, :, :] += Vinv + k0 * np.outer(mudiff, mudiff)
            Sigmas[x, :, :] /= p[x] - self.dg + nu # > 0 since nu > self.dg
    
        ## MAP-estimate of p
        p = (p + k) / (p.sum() + k * p.size)
        
        ## reshape to the correct shapes
        p = p.reshape(self.sizes)
        mus = mus.reshape(self.sizes+[self.dg])
        Sigmas = Sigmas.reshape(self.sizes + [self.dg, self.dg])
        
        return p, mus, Sigmas

    def get_PLHvalue(self, meanparams):
        """ returns pseudo-likelihood value of current data set """
        e1, e2, lval = self.crossvalidate(meanparams) # few redundant computations
        return lval

    def crossvalidate(self, meanparams):
        """
        perform crossvalidation
        meanparams ... tuple (p, mus, Sigmas)
        p:      has shape sizes
        mus:    has shape sizes +(dg)
        Sigmas: has shape (dg,dg) if independent of discrete variables x
                else has shape sizes + (dg, dg) if dependent on x

        """
        p, mus, Sigmas = meanparams
        if len(Sigmas.shape)==2:
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
        shapes = (p.shape, mus.shape, Sigmas.shape) # store shapes
        p= p.reshape(-1)
        mus = mus.reshape((n_discrete_states, self.dg))
        Sigmas = Sigmas.reshape((n_covmats, self.dg, self.dg))

        ## discrete only models
        if self.dg == 0:
            assert self.dc > 0
            for i in range(self.n):
                x = self.D[i] # flat index, or empty list if dc==0
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
        Sigmas_inv = np.empty((n_covmats, self.dg, self.dg))
        Sigmas_red_inv = np.empty((n_covmats, self.dg, self.dg - 1, self.dg - 1)) 
        
        for x in range(n_covmats):
            dets[x] = np.linalg.det(Sigmas[x, :, :]) ** (-0.5)
            Sigmas_inv[x, :, :] = np.linalg.inv(Sigmas[x, :, :])

            # for each x, s: store Sigma[x]_{-s, -s}^{-1}
            cond_inds = list(range(1,self.dg)) # indices to keep
            for s in range(self.dg): # reduced det of cov with rows and col s deleted
                S = Sigmas[x, :, :][ix_(cond_inds, cond_inds)] # this is not a view
                Sigmas_red_inv[x, s, :, :] = np.linalg.inv(S)
                if s <self.dg - 1:
                    cond_inds[s] -= 1 # include index s and remove index s+1

        ## cross validation
        for i in range(self.n):
            yi = self.Y[i, :]
            
            ## discrete ##
            if self.dc > 0:
                x = self.D[i] # flat index, or empty list if dc==0
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
                        exps[k] = -0.5 * np.dot(np.dot(y_mu.T, Sigmas_inv[ind*cov_variable, :, :]), y_mu)
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
                Sigma_x_OH = Sigmas[covindex, :, :][ix_([s], cond_inds)] # O = s, H = -s, 1 by dg-1
                tmp_expr = np.dot(Sigma_x_OH, Sigmas_red_inv[covindex, s, :, :])
                mu_hat = mus[x][s] + np.dot(tmp_expr, y_s_mu_s)
                if s < self.dg - 1:
                    cond_inds[s] -= 1 # include index s and remove index s+1
                
                residual = yi[s] - mu_hat
    
                cts_errors[s] += (residual) ** 2 # squared residual
                
                var_s = Sigmas[covindex, :, :][s, s] - np.dot(tmp_expr, Sigma_x_OH.T) # precalculate schur complements?
                lval_testdata +=  0.5 * residual ** 2 / var_s + 0.5 * np.log(var_s) 
                # TODO: what about the constant pi part? -- ok iff left out everywhere
            
        ## reshape parameters to original form
        p.reshape(shapes[0])
        mus.reshape(shapes[1])
        Sigmas.reshape(shapes[2])
                
        dis_errors /= self.n
        cts_errors /= self.n
        lval_testdata /= self.n
    
        return dis_errors, cts_errors, lval_testdata
    