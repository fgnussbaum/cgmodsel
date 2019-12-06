# Copyright (c) 2017-2019 Frank Nussbaum (frank.nussbaum@uni-jena.de)
"""
@author: Frank Nussbaum

CG model selection via MAP-estimation using mean parameters.
"""

import numpy as np
from numpy import ix_

from cgmodsel.base_solver import BaseCGSolver

# pylint: disable=W0511 # todos
# pylint: disable=R0914 # too many locals


class MAP(BaseCGSolver):
    """
    this class can be used to estimate maximum a-posteriori models
    for CG distributions that are parametrized using mean parameters
    """

    def __init__(self):
        """pass a dictionary that provides with keys dg, dc, and L"""
        super().__init__()

        self.name = 'MAP'
        self.cat_format_required = 'flat'

    def _fit_gaussian(self, mat_v_inv, deg_freedom: int, vec_mu0,
                      n_art_cg: int):
        """ fit Gaussian MAP-estimate

        Wishart prior for precision matrix
        deg_freedom    ... degrees of freedom (=nu)
        mat_v_inv  ... inverse of V for the prior W(Lambda| V, nu)

        Gaussian prior for mean
        n_art_cg    ... number of artificial observations
        vec_mu0   ... mean of prior N(mu | mu0, (n_art_cg * Lambda)^{-1})

        Note: setting n_art_cg=0 (and nu = #Gaussians) produces ML estimate

        returns MAP estimates (vec_mu, mat_sigma)
        """
        assert self.meta['n_cat'] == 0, 'use for continuous variables only'
        assert self.meta['n_cg'] > 0

        vec_mu = np.sum(self.cont_data, axis=0)  # sum over rows
        vec_mu = n_art_cg * vec_mu0 + vec_mu
        vec_mu /= n_art_cg + self.meta['n_data']

        mat_sigma = mat_v_inv  # this is V^{-1} from the doc
        for i in range(self.meta['n_data']):
            # add 'scatter matrix' of the evidence
            diff_yi_mumap = self.cont_data[i, :] - vec_mu
            mat_sigma += np.outer(diff_yi_mumap, diff_yi_mumap)

        mudiff = vec_mu - vec_mu0
        mat_sigma += n_art_cg * np.outer(mudiff, mudiff)
        mat_sigma /= self.meta['n_data'] + deg_freedom - self.meta['n_cg']

        return vec_mu, mat_sigma

    def fit_fixed_covariance(self,
                             n_art_cat: int = 1,
                             n_art_cg: int = 1,
                             deg_freedom: int = None,
                             vec_mu0=None,
                             mat_sigma0=None):
        """fit MAP-CG model with a unique single covariance matrix and
        individual means for all conditional gaussians.

        ** components of the Dirichlet-Normal-Wishart prior **
        Dirichlet prior parameters (prior for discrete distribution)
        n_art_cat     ... Laplacian smoothing parameter
                = artificial observations per discrete variable
                (this is alpha in the doc)

        Gaussian prior parameters (prior for means of conditional Gaussians)
        n_art_cg    ... number of 'artificial' data points per CG
                  (this is kpa in the doc)
        mu0   ... value of artificial data points

        params Wishart prior (for shared precision matrix of CG distributions)
        deg_freedom     .. degrees of freedom (=nu)
        mat_sigma0 .. initial guess for the covariance matrix

        returns MAP-estimate (p(x)_x, mu(x)_x, mat_sigma),
                where x are the discrete outcomes

        A (computational) warning: This method of model estimation uses
        sums over the whole discrete state space.
        """
        # TODO(franknu): publish doc
        # future ideas(franknu):
        # (1) iter only over observed discrete examples,
        #     all other outcomes default to 'prior'
        #     (often times much less than the whole discrete state space)
        #     use dictionary + counts?
        # (2) use flag to indicate if cov is fixed or variable
        #     (avoid code redundancy)
        n_cg = self.meta['n_cg']
        assert self.meta['n_data'] > 0, 'No data loaded.. use method dropdata'

        ## defaults for smoothing parameters
        if vec_mu0 is None:
            # reasonable choice when using standardized data Y
            vec_mu0 = np.zeros(n_cg)
        assert vec_mu0.shape == (n_cg,)
        if deg_freedom is None:
            deg_freedom = n_cg  # least informative non-degenerate prior
        assert deg_freedom >= n_cg, 'need deg >= n_cg for non-degenerate prior'
        if mat_sigma0 is None:
            mat_sigma0 = np.eye(n_cg)  # standardized data --> variances are 1
        assert mat_sigma0.shape == (n_cg, n_cg)

        # choose V = 1/deg_freedom * mat_sigma0 for the Wishart prior
        mat_v_inv = deg_freedom * np.linalg.inv(mat_sigma0)
        # note: formerly used n_cg instead of deg_freedom here

        ## MAP-estimate Gaussians only (with unknown mean and covariance)
        if self.meta['n_cat'] == 0:
            vec_mu, mat_sigma = self._fit_gaussian(mat_v_inv, deg_freedom,
                                                   vec_mu0, n_art_cg)
            return np.array([]), vec_mu, mat_sigma

        ## MAP-estimation in the presence of discrete variables
        n_discrete_states = int(np.prod(self.meta['sizes']))

        probs_cat = np.zeros(n_discrete_states)
        mus = np.zeros((n_discrete_states, n_cg))
        mat_sigma = np.zeros((n_cg, n_cg))

        ## mu and p
        for i, state in enumerate(self.cat_data):
            probs_cat[state] += 1
            mus[state, :] += self.cont_data[i, :]

        ## MAP-estimates of mu(# MAP estimator for mu(x))
        for state in range(n_discrete_states):
            # MAP estimator for mu(# MAP estimator for mu(x))
            mus[state, :] = (n_art_cg * vec_mu0 + mus[state, :])
            mus[state, :] /= n_art_cg + probs_cat[state]

        ## MAP-estimate of mat_sigma
        mat_sigma = mat_v_inv  # this is V^{-1} from the doc
        for i, state in enumerate(self.cat_data):
            # add 'scatter matrix' of the evidence
            diff_yi_mumap = self.cont_data[i, :] - mus[state, :]
            mat_sigma += np.outer(diff_yi_mumap, diff_yi_mumap)

        for state in range(n_discrete_states):
            # add scatter part of artificial observations mu0
            mudiff = mus[state, :] - vec_mu0
            mat_sigma += n_art_cg * np.outer(mudiff, mudiff)
        mat_sigma /= deg_freedom + self.meta[
            'n_data'] - n_cg - 1 + n_discrete_states

        ##  MAP-estimate of p
        probs_cat = probs_cat + n_art_cat
        probs_cat /= probs_cat.sum() + n_art_cat * probs_cat.size
        # note: without smoothing would yield p = p/n

        ## reshape to the correct shapes
        probs_cat = probs_cat.reshape(self.meta['sizes'])
        mus = mus.reshape(self.meta['sizes'] + [n_cg])

        return probs_cat, mus, mat_sigma

    def fit_variable_covariance(self,
                                n_art_cat: int = 1,
                                n_art_cg: int = 1,
                                deg_freedom: int = None,
                                vec_mu0=None,
                                mat_sigma0=None):
        """fit MAP-CG model with  individual covariance matrices
        and means for all conditional gaussians.

        ** Components of the Dirichlet-Normal-Wishart Prior **
        Dirichlet prior parameters (prior for discrete distribution)
        k     ... Laplacian smoothing parameter
                  = 'artificial' observations per discrete variable
                  (this is alpha in the doc)

        Gaussian prior parameters (prior for means of conditional Gaussians)
        n_art_cg    ... # 'artificial' data points per conditional Gaussian
                  (this is kpa in the doc)
        mu0   ... value of artificial data points

        params Wishart prior (for shared precision matrix of CG distributions)
        deg_freedom     .. degrees of freedom (=nu)
        mat_sigma0 .. initial guess for the covariance matrices mat_sigma(x)

        returns MAP-estimate (p(x)_x, mu(x)_x, mat_sigma(x)_x),
                where x are the discrete outcomes

        A (computational) warning: This method of model estimation uses
        sums over the whole discrete state space.
        """
        # TODO(franknu): publish doc
        assert self.meta['n_data'] > 0, 'No data loaded.. use method dropdata'
        n_cg = self.meta['n_cg']

        ## defaults for smoothing parameters
        if vec_mu0 is None:
            vec_mu0 = np.zeros(n_cg)  # reasonable when using standardized data
        assert vec_mu0.shape == (n_cg,)
        if deg_freedom is None:
            deg_freedom = n_cg + 1
            # yields mat_sigma(x)=mat_sigma_0 if x not observed
        string = 'need deg >= dg+1 for non-degenerate'
        string += 'prior and deal with unobserved discrete outcomes'
        assert deg_freedom >= n_cg + 1, string

        if mat_sigma0 is None:
            mat_sigma0 = np.eye(n_cg)
        assert mat_sigma0.shape == (n_cg, n_cg)
        # choose V = 1/deg_freedom * mat_sigma0 for the Wishart prior
        # -> prior mean of W(Lambda(x)|V, deg_freedom) is
        #   deg_freedom*V= mat_sigma0
        mat_v_inv = deg_freedom * np.linalg.inv(mat_sigma0)
        # note: formerly used n_cg instead of nu here

        ## MAP-estimate Gaussians only (with unknown mean and covariance)
        if self.meta['n_cat'] == 0:
            vec_mu, mat_sigma = self._fit_gaussian(mat_v_inv, deg_freedom,
                                                   vec_mu0, n_art_cg)
            return np.array([]), vec_mu, mat_sigma

        ## initialization
        n_discrete_states = int(np.prod(self.meta['sizes']))
        probs_cat = np.zeros(n_discrete_states)
        mus = np.zeros((n_discrete_states, n_cg))
        sigmas = np.zeros((n_discrete_states, n_cg, n_cg))

        ## mu and p
        for i, state in enumerate(self.cat_data):
            probs_cat[state] += 1
            mus[state, :] += self.cont_data[i, :]

        ## MAP-estimates of mu(state)
        for state in range(n_discrete_states):
            mus[state, :] = (n_art_cg * vec_mu0 + mus[state, :]) / \
                (n_art_cg + probs_cat[state]) # MAP estimator for mu(state)

        ## MAP-estimate of mat_sigma(state)
        for i, state in enumerate(self.cat_data):
            # scatter matrix of the evidence
            diff_yi_mumap = self.cont_data[i, :] - mus[state, :]
            sigmas[state, :, :] += np.outer(diff_yi_mumap, diff_yi_mumap)

        for state in range(n_discrete_states):
            mudiff = mus[state, :] - vec_mu0
            sigmas[state, :, :] += mat_v_inv + \
                n_art_cg * np.outer(mudiff, mudiff)
            sigmas[state, :, :] /= probs_cat[state] - n_cg + deg_freedom
            # note: divisor is > 0 since deg_freedom > n_cg

        ## MAP-estimate of p
        probs_cat = probs_cat + n_art_cat
        probs_cat /= probs_cat.sum() + n_art_cat * probs_cat.size

        ## reshape to the correct shapes
        probs_cat = probs_cat.reshape(self.meta['sizes'])
        mus = mus.reshape(self.meta['sizes'] + [n_cg])
        sigmas = sigmas.reshape(self.meta['sizes'] + [n_cg, n_cg])

        return probs_cat, mus, sigmas

    def get_plhvalue(self, mean_params):
        """ returns pseudo-likelihood value of current data set """
        _, _, lval = self.crossvalidate(mean_params)
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
        n_cat = self.meta['n_cat']
        n_cg = self.meta['n_cg']
        sizes = self.meta['sizes']
        probs_cat, mus, sigmas = mean_params
        if len(sigmas.shape) == 2:
            cov_variable = 0
            # use zero to scale indices and toggle between modes of covariance
        else:
            cov_variable = 1

        ## initialization
        n_discrete_states = int(np.prod(self.meta['sizes']))
        # n_discrete_states is 1 if dc==0 since np.prod([])=1
        n_covmats = (n_discrete_states - 1) * cov_variable + 1
        # number of different covariance matrices

        dis_errors = np.zeros(n_cat)
        cts_errors = np.zeros(n_cg)

        lval_testdata = 0  # likelihood value

        ## reshape by collapsing discrete dimensions
        shapes = (probs_cat.shape, mus.shape, sigmas.shape)  # store shapes
        probs_cat = probs_cat.reshape(-1)
        mus = mus.reshape((n_discrete_states, n_cg))
        sigmas = sigmas.reshape((n_covmats, n_cg, n_cg))

        ## discrete only models
        if n_cg == 0:
            assert n_cat > 0
            for i in range(self.meta['n_data']):
                x = self.cat_data[i]  # flat index, or empty list if dc==0
                cat = list(np.unravel_index(x, sizes))
                # cat is multiindex of discrete outcome

                for r in range(n_cat):
                    probs = np.empty(sizes[r])
                    tmp = cat[r]
                    for k in range(sizes[r]):
                        # calculate conditional probs
                        cat[r] = k
                        ind = np.ravel_multi_index(tuple(cat), sizes)
                        probs[k] = probs_cat[ind]  # not yet unnormalized
                    cat[r] = tmp
                    prob_xr = probs[tmp] / np.sum(probs)  # normalize
                    dis_errors[r] += 1 - prob_xr
                    lval_testdata -= np.log(prob_xr)
                    # note: log(x) ~ x-1 around 1

            return dis_errors, cts_errors, lval_testdata

        ## setting with Gaussian variables
        ## precalculate determinants, inverse matrices
        # (full with dim recuced by 1)
        dets = np.empty(n_covmats)
        sigmas_inv = np.empty((n_covmats, n_cg, n_cg))
        sigmas_red_inv = np.empty((n_covmats, n_cg, n_cg - 1, n_cg - 1))

        for x in range(n_covmats):
            dets[x] = np.linalg.det(sigmas[x, :, :])**(-0.5)
            sigmas_inv[x, :, :] = np.linalg.inv(sigmas[x, :, :])

            # for each x, s: store mat_sigma[x]_{-s, -s}^{-1}
            cond_inds = list(range(1, n_cg))  # indices to keep
            for s in range(n_cg):
                # reduced det of cov with rows and col s deleted
                mat_s = sigmas[x, :, :][ix_(cond_inds, cond_inds)]  # not a view
                sigmas_red_inv[x, s, :, :] = np.linalg.inv(mat_s)
                if s < n_cg - 1:
                    cond_inds[s] -= 1  # include index s and remove index s+1

        ## cross validation
        for i in range(self.meta['n_data']):
            y_i = self.cont_data[i, :]

            ## discrete ##
            if n_cat > 0:
                x = self.cat_data[i]  # flat index, or empty list if dc==0
                covindex = x * cov_variable
                cat = list(np.unravel_index(x, sizes))
                # cat is multiindex of discrete outcome

                for r in range(n_cat):
                    probs = np.empty(sizes[r])
                    exps = np.empty(sizes[r])
                    tmp = cat[r]
                    for k in range(sizes[r]):  # calculate all conditional probs
                        cat[r] = k
                        ind = np.ravel_multi_index(tuple(cat), sizes)

                        y_mu = y_i - mus[ind]
                        exps[k] = -0.5 * np.dot(
                            np.dot(y_mu.T,
                                   sigmas_inv[ind * cov_variable, :, :]), y_mu)
                        probs[k] = probs_cat[ind] * dets[ind * cov_variable]
                    cat[r] = tmp
                    exps = np.exp(exps - np.max(exps))  # stable exp (!)
                    probs = np.multiply(probs, exps)  # unnormalized (!)

                    prob_xr = probs[tmp] / np.sum(probs)
                    dis_errors[r] += 1 - prob_xr
                    lval_testdata -= np.log(
                        prob_xr)  # note that log(x) ~ x-1 around 1
            else:  # no discrete variables: set dimension to zeros
                x = 0
                covindex = 0

            ## continuous ##
            cond_inds = list(range(1, n_cg))
            # cond_inds is list of indices to keep for Schur complement
            for s in range(n_cg):
                # p(y_s|..) = sqrt(1/(2pi))*|Schur|^{-1/2}*
                # ... exp(-.5*(y_s-mu)var_s^{-1}(y_s-mu))
                # since by fixing x this is a node conditional
                # of a purely Gaussian model
                y_s_mu_s = y_i[ix_(cond_inds)] - mus[x][ix_(cond_inds)]
                mat_sigma_x_oh = sigmas[covindex, :, :][ix_([s], cond_inds)]
                # O = s, H = -s, 1 by dg-1
                tmp_expr = np.dot(mat_sigma_x_oh, sigmas_red_inv[covindex,
                                                                 s, :, :])
                mu_hat = mus[x][s] + np.dot(tmp_expr, y_s_mu_s)
                if s < n_cg - 1:
                    cond_inds[s] -= 1  # include index s and remove index s+1

                residual = y_i[s] - mu_hat

                cts_errors[s] += (residual)**2  # squared residual

                var_s = sigmas[covindex, :, :][s, s]
                var_s -= np.dot(tmp_expr, mat_sigma_x_oh.T)
                # TODO(franknu): precalculate schur complements?
                lval_testdata += 0.5 * residual**2 / var_s
                lval_testdata += 0.5 * np.log(var_s)
                # TODO(franknu): what about the constant pi part?
                # -- ok iff left out everywhere

        ## reshape parameters to original form
        probs_cat.reshape(shapes[0])
        mus.reshape(shapes[1])
        sigmas.reshape(shapes[2])

        dis_errors /= self.meta['n_data']
        cts_errors /= self.meta['n_data']
        lval_testdata /= self.meta['n_data']

        return dis_errors, cts_errors, lval_testdata
