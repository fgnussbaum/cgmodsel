# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
import numpy as np

from cgmodsel.models.base import _invert_indices
from cgmodsel.models.base import BaseModelPW

# pylint: disable=R0914 # too many local variable
# pylint: disable=R0913 # too many arguments

##################################
class ModelPW(BaseModelPW):
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

    def __init__(self, pw_params, meta, **kwargs):
        BaseModelPW.__init__(self, pw_params, meta, **kwargs)
        # possible extension: pw params in dict, allow passing of pwmat

        self.name = 'PW'

    def __str__(self):
        """a string representation of the model"""
        string = ''
        if np.prod(self.mat_q.shape) > 0:
            string += '\nQ:\n' + str(self.mat_q)
        if np.prod(self.vec_u.shape) > 0:
            string += '\nu:' + str(self.vec_u.T)
        if np.prod(self.mat_r.shape) > 0:
            string += '\nR:\n' + str(self.mat_r)
        if np.prod(self.alpha.shape) > 0:
            string += '\nalpha:' + str(self.alpha)
        if np.prod(self.mat_lbda.shape) > 0:
            string += '\nLambda:\n' + str(self.mat_lbda)

        return string[1:]

    def add_latent_cg(self,
                      n_latent=1,
                      seed=-1,
                      connectionprob=.95,
                      strength=.5,
                      disscale=.3,
                      ctsscale=.2,
                      marginalize=True):
        """
        augment n_latent latent variables to given pairwise model pwmodel

        connectionprob ... probability of interaction between latent var

        marginalize ... if True, return S+L model (class Model_PWSL)
                        with the latent variables marginalized out

        experimental code,
        various other parameters to determine interactions strengths etc.
        """
        if seed != -1:
            print('Model_PW.add_latent_CG: Set seed to %d' % (seed))
            np.random.seed(seed)

        lbda_ho = np.zeros((n_latent, self.meta['n_cg']))
        alpha_h = np.zeros(n_latent)
        mat_r_h = np.zeros((n_latent, self.meta['ltot']))

        def get_entry(dim, prob=.8, offset=.7, scale=.3, samesign=True):
            """sample one interaction parameter"""
            if np.random.random() > prob:
                return np.zeros(dim)
            if samesign:
                sign = -1
                if np.random.random() > .5:
                    sign = 1
            group = np.empty(dim)
            for i in range(dim):
                if not samesign:
                    sign = -1
                    if np.random.random() > 0.5:
                        sign = 1
                group[i] = sign * (offset + scale * np.random.random())

            return group

        ## add edges with specified probability
        glims = self.meta['cat_glims']
        for i in range(n_latent):
            for r in range(self.meta['n_cat']):
                mat_r_h[i, glims[r]+1:glims[r+1]] = \
                   get_entry(self.meta['sizes'][r]-1, prob=connectionprob,
                             offset=strength, scale=disscale)

            for s in range(self.meta['n_cg']):  # n_cg
                lbda_ho[i, s] = get_entry(1,
                                          prob=connectionprob,
                                          offset=strength + .3,
                                          scale=ctsscale)

        ## update model parameters

        scale = 1.0
        if self.meta['n_cg'] > 0:  # ensure positive definiteness of precision matrix
            tmp = np.dot(lbda_ho.T, lbda_ho)  # n_cg by n_cg
            # calculate smallest eigenvalue of A=Lambda-c*tmp
            # choose c s.t. smallest eigenvalue is positive for a valid distribution
            # could use power method on A^(-1)
            while True:
                lamin = np.min(np.linalg.eigvals(self.mat_lbda - scale * tmp))
                if lamin < .2:
                    scale *= 2 / 3
                else:
                    break

        if scale != 1.0:
            print(
                """Warning(model_pw.add_latent_gaussians):
                    made precision matrix of full model PD, c=%f"""
                % (scale))

        fac = np.sqrt(scale)
        # construct new precision matrix of full model
        mat_b = np.empty((self.meta['n_cg'] + n_latent, self.meta['n_cg'] + n_latent))
        mat_b[:-n_latent, :-n_latent] = self.mat_lbda
        mat_b[:-n_latent, -n_latent:] = fac * lbda_ho.T  # n_cg by n_latent
        mat_b[-n_latent:, :-n_latent] = fac * lbda_ho
        mat_b[-n_latent:,
              -n_latent:] = np.eye(n_latent)  # orthogonal latent variables

        if self.meta['n_cg'] > 0:
            if self.meta['n_cat'] > 0:
                self.mat_r = np.concatenate((self.mat_r, mat_r_h), axis=0)
#            print(self.alpha, alpha_h)
            self.alpha = np.concatenate((self.alpha, alpha_h))
        else:
            self.alpha = alpha_h
            self.mat_r = mat_r_h

        self.meta['n_cg'] += n_latent

        self.mat_lbda = mat_b

        if marginalize:  # return marginal model
            drop_idx = [self.meta['n_cg'] - i - 1 for i in range(n_latent)]
            return self.marginalize(drop_idx=drop_idx, verb=0)

    def get_meanparams(self):
        """
        wrapper for _get_meanparams from base class Model_PW_Base
        """
        pwparams = self.vec_u, self.mat_q, self.mat_r, self.alpha, self.mat_lbda

        return self._get_meanparams(pwparams)

    def get_params(self):
        """return model parameters """
        return self.vec_u, self.mat_q, self.mat_r, self.alpha, self.mat_lbda

    def marginalize(self, drop_idx, verb=False):
        """ marginalize out CG variables

        drop_idx ... indices of (conditional) Gaussians to be marginalized out

        returns Model_PWSL instance

        marginalization follows a Schur-complement formula given in the Dissertation script
        (see section marginalization of pairwise CG distributions)
        the decomposition is of the form S + L
        """
        # import here to avoid files for both classes importing each other
        from cgmodsel.models.model_pwsl import ModelPWSL

        n_latent = len(drop_idx)
        assert n_latent > 0, "Supply with indices for marginalization"
        #print(self.meta['n_cg'], n_latent)
        n_do = self.meta['n_cg'] - n_latent  # number of remaining CG variables

        keep_idx = _invert_indices(
            drop_idx, self.meta['n_cg'])  # indices of Gaussians to keep (visible)

        ## observed direct interactions (together they form S)
        mat_s_q = self.mat_q
        mat_s_r = self.mat_r[keep_idx, :]
        mat_s_lbda = self.mat_lbda[np.ix_(keep_idx, keep_idx)]

        ## latent interactions (together they form L)
        mat_r_h = self.mat_r[drop_idx, :]  # h for hidden
        lbda_hh = self.mat_lbda[np.ix_(drop_idx, drop_idx)]
        sigma_hh = np.linalg.inv(lbda_hh)
        lbda_ho = self.mat_lbda[np.ix_(drop_idx, keep_idx)]

        ## calculate L from latent interactions
        tmp_expr1 = np.dot(lbda_ho.T, sigma_hh)

        mat_l_q = np.dot(np.dot(mat_r_h.T, sigma_hh),
                         mat_r_h)  # has univariate effects on its diagonal
        mat_l_r = -np.dot(tmp_expr1, mat_r_h)
        mat_l_lbda = np.dot(tmp_expr1, lbda_ho)  # see Diss(3.9), Schur complement

        ltot = self.meta['ltot']
        mat_l = np.empty((ltot + n_do, ltot + n_do))
        mat_l[:ltot, :ltot] = mat_l_q
        mat_l[ltot:, :ltot] = mat_l_r
        mat_l[:ltot, ltot:] = mat_l_r.T
        mat_l[ltot:, ltot:] = mat_l_lbda

        ## the observed alpha is also changed
        tialpha = self.alpha[keep_idx] - \
            np.dot(tmp_expr1, self.alpha[drop_idx]) # see (3.8)

        # note that Theta_Marg = S + L (where Lambda has neg sign in Theta_Marg)

        ## construct a PW-SL model and return it
        sl_params = self.vec_u, mat_s_q, mat_s_r, tialpha, mat_s_lbda, mat_l

        meta = {'n_cat': self.meta['n_cat'],
                'n_cg': n_do,
                'sizes': self.meta['sizes']}

        sl_params_class = ModelPWSL(sl_params, meta)

        if verb:
            eigvals = np.linalg.eigvals(mat_l)
            print('eigvals L:', eigvals)

        return sl_params_class


#    def normalize(self):
#        """ scale all pairwise parameters"""
#        raise NotImplementedError
