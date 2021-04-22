# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019-2021

"""
import pickle
import numpy as np

from cgmodsel.models.base import _invert_indices, unpad
from cgmodsel.models.base import BaseModelPW

# pylint: disable=R0914 # too many local variable
# pylint: disable=R0913 # too many arguments

##################################
class ModelPW(BaseModelPW):
    """
    A class for pairwise CG models.
    
    Note:
        Model to store parameters of pairwise CG model with density
        
        p(x,y) ~ exp(1/2 (C_x, y)^T Theta (C_x y) + u^T C_x + alpha^T y).

        Here, C_x is a dummy representation of x,
        u are univariate discrete parameters,
        alpha are univariate cont. parameters,
        Theta is the pairwise interaction parameter matrix given by
        Theta = (Q & R^T \\ R & -Lambda)
        with
        discrete-discrete interactions Q,
        discrete-cont. interactions R,
        and cont.-cont. interactions Lambda.

        This class uses padded parameters (that is,
        parameters for 0-th levels are included, however,
        one might want them to be constrained to 0 for identifiability reasons)

    Warning:
        Attention to the special case of only Gaussians observed variables,
        uses as above
        
        p(y) ~ exp(1/2  y^T Theta y + alpha^T y),
        
        i.e. pairwise parameters (Theta) have inverted sign
        compared to the regular precision matrix.
        The density parametrized by the precision matrix would read as
        p(y) ~ exp(-1/2 y^T precisionmatrix y + alpha^T y).
    """

    name = 'PW'

    def __init__(self,
                 pw_params: tuple = None,
                 meta: dict = None,
                 infile: str = None,
                 annotations: dict = {},
                 **kwargs):
        """
        Args:
            pw_params: parameters, either as dictionary, or tuple/list.
                As tuple/list use form (u, Q, R, alpha, Lambda),
                as dictionary provide keys pw_mat (alpha, u).
            meta (dict): dictionary of meta info (similar to meta dictionary
                 from loading data).
            infile (str): filename of a PW model file. Loading model from file
                 has the highest priority.
            annotations (dict, optional): pass annotations to the model.
            in_padded (bool): whether parameters are padded
        """

        if not infile is None:
            # load model from file
            params = pickle.load(open(infile, "rb"))
            assert params['type'] == self.name, \
            "Wrong model type (is %s, should be %s)"%(params['type'], self.name)
            pw_params = params['params']
            meta = params['meta']
            annotations.update(params['annotations'])
        
        assert (not pw_params is None) and (not meta is None), "Incomplete data"
        
        BaseModelPW.__init__(self, pw_params, meta,
                             annotations=annotations,
                             **kwargs)

    def __str__(self):
        """a string representation of the model"""
        string = ''
        if self.meta['n_cat'] > 0:
            string += '\nQ:\n' + str(self.mat_q)
            string += '\nu:' + str(self.vec_u.T)
            if self.meta['n_cg'] > 0:
                string += '\nR:\n' + str(self.mat_r)
        if self.meta['n_cg'] > 0:
            string += '\nalpha:' + str(self.alpha)
            string += '\nLambda:\n' + str(self.mat_lbda)
        return string[1:]

    def add_latent_cg(self,
                      n_latent: int = 1,
                      seed: int = -1,
                      connectionprob=.95,
                      strength=.5,
                      disscale=.3,
                      ctsscale=.2,
                      marginalize=True):
        """Augment model with latent variables.
        
        Args:
            n_latent (int): number of latent variables to add.
            connectionprob (float): probability of interaction
                between latent variables.
            marginalize (bool): if True, return S+L model (class Model_PWSL)
                        with the latent variables marginalized out.

        Warning:
            Experimental code with various other parameters
            to determine interactions strengths etc.
            Ensures that the model remains normalizable.
        
        Returns:
            Model_PWSL: marginalized model (only if marginalize=True,
            return None else)
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
        if self.meta[
                'n_cg'] > 0:  # ensure positive definiteness of precision matrix
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
            print("""Warning(model_pw.add_latent_gaussians):
                    made precision matrix of full model PD, c=%f""" % (scale))

        fac = np.sqrt(scale)
        # construct new precision matrix of full model
        mat_b = np.empty(
            (self.meta['n_cg'] + n_latent, self.meta['n_cg'] + n_latent))
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
        
        Returns:
            tuple: mean parameters (p, mus, Sigmas)
        """
        pwparams = self.vec_u, self.mat_q, self.mat_r, self.alpha, self.mat_lbda

        return self._get_meanparams(pwparams)

    def get_no_edges(self, threshold=1e-2, **kwargs):
        """Calculate number of edges.
        
        Note:
            kwargs are forwarded to get_graph method from base class.
        
        Args:
            threshold (float): cutoff for edges.
            
        Returns:
            int: number of edges.
        """

        graph = self.get_graph(threshold=threshold, **kwargs)
        graph = graph.astype(int)
        no_edges = int(np.sum(graph) / 2)

        return no_edges
    
    def get_params(self):
        """Get model parameters.
        
        Returns:
            tuple: (u, Q, R, alpha, Lambda)
        """
        return self.vec_u, self.mat_q, self.mat_r, self.alpha, self.mat_lbda
    
    def get_pairwiseparams(self, padded=True):
        """Get pairwise parameter matrix Theta, u, alpha
        
        Args:
            padded (bool): whether return padded version of parameters
            
        Returns:
            tuple: Theta, u, alpha.
        """
        theta = self.get_group_mat(aggr=False)
        if not padded:
            sizes = self.meta['sizes']
            theta = unpad(theta, sizes)
            return theta, unpad(self.vec_u, sizes), self.alpha
        return theta, self.vec_u, self.alpha

    def marginalize(self, drop_idx: (tuple, list), verb: bool = False):
        """Marginalize out CG variables.
        
        Args:
            drop_idx: indices of CG to be marginalized out.
            verb (bool): controls verbose mode.

        Returns:
            Model_PWSL: sparse + low-rank instance

        Note:
            marginalization follows a Schur-complement formula, see
            the dissertation - the decomposition is of the form S + L.
        """
        # import here to avoid files for both classes importing each other
        from cgmodsel.models.model_pwsl import ModelPWSL

        n_latent = len(drop_idx)
        assert n_latent > 0, "Supply with indices for marginalization"
        #print(self.meta['n_cg'], n_latent)
        n_do = self.meta['n_cg'] - n_latent  # number of remaining CG variables

        keep_idx = _invert_indices(
            drop_idx,
            self.meta['n_cg'])  # indices of Gaussians to keep (visible)

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
        mat_l_lbda = np.dot(tmp_expr1,
                            lbda_ho)  # see Diss(3.9), Schur complement

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

        meta = {
            'n_cat': self.meta['n_cat'],
            'n_cg': n_do,
            'sizes': self.meta['sizes']
        }

        sl_params_class = ModelPWSL(sl_params, meta)

        if verb:
            eigvals = np.linalg.eigvals(mat_l)
            print('eigvals L:', eigvals)

        return sl_params_class


#    def normalize(self):
#        """ scale all pairwise parameters"""
#        raise NotImplementedError

    def sample(self, n: int, gibbs_iter: int = 10):
        """Sample from the model.
        
        Note:
            This implements a naive Gibbs sampler for pairwise models.
            Each sample is sampled after reinitialization,
            followed by gibbs_iter steps of the Markov chain.
            Code is not optimized.
        
        Args:
            n (int): number of samples to be generated. 
            gibbs_iter (int): steps of the Markov Chain.
        
        Returns:
            tuple: cat_data (np.array), cont_data (np.array).
        """
        # TODO(franknu): speed up sampling if n is large
        # by not reinitializing and recording states after, e.g., every 5 steps
        # then, occasionally reinitialize (account for numerical errors)

        n_cat = self.meta['n_cat']
        n_cg = self.meta['n_cg']
        if n_cat == 0:
            # TODO(franknu): code redundancy
            mat_sigma = np.linalg.inv(self.mat_lbda)
            mat_l = np.linalg.cholesky(mat_sigma)  # Sigma = LL^T with lower tri L

            cont_data = np.random.standard_normal((n, n_cg))  # zero mean
            cont_data = np.dot(cont_data, mat_l.T)  # adjust cov matrix to Sigma

            mu_offset = np.squeeze(np.dot(mat_sigma, self.alpha))
            cont_data += np.outer(np.ones(n), mu_offset)

        glims = self.meta['cat_glims']

        ## get discrete marginal model
        if n_cg > 0:  # get marginal discrete model
            pwslmodel = self.marginalize(drop_idx=list(range(n_cg)))
            pwmodel = pwslmodel.to_pwmodel()
            vec_u, mat_q, _, _, _ = pwmodel.get_params()
        else:
            vec_u = self.vec_u
            mat_q = self.mat_q
        # now, vec_u are univariate params of the (marginal) discrete model,
        # and mat_q are pairwise params of the (marginal) discrete model

        ## Gibbs sampling for the (marginal) discrete model
        cat_data = np.empty((n, n_cat), dtype=np.int)
        for i in range(n):
            ## initialization of discrete states x
            x = np.zeros(n_cat, dtype=np.int)  # simple initialization at zero
            # Markov chain converges with any initialization
            under_exp = vec_u.copy()
            # under_exp is an array of length self.ltot
            # it stores the terms 'under the exp' from the enumerator
            # of the node conditionals given the current state x in the sense
            # that under_exp[r:k] = p(x_r=k|x_{-r}), here r:k=glims[r]+k.
            # We update under_exp whenever x changes.

            for _ in range(gibbs_iter):

                for r in range(n_cat):
                    # for each discrete variable:
                    # sample x[r] from node coditional distribution
                    # given current other values x[j] for j\neq r
                    xr_old = x[r]
                    under_exp_r = under_exp[glims[r]:glims[r + 1]]  # a view
                    # print('x[at]r', r, x)

                    # TODO(franknu): reuse stable exp
                    conditionalprobs = np.exp(under_exp_r -
                                              np.amax(under_exp_r))
                    conditionalprobs /= np.sum(conditionalprobs)
                    cumulative_probs = np.cumsum(conditionalprobs)

                    rand = np.random.rand()
                    ind = 0
                    while cumulative_probs[ind] < rand:
                        ind += 1
                    x[r] = ind  # set new value

                    if xr_old != x[r]:
                        # update values of under_exp
                        # need only diff for 'newly activated'
                        # pairwise interaction parameters
                        under_exp -= mat_q[:, glims[r] + xr_old]
                        under_exp += mat_q[:, glims[r] + x[r]]

    #        print('x%d='%(i), x)
            cat_data[i, :] = x

        ## sample conditional Gaussians Y
        mat_sigma = np.linalg.inv(self.mat_lbda)
        mat_l = np.linalg.cholesky(mat_sigma)  # Sigma = LL^T with lower tri L

        cont_data = np.random.standard_normal((n, n_cg))  # zero mean
        cont_data = np.dot(cont_data, mat_l.T)  # adjust cov matrix to Sigma

        mu_offset = np.squeeze(np.dot(mat_sigma, self.alpha))
        cont_data += np.outer(np.ones(n), mu_offset)

        for i in range(n):
            # add means, note that conditional Gaussian is given by
            # p(y|x) ~ exp((alpha + R dummy(x))^T y -.5 y^T Lambda y)
            # from the conversion formulas: mu(x) = Sigma(alpha + R dummy(x))
            x = cat_data[i, :]
            for r in range(n_cat):
                cont_data[i, :] += np.dot(mat_sigma,
                                          self.mat_r[:, glims[r] + x[r]])
            # TODO(franknu): perhaps group calculations by discrete states
            # (check bottlenecks before)

        return cat_data, cont_data
