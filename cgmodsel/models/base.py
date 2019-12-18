# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
import abc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# pylint: disable=C0103 # snake_case
# pylint: disable=R0914 # too many local variable
# pylint: disable=R0913 # too many arguments
# pylint: disable=W0511 # todos

###################################### model types

GAUSSIAN = 'Gaussian'
BINARY = 'Binary'
DISCRETE = 'Discrete'
CG = 'CG'


def get_modeltype(meta):
    """derive modeltype from variable specifications"""
    if meta['n_cat'] == 0:
        modeltype = GAUSSIAN
    elif meta['n_cg'] == 0:
        binary = True
        for no_labels in meta['sizes']:
            if no_labels != 2:
                binary = False
        if binary:
            modeltype = BINARY
        else:
            modeltype = DISCRETE
    else:
        modeltype = CG
    return modeltype


def _invert_indices(arr, range_max):
    """return all indices from range(range_max) that are not in arr as a list"""
    inv = []
    for j in range(range_max):
        if j not in arr:
            inv.append(j)
    return inv


def _schur_complement(mat, keep_idx=None, drop_idx=None):
    """Returns S and L from upper Schur complement S-L of array_like M
    where the 'upper block' is indexed by keep_idx and the lower block by drop_idx
    (only one list should be given).
    """
    ## derive index lists
    assert keep_idx or drop_idx
    if not keep_idx is None:
        drop_idx = _invert_indices(keep_idx, mat.shape[0])
    else:
        keep_idx = _invert_indices(drop_idx, mat.shape[0])

    mat_s = mat[np.ix_(keep_idx, keep_idx)]
    mat_l = np.dot(
        mat[np.ix_(keep_idx, drop_idx)],
        np.dot(np.linalg.inv(mat[np.ix_(drop_idx, drop_idx)]),
               mat[np.ix_(drop_idx, keep_idx)]))

    return mat_s, mat_l


######################################


def _split_theta(theta, ltot):
    """split pairwise interaction parameter matrix into components"""
    mat_q = theta[:ltot, :ltot]
    mat_r = theta[ltot:, :ltot]
    mat_lbda = -theta[ltot:, ltot:]
    return mat_q, mat_r, mat_lbda


def _theta_from_components(mat_q, mat_r, mat_lbda):
    """
    stack components together into pairwise interaction matrix of CG model

    mat_q ... discrete - discrete interactions
    mat_r ... continuous - discrete interactions
    mat_lbda ... continuous - continuous interactions
    """
    ltot = mat_q.shape[0]
    dim = ltot + mat_lbda.shape[0]
    theta = np.empty((dim, dim))
    if ltot > 0:
        theta[:ltot, :ltot] = mat_q
        theta[ltot:, :ltot] = mat_r
        theta[:ltot, ltot:] = mat_r.T
    theta[ltot:, ltot:] = -mat_lbda
    return theta


def pad(arr, sizes, rowsonly=False):
    """ padding for pairwise parameter matrix or univariate parameter vector u
    (adds rows/columns of zeros for the 0-th levels,
    they correspond to interactions parameters set to 0 for identifiability
    reasons)

    sizes ... list of number of levels for each discrete variable
    rowsonly ... if True, pad only rows"""
    ltot = sum(sizes)
    n_cat = len([n_levels for n_levels in sizes if n_levels > 0])
    sizes_cum = np.cumsum([0] + sizes)

    d_unpadded = arr.shape[0]
    n_cg = d_unpadded - ltot + n_cat

    rows_discrete_indices = _invert_indices(sizes_cum[:-1], ltot)
    rows_gauss_indices = list(range(ltot, ltot + n_cg))
    row_indices = rows_discrete_indices + rows_gauss_indices

    if len(arr.shape) == 2:
        if rowsonly:
            col_indices = list(range(ltot - n_cat + n_cg))
            newarr = np.zeros((d_unpadded + n_cat, d_unpadded))
        else:
            col_indices = row_indices
            newarr = np.zeros((d_unpadded + n_cat, d_unpadded + n_cat))

        newarr[np.ix_(row_indices, col_indices)] = arr

        return newarr

    newu = np.zeros(d_unpadded + n_cat)
    newu[np.ix_(rows_discrete_indices)] = arr
    return newu


def unpad(theta, sizes):
    """unpad pairwise parameter matrix
    sizes ... list of number of levels for each discrete variable"""
    sizes_cum = np.cumsum([0] + sizes)

    if len(theta.shape) == 2:
        theta = np.delete(theta, sizes_cum[:-1], axis=1)  # delete rows
    return np.delete(theta, sizes_cum[:-1], axis=0)  # delete columns


###############################################################################
# Parameter Conversion
###############################################################################
def mean_to_canonparams(meanparams: (tuple, list)):
    """
    convert mean parameters to canonical parameters

    meanparameters is a tupe (p, mus, sigmas),
    where sigmas is the covariance for all conditional distributions p(y|x)
    for pairwise model -> 2D array, and sigmas is a 3D array if the covariances
    are different for p(y|x), in this case sigmas(x, :, :) is the covariance of
    p(y|x)
    """
    p, mus, sigmas = meanparams

    sizes = p.shape
    if len(sizes) == 1 and sizes[0] == 0:
        sizes = []
    n_cat = len(sizes)
    n_cg = sigmas.shape[-1]

    if n_cg == 0:
        q = np.log(p) - n_cg / 2 * np.log(2 * np.pi)  # pylint: disable=invalid-name
        return q.reshape(sizes), np.empty(0), np.empty(0)
    if n_cat == 0:
        lambdas = np.linalg.inv(sigmas)
        nus = np.dot(lambdas, mus)
        return np.empty(0), nus, lambdas

    if len(sigmas.shape) == 2:
        cov_variable = 0  # use zero to scale indices and
        # toggle between modes of covariance
    else:
        cov_variable = 1

    n_discrete_states = int(np.prod(sizes))  # prod([]) is 1.0
    n_covmats = (n_discrete_states - 1) * cov_variable + 1
    # n_covmats is the number of different covariance matrices

    shapes = mus.shape, sigmas.shape  # store shapes
    p = p.reshape(-1)
    mus = mus.reshape((n_discrete_states, n_cg))
    sigmas = sigmas.reshape((n_covmats, n_cg, n_cg))

    ## initialization
    q = np.empty(n_discrete_states)
    nus = np.empty(mus.shape)
    lambdas = np.empty(sigmas.shape)

    ## precompute determinants from normalization and lambdas
    dets = np.empty(n_covmats)
    for x in range(n_covmats):
        lambdas[x, :, :] = np.linalg.inv(sigmas[x, :, :])
        dets[x] = np.linalg.det(sigmas[x, :, :])

    for x in range(n_discrete_states):
        covindex = x * cov_variable
        mu_x = mus[x, :]
        q[x] = np.log(p[x]) - n_cg / 2 * np.log(2 * np.pi) \
            - 0.5 * np.log(dets[covindex]) \
            - 0.5 * np.dot(np.dot(mu_x.T, lambdas[covindex, :, :]), mu_x)

        nus[x] = np.dot(lambdas[covindex, :, :], mu_x)

    q = q.reshape(sizes)
    nus = nus.reshape(shapes[0])
    lambdas = lambdas.reshape(shapes[1])

    return q, nus, lambdas


def canon_to_meanparams(canparams: (tuple, list)):
    """
    convert canonical parameters to mean parameters

    canparams is a tupe (q, nus, lambdas),
    where lambdas is the precision for all conditional distributions p(y|x)
    for pairwise model -> 2D array,
    and lambdas is a 3D array if the precision matrices
    are different for p(y|x), in this case lambdas(x, :, :) is the precision of
    p(y|x)
    """
    q, nus, lambdas = canparams

    sizes = q.shape
    if len(sizes) == 1 and sizes[0] == 0:
        sizes = []
    n_cat = len(sizes)
    n_cg = lambdas.shape[-1]

    if n_cg == 0:
        p = np.exp(q - np.max(q))
        p /= np.sum(p)
        return p.reshape(sizes), np.empty(0), np.empty(0)
    if n_cat == 0:
        sigmas = np.linalg.inv(lambdas)
        mus = np.dot(lambdas, nus)
        return np.empty(0), nus, sigmas

    if len(lambdas.shape) == 2:
        cov_variable = 0  # use zero to scale indices and
        # toggle between modes of covariance
    else:
        cov_variable = 1

    n_discrete_states = int(np.prod(sizes))  # prod([]) is 1.0
    n_covmats = (n_discrete_states - 1) * cov_variable + 1
    # n_covmats is the number of different covariance matrices

    ## reshape by collapsing discrete dimensions
    shapes = q.shape, nus.shape, lambdas.shape  # store shapes
    q = q.reshape(-1)
    nus = nus.reshape((n_discrete_states, n_cg))
    lambdas = lambdas.reshape((n_covmats, n_cg, n_cg))

    p = np.zeros(n_discrete_states)
    mus = np.zeros(nus.shape)
    sigmas = np.empty(lambdas.shape)

    ## precompute determinants from normalization and sigmas
    dets = np.empty(n_covmats)
    for x in range(n_covmats):
        sigmas[x, :, :] = np.linalg.inv(lambdas[x, :, :])
        dets[x] = np.linalg.det(sigmas[x, :, :])
        # print(x, dets[x])

    qx_store = np.empty(n_discrete_states)
    for x in range(n_discrete_states):
        covindex = x * cov_variable
        nu_x = nus[x, :]
        q_x = q[x]
        mu_x = np.dot(sigmas[covindex, :, :], nu_x)
        mus[x, :] = mu_x.reshape(-1)

        qx_store[x] = q_x + 0.5 * np.einsum('i,i', nu_x, mu_x)
        p[x] = dets[covindex]**0.5  # omitted constant part
        # (later normalize anyway)

    qx_store = np.exp(qx_store - np.max(qx_store))  # save exp
    # note: normalization is done later
    p = np.multiply(p, qx_store)
    p /= np.sum(p)

    p = p.reshape(shapes[0])
    mus = mus.reshape(shapes[1])
    sigmas = sigmas.reshape(shapes[2])

    return p, mus, sigmas


###############################################################################


class BaseModel(abc.ABC):
    """
    base class for all models
    """

    def __init__(self):
        self.n_latent = 0  # number of latent variables
        self.annotations = {}  # textual information about the model
        # (e.g. how it was learned)

        self.meta = {}
        self.name = 'base'

    def get_numberofedges(self, threshold=10E-2):
        """ return number of edges in the graph"""
        graph = self.get_graph(threshold=threshold)
        return int(np.sum(graph) / 2)

    @abc.abstractmethod
    def get_group_mat(self, diagonal=False, norm=True):
        """return matrix of group norms """
        raise NotImplementedError  # Implement in derived class

    def get_meta(self):
        """get dictionary of meta data"""
        return self.meta

#    @abc.abstractmethod
#    def get_params(self):
#        """ return parameters of the class"""
#        raise NotImplementedError  # Implement in derived class

    @abc.abstractmethod
    def get_graph(self, threshold=1e-2):
        """ return graph of the model"""
        raise NotImplementedError  # Implement in derived class

    def repr_graphical(self,
                       diagonal=True,
                       graph=False,
                       threshold=None,
                       caption='',
                       norm=True,
                       save=False,
                       vmax=None):
        """ a graphical representation of the groupnorm for PW and CLZ models
        (this method is overwritten by the Model_PWSL model class)

        experimental code"""
        grpnormat = self.get_group_mat(diagonal=diagonal, norm=norm)
        vmin = np.min(grpnormat)
        if vmax is None:
            vmax = np.max(grpnormat)

        if not norm:
            cmap = cm.coolwarm
            # center colormap at zero
            max_range = max((np.abs(vmin), vmax))
            vmin = -max_range
            vmax = max_range
        else:
            cmap = cm.viridis
            cmap = cm.Greys
        if graph:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
            if not threshold is None:
                modelgraph = self.get_graph(threshold=threshold)
            else:
                modelgraph = self.get_graph()
            axes.flat[0].matshow(modelgraph * np.max(grpnormat),
                                 cmap=cmap,
                                 interpolation='nearest')
            axes.flat[0].set_title('%s graph' % (self.name), y=1.10)

            im0 = axes.flat[1].matshow(grpnormat,
                                       interpolation='nearest',
                                       cmap=cmap,
                                       vmin=vmin,
                                       vmax=vmax)
            axes.flat[1].set_title('%s (%s) edge weights' %
                                   (self.name, caption),
                                   y=1.10)
            fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=.7)
        else:
            fig = plt.figure(figsize=(6, 5))
            axes = fig.gca()

            im0 = axes.matshow(grpnormat,
                               interpolation='nearest',
                               vmin=vmin,
                               vmax=vmax,
                               cmap=cmap)
            axes.set_title('%s pairwise parameters %s' % (self.name, caption),
                           y=1.10)

            fig.colorbar(im0, ax=axes, shrink=.7)
        if save:
            fig.savefig('savedmodel.pdf', bbox_inches='tight')

        plt.show()


###############################################################################


class BaseModelPW(BaseModel):
    """
    this class specializes to standard PW model and S+L model,
    see the classes ModelPW and ModelPWSL
    """

    def __init__(self,
                 pw_params: (dict, tuple, list),
                 meta: dict,
                 annotations: dict = None,
                 in_padded: bool = True):
        BaseModel.__init__(self)

        if not annotations is None:
            self.annotations.update(annotations)

        assert 'n_cg' in meta
        assert 'n_cat' in meta
        assert 'sizes' in meta
        assert len(meta['sizes']) == meta['n_cat']

        self.meta = meta.copy()  # TODO(franknu): sanity checks

        self.meta['ltot'] = int(np.sum(meta['sizes']))  # np.sum([]) is 0.0
        self.meta['cat_glims'] = np.cumsum([0] + meta['sizes'])

        self.modeltype = get_modeltype(self.meta)

        assert isinstance(pw_params, (list, tuple, dict))
        # TODO(franknu): more assertions for sizes of components
        if isinstance(pw_params, (list, tuple)):
            self.vec_u, mat_q, self.mat_r, self.alpha, self.mat_lbda = pw_params

            if not in_padded:  # pad components
                theta = _theta_from_components(mat_q, self.mat_r, self.mat_lbda)
                theta = pad(theta, self.meta['sizes'])
                mat_q, self.mat_r, self.mat_lbda = _split_theta(
                    theta, self.meta['ltot'])
        else:
            # retrieve parameters from dictionary
            if self.meta['n_cat'] > 0:
                if 'u' in pw_params:
                    self.vec_u = pw_params['u']
                else:
                    assert 'pw_mat' in pw_params
                    ind = pw_params['pw_mat'].shape[0] - self.meta['n_cg']
                    self.vec_u = np.diag(pw_params['pw_mat'][:ind, :ind])
            else:
                self.vec_u = np.empty(0)
            if self.meta['n_cg'] > 0:
                assert 'alpha' in pw_params
                self.alpha = pw_params['alpha']
            else:
                self.alpha = np.empty(0)
            if 'pw_mat' in pw_params:
                theta = pw_params['pw_mat']
                if not in_padded:
                    theta = pad(theta, self.meta['sizes'])
                mat_q, self.mat_r, self.mat_lbda = _split_theta(
                    theta, self.meta['ltot'])

        self.vec_u = self.vec_u.flatten()
        self.alpha = self.alpha.flatten()
        if not in_padded:
            self.vec_u = pad(self.vec_u, meta['sizes'])

        # zero out block diagonal of mat_q
        # since we store univariate effects extra in vec_u
        self.mat_q = mat_q.copy()  # safe: operate on copy
        old_glim = 0
        for glim_r in self.meta['cat_glims'][1:]:
            self.mat_q[old_glim:glim_r, old_glim:glim_r] = 0
            old_glim = glim_r

        self.alpha = self.alpha.flatten()

    def _get_group_mat(self,
                       mat_q,
                       mat_r,
                       mat_lbda,
                       diagonal: bool = False,
                       aggr: bool = True,
                       norm: bool = True):
        """
        calculate group aggregations, e.g., the l_2 norms of all groups

        diagonal ... if False and aggr=True, then exclude groups
                     of univariate parameters (that are on the diagonal)

        aggr ... if True, aggregate values of group into single value

        norm ... if True, use np.linalg.norm for aggregation, else use np.sum
        """
        if not aggr:  # do not use an aggregation functions for the groups
            return _theta_from_components(mat_q, mat_r, mat_lbda)

        if norm:
            func = np.linalg.norm
        else:
            func = np.sum

        dim = self.meta['n_cat'] + self.meta['n_cg']
        aggrmat = np.zeros((dim, dim))
        glims = self.meta['cat_glims']

        for r in range(self.meta['n_cat']):  # dis-dis
            for j in range(self.meta['n_cat']):
                aggrmat[r, j] = func(mat_q[glims[r]:glims[r + 1],
                                           glims[j]:glims[j + 1]])

            for s in range(self.meta['n_cg']):
                aggrmat[self.meta['n_cat']+s, r] = aggrmat[r, self.meta['n_cat']+s] \
                    = func(mat_r[s, glims[r]:glims[r+1]])
        if norm:
            aggrmat[self.meta['n_cat']:, self.meta['n_cat']:] = np.abs(mat_lbda)
        else:
            aggrmat[self.meta['n_cat']:, self.meta['n_cat']:] = -mat_lbda

        if not diagonal:
            aggrmat -= np.diag(np.diag(aggrmat))

        return aggrmat

    def get_group_mat(self, **kwargs):
        """ returns matrix of group norms of the direct interactions"""
        return self._get_group_mat(self.mat_q, self.mat_r, self.mat_lbda,
                                   **kwargs)

    def get_graph(self, threshold=1e-1):
        """ calculate group norms of the parameters associated with
        each edge and threshold

        plot graph if disp is True"""
        # perhaps do variable threshold for different group types
        grpnormmat = self._get_group_mat(self.mat_q,
                                         self.mat_r,
                                         self.mat_lbda,
                                         diagonal=False,
                                         norm=True)
        graph = grpnormmat > threshold

        for i in range(graph.shape[0]):
            graph[i, i] = False  # no edges on diagonal

        return graph

    def _get_meanparams(self, pwparams: (tuple, list)):
        """
        convert pairwise parameters to mean parameters
        p(x) ~ (2pi)^{n/2}|La^{-1}|^{1/2}exp(q(x) + 1/2 nu(x)^T La^{-1} nu(x) )
        mu(x) = La^{-1}nu(x)
        Sigma = La^{-1}

        with nu(x) = alpha + R D_x and q(x) = u^T D_x + 1/2 D_x^T Q D_x.
        Here D_x is the dummy representation of the categorical values in x.
        """
        vec_u, mat_q, mat_r, alpha, mat_lbda = pwparams
        assert vec_u.shape == (self.meta['ltot'],)
        assert alpha.shape == (self.meta['n_cg'],)

        if self.meta['n_cat'] == 0 and self.meta['n_cg'] == 0:
            raise "Empty model"

        if self.meta['n_cat'] == 0:
            sigma = np.linalg.inv(mat_lbda)
            return np.empty(0), np.dot(sigma, alpha), sigma

        n_discrete_states = np.prod(self.meta['sizes'])
        # np.prod([]) would be 1.0, but n_cat>0 here

        q = np.empty(n_discrete_states)
        nus = np.zeros((n_discrete_states, self.meta['n_cg']))

        for x in range(n_discrete_states):
            unrvld_ind = np.unravel_index([x], self.meta['sizes'])

            mat_dx = np.zeros(self.meta['ltot'])
            for r in range(self.meta['n_cat']):
                mat_dx[self.meta['cat_glims'][r] +
                       unrvld_ind[r][0]] = 1  # dummy repr
            nus[x, :] = alpha + np.dot(mat_r, mat_dx)
            q[x] = np.inner(vec_u, mat_dx) + \
                0.5 * np.inner(mat_dx, np.dot(mat_q, mat_dx))

        return canon_to_meanparams(
            (q.reshape(self.meta['sizes']), nus, mat_lbda))

    def get_thetaalpha(self, **kwargs):
        """return compound pairwise interaction parameter matrix and alpha,
        that is, univariate continuous parameter"""
        theta = self.get_pw_mat(**kwargs)
        return theta, self.alpha

    def get_pw_mat(self, padded: bool = True, addunivariate: bool = False):
        """
        returns the pairwise matrix of direct interaction parameters.
        in the case of the standard pairwise model this is the matrix of all
        pairwise interaction parameters

        padded ... Boolean to control whether return matrix
                    is padded with zero rows/cols
                    (corresponding to first levels of discrete variables)

        precmatsign ... sign of the cts-cts interaction,
                        use negative sign to obtain valid representation
                        of Schur complement
        """

        mat_pw = _theta_from_components(self.mat_q, self.mat_r, self.mat_lbda)
        # inverts sign of mat_lbda

        if addunivariate:
            assert len(self.vec_u.shape) <= 1
            mat_pw[:self.meta['ltot'], :self.meta['ltot']] += 2 * np.diag(
                self.vec_u)
        if not padded:
            return unpad(mat_pw, self.meta['sizes'])

        return mat_pw

    @abc.abstractmethod
    def sample(self, n: int, gibbs_iter: int = 10):
        """sample n data points from the model"""
        raise NotImplementedError
