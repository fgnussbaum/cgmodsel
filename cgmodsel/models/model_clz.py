# -*- coding: utf-8 -*-
"""
Copyright: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

This is experimental code
"""
import numpy as np

from cgmodsel.models.base import BaseModel, canon_to_meanparams

# pylint: disable=R0914 # too many local variable
# pylint: disable=R0913 # too many arguments

###############################################################################


class ModelCLZ(BaseModel):
    """
    class for parameters of distribution
    p(x,y) ~ exp(1/2 C_x Q C_x +  y^T R C_x -1/2 y^T[La0 + Las C_x]y
                  + u^T C_x + alpha^T y)

    this class uses padded parameters (that is, parameters for 0-th levels are
    included, however, one might want them to be constrained to 0 for
    identifiability reasons)

    here:
    [C_x .. dummy representation of x]
    u .. univariate discrete parameters
    alpha .. univariate cont. parameters
    with
    Q .. discrete-discrete interactions
    R .. discrete-cont. interactions
    La0 .. cont-cont interaction parameters
    Las .. cont-cont-discrete interaction parameters (n_cg x n_cg x ltot)

    initialize with tuple (u, Q, R, alpha, La0, Las)
    """

    def __init__(self, clz_params: (tuple, list), meta: dict):
        # meta must provided with n_cg, n_cat
        BaseModel.__init__(self)
        self.meta['n_cg'] = meta['n_cg']
        self.meta['n_cat'] = meta['n_cat']

        self.meta['sizes'] = meta['sizes']
        self.meta['cat_glims'] = np.cumsum([0] + self.meta['sizes'])
        self.meta['ltot'] = np.sum(self.meta['sizes'])

        self.name = 'CLZ'

        self.vec_u, self.mat_q, self.mat_r, self.alpha, \
            self.mat_lbda0, self.mat_lbdas = clz_params

    def __str__(self):
        """string representation of the model"""
        string = 'u:' + str(self.vec_u) + '\nQ:\n' + str(self.mat_q) + \
            '\nR:\n' + str(self.mat_r) + \
         '\nalpha:' + str(self.alpha) + '\nLambda0:\n' + str(self.mat_lbda0)
        for r in range(self.meta['n_cat']):
            for k in range(1, self.meta['sizes'][r]):
                # for k=0 self.mat_lbdas[:, :, rk] is zero (identifiability constraint)
                r_k = self.meta['cat_glims'][r] + k
                string += '\nLambda_%d:%d' % (r, k)
                string += str(self.mat_lbdas[:, :, r_k])
        return string

    def get_graph(self, threshold=1e-1):
        """ calculate group norms of the parameters associated with each edge and threshold
        plot graph if disp is True"""
        # perhaps do variable threshold for different group types
        grpnormmat = self.get_group_mat(diagonal=False, norm=True)
        graph = grpnormmat > threshold

        for i in range(graph.shape[0]):
            graph[i, i] = False  # no edges on diagonal

        return graph

    def get_group_mat(self, diagonal=False, norm=True, aggr=True):
        # calibration? optional class param?

        assert aggr, "only implemented for doing aggregation"
        assert norm, "l2-norm is the only implemented aggregation function"

        n_cat = self.meta['n_cat']
        n_cg = self.meta['n_cg']
        sizes = self.meta['sizes']
        dim = n_cat + n_cg
        glims = self.meta['cat_glims']
        ltot = self.meta['ltot']
        grpnormmat = np.zeros((dim, dim))

        for r in range(n_cat):  # dis-dis
            for j in range(r):
                grpnormmat[r, j] = \
                np.linalg.norm(self.mat_q[glims[r]:glims[r+1], glims[j]:glims[j+1]])
        self.mat_lbdas = self.mat_lbdas.reshape((n_cg * n_cg, ltot))

        for r in range(n_cat):
            tmp_group = np.empty(sizes[r] * n_cg)
            for s in range(n_cg):
                offset = s * n_cg

                tmp_group[:sizes[r]] = self.mat_r[s, glims[r]:glims[r + 1]]
                tmp_group[sizes[r]:(s+1) * sizes[r]] = \
                    self.mat_lbdas[offset: offset+s, glims[r]:glims[r+1]].flatten()
                tmp_group[(s+1) * sizes[r]:] = \
                    self.mat_lbdas[offset+s+1: offset+n_cg, glims[r]:glims[r+1]].flatten()

                grpnormmat[n_cat + s, r] = np.linalg.norm(tmp_group)

        tmp_group = np.empty(ltot + 1)
        for t in range(n_cg):  # upper triangle s<t
            for s in range(t):
                tmp_group[:ltot] = self.mat_lbdas[s * n_cg + t, :]
                tmp_group[ltot] = self.mat_lbda0[s, t]
                grpnormmat[n_cat + s, n_cat + t] = \
                    np.linalg.norm(tmp_group)

        grpnormmat += grpnormmat.T

        if not diagonal:
            grpnormmat -= np.diag(np.diag(grpnormmat))
        else:
            for s in range(n_cg):  # add diagonal of cts-cts interactions
                tmp_group[:ltot] = self.mat_lbdas[s * n_cg + s, :]
                tmp_group[ltot] = self.mat_lbda0[s, s]
                grpnormmat[n_cat + s, n_cat + s] = \
                    np.linalg.norm(tmp_group)

        self.mat_lbdas = self.mat_lbdas.reshape((n_cg, n_cg, ltot))

        return grpnormmat

    def get_params(self):
        """return parameters"""
        return self.vec_u, self.mat_q, self.mat_r, self.alpha, self.mat_lbda0, self.mat_lbdas

    def get_meanparams(self):
        """convert CLZ parameters into mean parameter representation
           (p(x)_x, mu(x)_x, Sigma(x)_x)
        Note: Some arrays might be empty
              (if not both discrete and continuous variables are present)

        ** conversion formulas to mean parameters **
        p(x) ~ (2pi)^{n/2}|La(x)^{-1}|^{1/2}exp(q(x) + 1/2 nu(x)^T La(x)^{-1} nu(x))
        mu(x) = La(x)^{-1}nu(x)
        Sigma(x) = La(x)^{-1}

        with nu(x) = alpha + R D_x and q(x) = u^T D_x + 1/2 D_x^T Q D_x
        and La(x) = Lambda0 + sum_r Lambda_r D_{x_r} = Lambda0 + Lambdas*D_x.
        Here D_x is the dummy representation of the categorical values in x.
        """
        assert self.meta['n_cat'] + self.meta['n_cg'] > 0

        sizes = self.meta['sizes']
        n_cat = self.meta['n_cat']
        n_cg = self.meta['n_cg']
        if self.meta['n_cat'] == 0:
            mat_sigma = np.linalg.inv(self.mat_lbda0)
            return np.empty(0), np.dot(mat_sigma, self.alpha), mat_sigma

        ## initialize mean params (reshape later)
        n_discrete_states = np.prod(sizes)
        q = np.zeros(n_discrete_states)

        ## discrete variables only
        if n_cg == 0:
            for x in range(n_discrete_states):
                unrvld_ind = np.unravel_index([x], sizes)
                # TODO(franknu): perhaps iter more systematically over dummy
                dummy = np.zeros((self.meta['ltot'], 1))
                for r in range(n_cat):  # construct dummy repr dummy of x
                    dummy[self.meta['cat_glims'][r] + unrvld_ind[r][0], 0] = 1
                q[x] = np.dot(self.vec_u.T, dummy) + 0.5 * np.dot(
                    dummy.T, np.dot(self.mat_q, dummy))
                canparams = q.reshape(sizes), np.empty(0), np.empty(0)
        else:
            precmatshape = (n_cg, n_cg)
            nus = np.empty((n_discrete_states, n_cg))
            lambdas = np.empty((n_discrete_states, n_cg, n_cg))

            for x in range(n_discrete_states):
                unrvld_ind = np.unravel_index([x], sizes)

                dummy = np.zeros((self.meta['ltot'], 1))
                for r in range(n_cat):  # construct full dummy repr dummy of x
                    dummy[self.meta['cat_glims'][r] + unrvld_ind[r][0], 0] = 1
                q[x] = np.dot(self.vec_u.T, dummy) + 0.5 * np.dot(
                    dummy.T, np.dot(self.mat_q, dummy))
                nus[x, :] = (self.alpha + np.dot(self.mat_r, dummy)).reshape(-1)
                lambdas[x, :, :] = self.mat_lbda0 + np.dot(
                    self.mat_lbdas, dummy).reshape(precmatshape)

                # the precmat components are "cols" in <lambdas>
                # (3rd component of the tensor),
                # the correct cols are selected by multiplying dummy
                # lambdas is a 3D tensor.
                # np.dot with a vector from the right maps to last component

            ## reshape to original forms
            q = q.reshape(sizes)
            nus = nus.reshape(sizes + [n_cg])
            lambdas = lambdas.reshape(sizes + [n_cg, n_cg])
            canparams = q, nus, lambdas
        return canon_to_meanparams(canparams)

    def is_valid(self) -> bool:
        """check if model represents a valid distribution, that is,
        check that all precision matrices are positive definite"""
        sizes = self.meta['sizes']
        n_discrete_states = np.prod(sizes)
        valid = True
        for x in range(n_discrete_states):
            unrvld_ind = np.unravel_index([x], sizes)

            dummy = np.zeros((self.meta['ltot'], 1))
            for r in range(
                    self.meta['n_cat']):  # construct full dummy repr dummy of x
                dummy[self.meta['cat_glims'][r] + unrvld_ind[r][0], 0] = 1

            lambda_x = self.mat_lbda0
            lambda_x += np.dot(self.mat_lbdas,
                               dummy).reshape(self.mat_lbda0.shape)

            tmp_eigvals = np.linalg.eigvals(lambda_x)
            tmp_min = np.min(tmp_eigvals)
            # assert tmp_min > 0, 'Non-PD covariance of (flat) state %d with la_min=%f.'
            # Pseudolikelihood estimation makes this possible.
            # Note that nodewise prediction using this model still works
            # (however it does not represent a valid joint distribution).'%(x, tmp_min)
            if tmp_min < 0:
                # print(x, lambdas[x, :, :], tmp_min)
                print('Warning: CLZ model has non-PD precision for state',
                      unrvld_ind, 'with smallest eigenvalue=%f' % (tmp_min))
                valid = False
        return valid
