# Copyright (c) 2019 Frank Nussbaum (frank.nussbaum@uni-jena.de)
"""
@author: Frank Nussbaum

base class for CG models and solvers
"""
import abc
from typing import Iterable
import numpy as np

#from cgmodsel.models.model_base import get_modeltype
from cgmodsel.models.model_pwsl import ModelPWSL
from cgmodsel.models.model_pw import ModelPW
from cgmodsel.utils import grp_soft_shrink, l21norm
# pylint: disable=W0511 # todos
# pylint: disable=R0914 # too many locals

DUMMY = 'dummy'
DUMMY_RED = 'dummy_red'
INDEX = 'index'
FLAT = 'flat'


def set_sparsity_weights(meta, cat_data, cont_data):
    """  use adjusted weights for all groups as suggested by LST2015
    (may be essential for "good", "consistent" results)"""
    n_data = meta['n_data']
    n_cg = meta['n_cg']
    n_cat = meta['n_cat']
    dim = n_cg + n_cat

    # CG variables
    mus = cont_data.sum(axis=0) / n_data
    sigmas_cg = np.sqrt((cont_data**2).sum(axis=0) / n_data - mus**2)
    # categoricals
    sigmas_cat = np.empty(n_cat)
    freqs = cat_data.sum(axis=0) / n_data
    for r in range(n_cat):
        sigma_r = 0
        for k in range(meta['sizes'][r]):
            p_xr_k = freqs[meta['cat_glims'][r] +
                           k]  # relative probability that x_r has value k
            sigma_r += p_xr_k * (1 - p_xr_k)
        sigmas_cat[r] = np.sqrt(sigma_r)

    weights = np.zeros((dim, dim))
    for r in range(n_cat):
        for j in range(r):
            weights[r, j] = sigmas_cat[r] * sigmas_cat[j]
        for s in range(n_cg):
            weights[n_cat + s, r] = sigmas_cat[r] * sigmas_cg[s]
    for j in range(n_cg):
        for i in range(j):
            weights[n_cat + j, n_cat + i] = sigmas_cg[j] * sigmas_cg[i]
    weights += weights.T
    # TODO(franknu): weights on diagonal

    return weights


###############################################################################
# base class for all CG model solvers
###############################################################################

class BaseCGSolver(abc.ABC):
    """
    base class for all CG model solver
    provides external interface to drop data along with meta information
    about this data
    """

    def __init__(self):
        """must call method drop_data after initialization"""
        super().__init__()

        self.cat_data = None  # discrete data, dropped later
        self.cat_format_required = None  # must override
        self.cont_data = None  # continuous data, dropped later

        #        self.problem_vars = None #TODO(franknu)
        self.meta = {'n_data': 0}

        if not hasattr(self, 'opts'):
            # since this may already be defined from other base classes
            self.opts = {}

        self.name = 'base'

    def _postsetup_data(self):
        """called after drop_data"""
        # may be overridden in derived classes
        # no pass because function has doc string

    def drop_data(self, data, meta: dict) -> None:
        """drop data, derived classes may perform additional computations

        uses and augments information contained in meta about the data

        categorical data must be provided in dummy-encoded form
        (leaving out 0-th levels if required by the solver)"""

        # process argument data
        if isinstance(data, tuple):
            assert len(data) == 2
            cat_data, cont_data = data
        else:
            counter = 0
            if 'n_cat' in meta and meta['n_cat'] > 0:
                counter += 1
                cat_data = data
                cont_data = np.empty((data.shape[0], 0))
            if 'n_cg' in meta and meta['n_cg'] > 0:
                counter += 1
                cont_data = data
                cat_data = np.empty((data.shape[0], 0))
            assert counter == 1, 'dictionary meta incompatible with provided data'

        self.cont_data = cont_data
        self.cat_data = cat_data

        self.meta = {}
        for key in ('n_cg', 'n_cat'):
            if key in meta:
                self.meta[key] = meta[key]
            else:
                meta[key] = 0

        if self.meta['n_cat'] > 0:
            assert 'sizes' in meta
            assert len(meta['sizes']) == meta['n_cat']
            self.meta['sizes'] = meta['sizes']

        # continue checking validity of meta
        if self.meta['n_cg'] > 0:
            assert not np.any(np.isnan(cont_data))
            assert meta['n_cg'] == cont_data.shape[1]

            self.meta['n_data'] = cont_data.shape[0]

        if self.meta['n_cat'] == 0:
            self.meta['sizes'] = []  # no discrete variables
            self.meta['ltot'] = 0
            self.meta['red_levels'] = False  # value irrelevant, no cat vars
            self.meta['cat_glims'] = [0]
        else:
            if 'n_data' in self.meta:
                assert self.meta['n_data'] == cat_data.shape[0]
            else:
                self.meta['n_data'] = cat_data.shape[0]

            ltot = np.sum(meta['sizes'])
            if self.cat_format_required == DUMMY:
                assert ltot == cat_data.shape[1]
                # 0-th levels of the discrete data are contained
                # for identifiability, assume that corresponding
                # parameters are constrained to zero
                self.meta['red_levels'] = False
            elif self.cat_format_required == DUMMY_RED:
                assert ltot - meta['n_cat'] == cat_data.shape[1]
                # assume that 0-th levels are left out in discrete data
                # assures identifiability of the model
                self.meta['red_levels'] = True
                self.meta['sizes'] = [size - 1 for size in meta['sizes']]
            elif self.cat_format_required == INDEX:
                assert meta['n_cat'] == cat_data.shape[1]
            elif self.cat_format_required == FLAT:  # MAP solver
                assert len(cat_data.shape) == 1
            else:
                raise Exception('invalid self.cat_format_required')

        if self.cat_format_required in (DUMMY, DUMMY_RED):
            self.meta['ltot'] = cat_data.shape[1]
            self.meta['dim'] = self.meta['ltot'] + self.meta['n_cg']

            # calculate cumulative # of levels/ group delimiters
            self.meta['cat_glims'] = np.cumsum([0] + self.meta['sizes'])

            self.meta['glims'] = list(self.meta['cat_glims']) + \
                [1 + self.meta['ltot'] + s for s in range(self.meta['n_cg'])]
            # TODO(franknu): self.meta['glims'] for sparse reg only

            self.meta['nonbinary'] = (self.meta['ltot'] > self.meta['n_cat'] *
                                      (2 - self.meta['red_levels']))

        self.meta['n_catcg'] = self.meta['n_cat'] + self.meta['n_cg']

        # self.meta['type'] = get_modeltype(self.n_cat, self.n_cg, self.sizes)
        #        fac = np.log(self.meta['n_cg'] + self.meta['n_cat'])
        fac = np.sqrt(np.log(self.meta['n_catcg']) / self.meta['n_data'])

        self.meta['reg_fac'] = fac  # potentially used as prescaling factor
        # for regularization parameters

        self._postsetup_data()

    def get_name(self):
        """return model name"""
        return self.name


class BaseSparseSolver(BaseCGSolver):
    """
    base class that contains proximal operators
    """
    def __init__(self, useweights=False):
        super().__init__()
        self.opts.setdefault('off', 0)  # if 1 regularize only off-diagonal
        # model options # TODO(franknu): find better place
        self.opts.setdefault('use_alpha', 1)
        self.opts.setdefault('use_u', 1)
        
        self.useweights = useweights
        self.weights = None

        self.name = 'base-sparse'

    def _postsetup_data(self):
        """called after drop_data"""
       
        if self.useweights:
            self.weights = set_sparsity_weights(self.meta,
                                                self.cat_data,
                                                self.cont_data)
        else:
            self.weights = None

    def shrink(self, mat_s, tau):
        """return (group)- soft shrink of matrix mat_s with tau """
        if self.meta['nonbinary']:
            return grp_soft_shrink(mat_s, tau,
                                   self.meta['n_cat'] + self.meta['n_cg'],
                                   self.meta['glims'],
                                   self.opts['off'],
                                   weights=self.weights)
        return grp_soft_shrink(mat_s,
                               tau,
                               off=self.opts['off'],
                               weights=self.weights)

    def sparse_norm(self, mat_s):
        """return l21/ l1-norm of mat_s"""
        if self.meta['nonbinary']:
            return l21norm(mat_s,
                           self.meta['n_cat'] + self.meta['n_cg'],
                           self.meta['glims'],
                           self.opts['off'],
                           weights=self.weights)
        return l21norm(mat_s, off=self.opts['off'], weights=self.weights)


class BaseGradSolver(abc.ABC):
    """
    Base solver for iterative (scipy L-BFGS-B) solvers
    provides with methods to pack/unpack parameters into vector
    """

    def __init__(self):
        #        print('Init BaseCGSolver')
        super().__init__()

        self.shapes = None
        self.n_params = None

        #        self.problem_vars = None

        if not hasattr(self, 'opts'):
            # should already be defined by other class
            self.opts = {}
        self._set_defaults()

    def _set_defaults(self):
        """default solver options"""
        self.opts.setdefault('verb', 1)  # write output

        ## objective variants
        self.opts.setdefault('use_alpha', 1)  # use univariate cts parameters?
        self.opts.setdefault('use_u', 1)  # use univariate discrete parameters?
        self.opts.setdefault('off', 0)  # if 1 regularize only off-diagonal

        ## stopping criteria and tolerancies
        #        self.opts.setdefault('abstol', 1e-5)
        #        self.opts.setdefault('reltol', 1e-5)
        self.opts.setdefault('tol', 1e-12)
        self.opts.setdefault('maxiter', 500)

#        self.opts.setdefault('useweights', False)
#        self.opts.setdefault('maxrank', -1)

#    @abc.abstractmethod
#    def get_bounds(self):
#        """get bounds"""
#        raise NotImplementedError # deferred to BaseHuber class

    @abc.abstractmethod
    def get_fval_and_grad(self, optvars, verb=0, **kwargs):
        """calculate function value and gradient for solver"""
        raise NotImplementedError

    def get_params(self, optvars):
        """a function to display the problem parameters"""
        params = self.unpack(optvars)
        for i, param in enumerate(params):
            print('%s:\n' % self.shapes[i][0], param)
        return params

    def pack(self, components):
        """pack (typically) gradients into vector x"""
        grad = np.empty(self.n_params)
        offset = 0
        for i, component in enumerate(components):
            size = np.prod(self.shapes[i][1])
            #            print(self.shapes[i][0], size, np.prod(component.shape))
            assert size == np.prod(component.shape)
            grad[offset:offset + size] = component.flatten()  # row-wise
            offset += size
        return grad

    def unpack(self, x) -> Iterable[np.ndarray]:
        """unpack model parameters from vector x, save: returns copy"""
        offset = 0
        params = []
        xcopy = x.copy()  # allows modifying the copy without modifying x
        for _, shapedim in self.shapes:
            tmp = np.prod(shapedim)
            params.append(xcopy[offset:offset + tmp].reshape(shapedim))
            offset += tmp

        return params


class BaseSolverSL(BaseSparseSolver):
    """
    base class for S+L model solvers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha, self.beta = None, None
        self.lbda, self.rho = None, None

        self.problem_vars = None

    def __str__(self):
        string = '<ADMMsolver> la=%s' % (self.lbda) + ', rho=%s' % (self.rho)
        string += ', alpha=%s' % (self.alpha) + ', beta=%s' % (self.beta)
        string += ', use_alpha=%d' % (self.opts.setdefault('use_alpha', 1))
        string += ', use_u=%d' % (self.opts.setdefault('use_u', 1))
        string += ', off=%d' % (self.opts.setdefault('off', 1))
        return string

    def get_canonicalparams(self):
        """Retrieves the PW S+L CG model parameters from flat parameter vector.

        output: Model_PWSL instance"""

        mat_s, mat_l, alpha = self.problem_vars

        ltot = self.meta['ltot']

        mat_lambda = -mat_s[ltot:, ltot:]  # cts-cts parameters
        # have negative sign in CG pairwise interaction parameter matrix

        if self.meta['n_cat'] > 0:

            glims = self.meta['cat_glims']
            sizes = self.meta['sizes']

            mat_q = mat_s[:ltot, :ltot]
            mat_r = mat_s[ltot:, :ltot]
            vec_u = 0.5 * np.diag(mat_q).copy().reshape(ltot)
            for r in range(self.meta['n_cat']):  # set block-diagonal to zero
                mat_q[glims[r]:glims[r+1],
                      glims[r]:glims[r+1]] = \
                      np.zeros((sizes[r], sizes[r]))

            if self.meta['red_levels']:
                fullsizes = [size + 1 for size in sizes]
            else:
                fullsizes = sizes
        else:
            mat_q = np.empty(0)
            mat_r = np.empty(0)
            vec_u = np.empty(0)
            fullsizes = []

        can_pwsl = vec_u, mat_q, mat_r, alpha, mat_lambda, mat_l

        annotations = {
            'n': self.meta['n_data'],
            'lambda': self.lbda,
            'rho': self.rho
        }

        meta = {
            'n_cat': self.meta['n_cat'],
            'n_cg': self.meta['n_cg'],
            'sizes': fullsizes
        }

        return ModelPWSL(can_pwsl,
                         meta,
                         annotations=annotations,
                         in_padded=False)

    def get_regularization_params(self):
        """get regularization parameters"""
        return self.lbda, self.rho

    def set_regularization_params(self,
                                  hyperparams,
                                  scales=None,
                                  set_direct=False,
                                  ptype: str = 'std') -> None:
        """set regularization parameters

        hyperparams ... pair of regularization parameters

        ptype ... if 'std',
                set lambda, rho = hyperparams * scaling(n, nvars), where
                the parameters are for the problem
                    min l(S-L) + lambda * ||S||_1 + rho * tr(L)
                    s.t. S-L>0, L>=0
                Here, scaling(n, nvars) is a scaling suggested by
                consistency results
                Argument <scales> is not used in this case!
              if 'direct', directly set lambda, rho = hyperparams
              if 'convex' assume that alpha, beta = hyperparams and
              alpha, beta are weights in [0,1] and the problem is
               min (1-alpha-beta) * l(S-L) + alpha * ||S||_1 + beta * tr(L)
               s.t. S-L>0, L>=0

        In addition to the specified regularization parameters,
        the regularization parameters can be scaled by a fixed value (depending
        on the number of data points and variables):

        scales ... if None, use standard scaling np.sqrt(log(dg)/n)
                   else  scales must be a two-tuple, and lambda and rho are
                   scaled according to the elements of this two-tuple
        """
        assert len(hyperparams) == 2
        assert hyperparams[0] >= 0 and hyperparams[1] >= 0

        if ptype == 'std':
            # standard regularization parameters
            # first for l21, second for nuclear norm
            self.lbda, self.rho = hyperparams
        elif ptype == 'convex':
            alpha, beta = hyperparams
            #            assert alpha + beta <= 1
            assert alpha + beta < 1, "must contain likelihood part"

            self.alpha = alpha
            self.beta = beta

            denom = 1 - alpha - beta

            if denom != 0:
                self.lbda = alpha / denom
                self.rho = beta / denom
#            else:
#                # no likelihood part
#                self.lbda, self.rho = 0, 0
        else:
            raise Exception('unknown ptype')

        if not set_direct:
            if not scales is None:
                scale_lbda, scale_rho = scales
            else:
                assert self.meta['n_data'] > 0, \
                    "data-dependent scaling, drop data first"
                # calculate prescaling factor for the regularization parameters
                # based on consistency analysis by Chandrasekaran et. al (2010)

                # assert 'reg_fac' in self.meta
                scale_lbda = self.meta['reg_fac']
                scale_rho = self.meta['reg_fac']
            self.lbda *= scale_lbda
            self.rho *= scale_rho


class BaseSolverPW(BaseSparseSolver):
    """
    base class for sparse graphical model solvers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = None
        self.lbda = None

        self.problem_vars = None

    def __str__(self):
        string = '<ADMMsolver> la=%s' % (self.lbda)
        string += ', use_alpha=%d' % (self.opts.setdefault('use_alpha', 1))
        string += ', use_u=%d' % (self.opts.setdefault('use_u', 1))
        string += ', off=%d' % (self.opts.setdefault('off', 1))
        return string

    def get_canonicalparams(self):
        """Retrieves the PW CG model parameters from flat parameter vector.

        output: Model_PW instance"""

        mat_s, alpha = self.problem_vars

        ltot = self.meta['ltot']

        mat_lambda = -mat_s[ltot:, ltot:]  # cts-cts parameters
        # have negative sign in CG pairwise interaction parameter matrix

        if self.meta['n_cat'] > 0:

            glims = self.meta['cat_glims']
            sizes = self.meta['sizes']

            mat_q = mat_s[:ltot, :ltot]
            mat_r = mat_s[ltot:, :ltot]
            vec_u = 0.5 * np.diag(mat_q).copy().reshape(ltot)
            for r in range(self.meta['n_cat']):  # set block-diagonal to zero
                mat_q[glims[r]:glims[r+1],
                      glims[r]:glims[r+1]] = \
                      np.zeros((sizes[r], sizes[r]))

            if self.meta['red_levels']:
                fullsizes = [size + 1 for size in sizes]
            else:
                fullsizes = sizes
        else:
            mat_q = np.empty(0)
            mat_r = np.empty(0)
            vec_u = np.empty(0)
            fullsizes = []

        can_params = vec_u, mat_q, mat_r, alpha, mat_lambda

        annotations = {
            'n': self.meta['n_data'],
            'lambda': self.lbda,
        }

        meta = {
            'n_cat': self.meta['n_cat'],
            'n_cg': self.meta['n_cg'],
            'sizes': fullsizes
        }

        return ModelPW(can_params,
                       meta,
                       annotations=annotations,
                       in_padded=False)

    def get_regularization_params(self):
        """get regularization parameters"""
        return self.lbda

    def set_regularization_params(self, regparam, set_direct=False,
                                  scale=None, ptype='std'):
        """set regularization parameters for
        min l(S) + la*||S||_{2/1}

        hyperparams ... pair of regularization parameters

        ptype ... if 'std',
                set lambda = regparam * scaling(n, nvars), where
                Here, scaling(n, nvars) is a scaling suggested by
                consistency results
                Argument <scales> is not used in this case!
              if 'direct', directly set lambda = regparam
              if 'convex' assume that alpha = regparam and
                min_S (1-alpha) * l(S) + alpha * ||S||_ {2,1}

        In addition to the specified regularization parameter,
        the regularization parameters can be scaled by a fixed value (depending
        on the number of data points and variables):

        scales ... if None, use standard scaling np.sqrt(log(dg)/n)
                   else  scales must be a nonnegative number with which lambda
                   is scaled
        """

        if ptype == 'std': # standard regularization parameter for l21-norm
            self.lbda = regparam

        elif ptype == 'convex': # convex hyperparams are assumed
            self.alpha = regparam
            assert self.alpha < 1, "must contain likelihood part"
            denom = 1 - self.alpha
            if denom != 0:
                self.lbda = self.alpha/denom
            else:
                self.lbda = 0
        else:
            raise Exception('unknown ptype')


        if not set_direct:
            if scale is None:
                assert self.meta['n_data'] > 0, \
                    "data-dependent scaling, drop data first"
                scale = self.meta['reg_fac']

            self.lbda *= scale
