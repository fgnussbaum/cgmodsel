#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
import abc
import numpy as np

#import time

# pylint: disable=R0914


class BaseAdmm(abc.ABC):
    """
    base class for ADMM solvers
    """

    def __init__(self):
        #        print('Init BaseAdmm')
        super().__init__()
        self.admm_param = 1  # TODO(franknu): external access

        self._set_defaults_admm()

    @abc.abstractmethod
    def _do_iter_admm(self, current_vars: tuple):
        """perform computations of one ADMM iteration"""
        raise NotImplementedError  # override in derived classes

    @abc.abstractmethod
    def _initialize_admm(self):
        """initialize ADMM variables"""
        raise NotImplementedError  # override in derived classes

#    def _collect(self, current_vars: tuple):
#        """store information of ADMM variables in dictionairy"""
#        raise NotImplementedError  # override in derived classes

    def _report(self, out):
        """print stats after solving"""
        print('>', out['message'])
        print('> (P)ADMM: obj: %e, iter: %d, resid:%e' %
              (out['obj'], out['iter'], out['resid']))
        try:
            print('> regparams used:', self.get_regparams())
        except AttributeError:
            # Method get_regparams does not exist
            pass

        for i, var in enumerate(out['solution']):
            print('norm optvar%d=%f' % (i, np.linalg.norm(var)))

    def _set_defaults_admm(self):
        """default solver options"""
        self.opts = {}
        self.opts.setdefault('abstol', 1e-5)
        self.opts.setdefault('reltol', 1e-5)

        self.opts.setdefault('continuation', 1)
        self.opts.setdefault('num_continuation', 10)
        #        self.opts.setdefault('cont_adaptive', 10)
        self.opts.setdefault('continuation_fac', 2)
        #        self.opts.setdefault('eta', .25)
        #        self.opts.setdefault('muf', 1e-6)
        self.opts.setdefault('maxiter', 500)
        self.opts.setdefault('stoptol', 1e-5)

        self.opts.setdefault('verb', 0)

    def _solve(self):
        """
        solve a generic ADMM problem
        """
        ## initialization
        current_vars = self._initialize_admm()

        hist = np.empty((5, self.opts['maxiter'] + 1))
        history = {
            'objval': hist[0, :],
            'r_norm': hist[1, :],
            's_norm': hist[2, :],
            'eps_pri': hist[3, :],
            'eps_dual': hist[4, :]
        }

        for i in range(1, self.opts['maxiter'] + 1):
            ## update Theta
            current_vars, residuals, stats = self._do_iter_admm(current_vars)

            ## diagnostics, reporting, termination checks
            rnorm, snorm, eps_pri, eps_dual = residuals

            history['objval'][i] = stats['admm_obj']

            history['r_norm'][i] = rnorm
            history['s_norm'][i] = snorm

            history['eps_pri'][i] = eps_pri
            history['eps_dual'][i] = eps_dual

            if self.opts['verb']:
                print('%3d\t%10.4f %10.4f %10.4f %10.4f %10.2f' %
                      (i, history['r_norm'][i], history['eps_pri'][i],
                       history['s_norm'][i], history['eps_dual'][i],
                       history['objval'][i]))  # resid %10.2f

            ## check stop
            pridualresids_below_tolerance = (rnorm < eps_pri and
                                             snorm < eps_dual)

            if pridualresids_below_tolerance:
                break

            if self.opts['continuation'] and \
                i % self.opts['num_continuation'] == 0:
                # self-adaptive update
                # generally: larger admm_param leads to lower penalty on
                # primal residual (resid)
                admm_param_old = self.admm_param

                admm_state = eps_pri, eps_dual, rnorm, snorm

                if (i / self.opts['num_continuation']) % 5 == 1:
                    self.cont_update_2019(admm_state)
                else:
                    self.cont_update_s2a(admm_state)

                if self.opts['verb'] and self.admm_param != admm_param_old:
                    print('>> New ADMM parameter', self.admm_param,
                          '(old was %f)' % (admm_param_old))

        out = {}
        if pridualresids_below_tolerance:
            out['message'] = b'CONVERGENCE: primal and dual residual below tolerance'
        else:
            out['message'] = b'STOP: TOTAL NO. of ITERATIONS EXCEEDS LIMIT'

        out['history'] = history
        out['iter'] = i
        # out['admm_obj'] = history['objval'][i] # already in stats, see below

        out.update(stats)

        return out

    def cont_update_s2a(self, admm_state, criticalratio: int = 5):
        """ update S2a from my master thesis
        uses more robust constant size updates of admm_param"""

        eps_pri, eps_dual, rnorm, snorm = admm_state

        snorm_rel = snorm / eps_dual
        rnorm_rel = rnorm / eps_pri
        if snorm_rel < 1 and rnorm_rel < 1:
            return  # residuals are already smaller than tolerancies

#        print(snorm_rel, rnorm_rel)

        ## do scaling of admm_param if necessary
        if rnorm_rel > snorm_rel * criticalratio:
            # decrease admm_param
            self.admm_param /= self.opts['continuation_fac']
        elif snorm_rel > rnorm_rel * criticalratio:
            # increase admm_param
            self.admm_param *= self.opts['continuation_fac']

    def cont_update_2019(self,
                         admm_state,
                         criticalratio: int = 5,
                         cutoff: tuple = (1e-2, 1e2)):
        """ update that respects current ratio of residuals with tolerancies"""
        # TODO(franknu): consider speed of improvement instead of
        # just current state? - hist as class variable??
        # ML to learn update?
        eps_pri, eps_dual, rnorm, snorm = admm_state
        snorm_rel = snorm / eps_dual
        rnorm_rel = rnorm / eps_pri

        if snorm_rel < 1 and rnorm_rel < 1:
            return  # residuals are already smaller than tolerancies

        # do cutoff
        lower, upper = cutoff
        snorm_rel = min(upper, max((snorm_rel, lower)))
        rnorm_rel = min(upper, max((rnorm_rel, lower)))

        if snorm_rel < 1:
            # decrease ADMM param
            # increases penalty for violations of primal feasibiity
            scaling_factor = 1 / rnorm_rel
        elif rnorm_rel < 1:
            # increase ADMM param
            # decreases penalty for violations of primal feasibiity
            scaling_factor = snorm_rel
        else:
            scaling_factor = snorm_rel / rnorm_rel

#        if self.opts['verb']:
#            print(snorm_rel, rnorm_rel)

        ## do scaling of admm_param if necessary
        if max((scaling_factor, 1 / scaling_factor)) > criticalratio:

            if rnorm_rel >= 1 and snorm_rel >= 1:
                scaling_factor = (scaling_factor)**.8
                # inhibition, draw towards 1

            if self.opts['verb']:
                print('scaling_factor', scaling_factor)
            self.admm_param *= scaling_factor

    def solve(self, report=0, **kwargs):
        """solve the problem """

        self.opts.update(**kwargs)  # update solver options

        if report:
            print('>solver options', self.opts)

        ## select appropriate sparsity-inducing norms and corresponding proximity operators

        out = self._solve()

        if report:
            self._report(out)

        if not out['message'].startswith(b'CONV'):
            print(out['message'])

        return out
