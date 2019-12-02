#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
import numpy as np
import abc
#import time

#################################################################################
# ADMM sparse norms and shrinkage operators
#################################################################################

def grp_soft_shrink(S, tau, groupdelimiters=None, Lcum=None, off=False):
    """
    calculate (group-)soft-shrinkage function of matrix S with shrinkage parameter tau
    soft shrink if no groupdelimiters are given
    else must provide with groupdelimiters (sizes of groups) and
    cumulative sizes of groups (Lcum)
    
    this code could be made much faster
    (by parallizing loops, efficient storage access)
    """
    if groupdelimiters is None:
        # soft shrink
        tmp = np.abs(S) - tau
        tmp[tmp < 1e-25] = 0
        
        tmp2 = np.multiply(np.sign(S), tmp)
        if off:
            tmp2 -= np.diag(np.diag(tmp2))
            tmp2 += np.diag(np.diag(S))
        return tmp2

    tmp = np.empty(S.shape)

    for i in range(len(groupdelimiters)):
        for j in range(len(groupdelimiters)):
            g = S[Lcum[i]:Lcum[i+1], Lcum[j]:Lcum[j+1]]
            if (i == j) and off:
                tmp[Lcum[i]:Lcum[i+1], Lcum[i]:Lcum[i+1]] = g
                continue
#            print(g.shape, (sizes[i], sizes[j]))
            gnorm = np.linalg.norm(g, 'fro')
            if gnorm <= tau:
                tmp[Lcum[i]:Lcum[i+1], Lcum[j]:Lcum[j+1]] = np.zeros(g.shape)
            else:
                tmp[Lcum[i]:Lcum[i+1], Lcum[j]:Lcum[j+1]] = g * (1 - tau / gnorm)
    
    return tmp

def l21norm(S, groupdelimiters=None, Lcum=None, off=False):
    """
    calculate l_{2,1}-norm or l_1-norm of S
    l_1-norm if no groupdelimiters are given
    else must provide with groupdelimiters (sizes of groups) and
    cumulative sizes of groups (Lcum)
    """
    if groupdelimiters is None:
        tmp = np.sum(np.abs(S.flatten()))
        if off:
            tmp -= np.sum(np.diag(np.abs(S)))
        return tmp
        # calculate regular l1-norm
    s = 0
    for i in range(len(groupdelimiters)):
        for j in range(i):
            g = S[Lcum[i]:Lcum[i+1], Lcum[j]:Lcum[j+1]]
            s += np.linalg.norm(g, 'fro')
    
    s *= 2
    if not off:
        for i in range(len(groupdelimiters)):
            g = S[Lcum[i]:Lcum[i+1], Lcum[i]:Lcum[i+1]]
            s += np.linalg.norm(g, 'fro')

    return s


class BaseAdmm(abc.ABC):
    """
    base class for ADMM solvers
    """
    def __init__(self):
        print('Init BaseAdmm')
        super().__init__()
        self.admm_param = 1

        self._set_defaults_admm()


    @abc.abstractmethod
    def _do_iter_admm(self, current_vars: tuple):
        """perform computations of one ADMM iteration"""
        raise NotImplementedError # overwrite in derived classes
        
    @abc.abstractmethod
    def _initialize_admm(self):
        """initialize ADMM variables"""
        raise NotImplementedError # overwrite in derived classes

#    def _collect(self, current_vars: tuple):
#        """store information of ADMM variables in dictionairy"""
#        raise NotImplementedError  # overwrite in derived classes

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
            print('norm optvar%d'%(i, np.linalg.norm(var)))
            
    def _set_defaults_admm(self):
        """default solver options"""
        self.opts = {}
        self.opts.setdefault('abstol', 1e-5)
        self.opts.setdefault('reltol', 1e-5)
        
        self.opts.setdefault('continuation', 1)
        self.opts.setdefault('num_continuation', 10)
#        self.opts.setdefault('eta', .25)
#        self.opts.setdefault('muf', 1e-6)
        self.opts.setdefault('maxiter', 500)
        self.opts.setdefault('stoptol', 1e-5 )

        self.opts.setdefault('verb', 0)

    def _solve(self):
        """
        solve a generic ADMM problem
        """
        ## initialization 
        current_vars = self._initialize()
        
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
            current_vars, residuals, stats = self._do_iter(current_vars)
            
            ## diagnostics, reporting, termination checks
            eps_pri, eps_dual, rnorm, snorm = residuals
            
            history['objval'][i] = stats['admm_obj']

            history['r_norm'][i]  = rnorm
            history['s_norm'][i]  = snorm
        
            history['eps_pri'][i] = eps_pri
            history['eps_dual'][i]= eps_dual
    
            if self.opts['verb']:
                print('%3d\t%10.4f %10.4f %10.4f %10.4f %10.2f %10.2f' %(i, 
                    history['r_norm'][i], history['eps_pri'][i], 
                    history['s_norm'][i], history['eps_dual'][i],
                    history['objval'][i])) # resid
            
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
                if self.opts['cont_adaptive']:
                    if (i / self.opts['num_continuation']) % 5 == 1:
                        self.cont_update_2019(admm_state)
                    else:
                        self.cont_update_s2a(admm_state)
 
                    if self.admm_param != admm_param_old:
                        print('>> New ADMM parameter', self.admm_param, 
                              '(old was %f)'%(admm_param_old))

        out = {}
        if pridualresids_below_tolerance:
            out['message'] = b'CONVERGENCE: primal and dual residual below tolerance'
        else:
            out['message'] = b'STOP: TOTAL NO. of ITERATIONS EXCEEDS LIMIT'
        
        out['history'] = history
        out['iter'] = i
#        out['admm_obj'] = history['objval'][i] # already in stats, see below
        
        out.update(stats)
        
        return out

    def cont_update_s2a(self,
                        admm_state,
                        criticalratio: int = 5):
        """ update S2a from my master thesis
        uses more robust constant size updates of admm_param"""
    
        eps_pri, eps_dual, rnorm, snorm = admm_state
    
        snorm_rel = snorm / eps_dual
        rnorm_rel = rnorm / eps_pri
        if snorm_rel < 1 and rnorm_rel < 1:
            return  # residuals are already smaller than tolerancies
    
#        print(snorm_rel, rnorm_rel)
        
        # do scaling of admm_param if necessary
        if rnorm_rel > snorm_rel * criticalratio:
            # decrease admm_param
            self.admm_param = self.admm_param / self.opts['cont_tau_inc']  
        elif snorm_rel > rnorm_rel * criticalratio:
            # increase admm_param
            self.admm_param = self.admm_param * self.opts['cont_tau_inc']

    
    def cont_update_2019(self,
                         admm_state,
                         criticalratio: int = 5,
                         cutoff: tuple = (1e-2, 1e2)):
        """ update that respects current ratio of residuals with tolerancies"""
    
        eps_pri, eps_dual, rnorm, snorm = admm_state
        snorm_rel = snorm / eps_dual
        rnorm_rel = rnorm / eps_pri
    
        if snorm_rel < 1 and rnorm_rel < 1:
            return # residuals are already smaller than tolerancies
    
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
        
        self.opts.update(**kwargs) # update solver options
        
        if report:
            print('>solver options', self.opts)

        ## select appropriate sparsity-inducing norms and corresponding proximity operators
        
        out = self._solve()
        


        if report:
            self._report(out)
            
        if not out['message'].startswith(b'CONV'):
            print(out['message'])

        return out