#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:34:33 2018

@author: frank (translation to Python)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file reimplements the PGADM algorithm described in 
% "Alternating Direction Methods for Latent Variable Gaussian Graphical 
% Model Selection", appeared in Neural Computation, 2013,
% by Ma, Xue and Zou, for solving 
% Latent Variable Gaussian Graphical Model Selection  
% min <R,SigmaO> - logdet(R) + alpha ||S||_1 + beta Tr(L) 
% s.t. R = S - L,  R positive definte,  L positive semidefinite 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np
import scipy # use scipy.linalg.eigh (for real symmetric matrix), does not check for symmetry
from scipy import optimize

from cgmodsel.models.model_pwsl import Model_PWSL
from cgmodsel.CG_base_ADMM import grp_soft_shrink as soft_shrink
from cgmodsel.CG_base_ADMM import l21norm as l1norm


class GLH_PWSL_ADMM:
    """
    solve the problem 
       min l(S-L) + lambda * ||S||_1 + rho * tr(L)
       s.t. S-L>0, L>=0
    where l is the Gaussian likelihood with pairwise parameters Theta=S-L,
    here Theta is the precision matrix (inverse of the covariance matrix)
    of the z(unnormalized) zero-mean Gaussian model
        p(y) ~ exp(-1/2 y^T Theta y)
        
    Solver is an ADMM algorithm with proximal gradient step    
    
    """
    
    def __init__(self, meta):
        self.dg = meta['dg']
        self.totalnumberofparams = 2 * self.dg ** 2
        
        self.alpha, self.beta = None, None
        self.lbda, self.rho = None, None
        
        self.opts = {}
    
    def __str__(self):
        s='<ADMMsolver> la=%s'%(self.lbda) + ', rho=%s'%(self.rho)
        s+=', alpha=%s'%(self.alpha) + ', beta=%s'%(self.beta)
        s+=', sc_a=%s'%(self.scale_l1) + ', sc_b=%s'%(self.scale_nuc)
        
        return s
        
    def drop_data(self,Y):
        """drop data"""
        self.n, self.dg = Y.shape

        assert not np.any(np.isnan(Y))
        self.Sigma0 = np.dot(Y.T, Y) / self.n # empirical second-moment matrix
        
        self.unscaledlbda = np.sqrt(np.log(self.dg) / self.n)
        self.scales = self.unscaledlbda, self.unscaledlbda
    
    def set_regularization_params(self, hyperparams, set_direct=False,
                                  scales=None, ptype='std'):
        """set regularization parameters        
        
        hyperparams ... pair of regularization parameters
        
        ptype ... if 'std', then lambda, rho = hyperparams
                    min l(S-L) + la * ||S||_1 + rho * tr(L)
                    s.t. S-L>0, L>=0
                  else assume that alpha, beta = hyperparams and
                  alpha, beta are weights in [0,1] and the problem is
                   min (1-alpha-beta) * l(S-L) + alpha * ||S||_1 + beta * tr(L)
                   s.t. S-L>0, L>=0
        
        In addition to the specified regularization parameters, 
        the regularization parameters can be scaled by a fixed value (depending
        on the number of data points and variables):
            
        set_direct ... if False, then do not use scaling
        
        scales ... if None, use standard scaling np.sqrt(log(dg)/n)
                   else  scales must be a two-tuple, and lambda and rho are 
                   scaled according to the elements of this two-tuple
        """
        if scales is None:
            self.scale_l1, self.scale_nuc = self.scales
        else:
            self.scale_l1, self.scale_nuc = scales
        if ptype == 'std': # standard regularization parameters, first for l21, second for nuclear norm
            self.lbda, self.rho = hyperparams
            
            if not set_direct:
                assert self.n > 0, "No data provided"
                self.lbda *= self.scale_l1
                self.rho *= self.scale_nuc
        else: # convex hyperparams are assumed
            alpha, beta = hyperparams
            self.alpha = alpha
            self.beta = beta
#            assert alpha+beta < 1, "must contain likelihood part"
            denom = 1 - alpha - beta

            if denom != 0:
                self.lbda = self.scale_l1 * alpha / denom
                self.rho = self.scale_nuc * beta  /denom
            else:
                self.lbda, self.rho = 0, 0

    def get_canonicalparams(self):
        """converts self.currentsolution into a Model_PWSL instance
        
        output: Model_PWSL class for sparse + low-rank parameters
        """
        S, L = self.currentsolution
        
        Q = np.empty(0)
        R = np.empty(0)
        u = np.empty(0)
        alpha = np.zeros(self.dg) # at this time only zero-mean solver

        can_pwsl = u, Q, R, alpha, S, L
        meta = {'dc': 0, 'dg': self.dg, 'sizes': []}
        return  Model_PWSL(can_pwsl, meta)


    def _set_defaults(self):
        """default solver options"""
        self.opts.setdefault('continuation', 1)
        self.opts.setdefault('mu', self.dg) # ADMM augmented Lagrangian parameter
        self.opts.setdefault('num_continuation', 10)
        self.opts.setdefault('eta', .25)
        self.opts.setdefault('muf', 1e-6)
        self.opts.setdefault('maxiter', 500)
        self.opts.setdefault('stoptol', 1e-5 )
        self.opts.setdefault('tau', 0.6) # proximal gradient step parameter
        self.opts.setdefault('verb', 0)
        self.opts.setdefault('off', 0)

    def solve(self, report=0, **opts):
        """
        solve the problem 
        min(S, L)  l(S-L) + la*||S||_1 + rho*||L||_* s.t. S-L>0, L>=0
        """

#        if self.n < self.dg:
#            print('Warning: degenerate Sigma0 (n < d)...')
        
        self.opts.update(opts)
        if report:
            print('>solver options', self.opts)
        self._set_defaults()       
        ## treat all cases
        """
        if not self.lh: # exclude likelihood from objective
            # (0,0) is a solution, but not feasible for likelihood
            # (S, 0) with S ~ eps*diag(1) asymptotically yields feasible solution
            # thus, objective may be arbitrarily close to 0, but 0 is not achieved
            eps = 1e-2
            out ={}
            out['S'] = eps * np.eye(self.dg)
            out['L'] = np.zeros((self.dg, self.dg))
            out['obj'] = self.get_objective(out['S'], out['L'])
#            print(self.get_lhvalue(out['S']))
            out['iter'] = 1
            out['message'] = b'ANALYTICAL SOLUTION'
            out['resid'] = np.nan
        """

        if not self.alpha is None: # use convex combinations of objective parts
            if self.alpha + self.beta == 1:
                if self.beta == 1:
                    print("Warning (objective consists only of nuclear norm)... any (S, 0) is a solution")
                else:
                    print('Warning (no lh in objective)...')
                ## no likelihood in objective
                # only need to assure that S-L is PD
                out = {}
                out['L'] = np.zeros((self.dg, self.dg)) # holds definitely if beta>0
                eps = 1e-8
                out['S'] = eps * np.eye(self.dg) # small PD matrix, will produce large logdet value
                out['message'] = b'APPROXIMATE SOLUTION wo LIKELIHOOD PART (small but feasible)'
                out['obj'] = self.unscaledlbda * self.alpha * eps * self.dg # f_vec compares better
                out['iter'] = 1
                out['resid'] = np.nan
                out['objdiff'] = np.nan
            elif self.alpha == 0:
                out = self._solve_unregularized(self.opts) # experimental
            else:
                out = self._solve()
        else: # use lambda and rho as regularization parameters
            out = self._solve()
            
        out['vec'] = self.get_x(out['S'], out['L'])
        out.setdefault('objdiff', None)
        
        self.currentsolution = out['S'], out['L']
        if report:
            print('>', out['message'])
            print('> ADMM_R: obj: %e, iter: %d, resid:%e' %
                  (out['obj'],out['iter'], out['resid']))
            
            print('> f_vec', self.get_f_vec(out['vec']).T)
            print('> |obj_ADMM - obj_SL|=', out['objdiff'])
            S = out['S']
            L = out['L']
            print('> S-norm=%.4f, L-norm=%.4f'%(np.linalg.norm(S), np.linalg.norm(L)))
            
            if not self.alpha is None:
                if self.alpha +self.beta !=1:
                    obj = self.get_objective(S, L) # this computes logdet(S-L) instead of logdet(R)
                    print('true objective =', obj)
                else:
                    f_vec = np.squeeze(self.get_f_vec(self.get_x(S, L)))
                    print('vector objective = ', f_vec)
                    obj = self.get_objective(S, L, lhonly=True)
                    print('true likelihood value', obj)

        return out
    
    def _solve(self):
        """
        solve the problem
          min l(S-L) + lambda * ||S||_1 + rho * tr(L)
        
        with ADMM, where one update is solved approximately with a proximal gradient
        step. Re-implementation of the solver from
        "Alternating Direction Methods for Latent Variable Gaussian Graphical 
        Model Selection", appeared in Neural Computation, 2013,
        by Ma, Xue and Zou"
        """

        ## initialization 
        verb = self.opts['verb']
        ABSTOL = 1e-5
        RELTOL = 1e-5
        n = self.Sigma0.shape[1] 
        Theta = np.eye(n)
        S = Theta
        L = np.zeros((n,n))
        Lambda = np.zeros((n,n))
        
        mu = self.opts['mu']
        eta = self.opts['eta']
        tau = self.opts['tau']
        
        hist = np.empty((5, self.opts['maxiter'] + 1))
        history = {}
        history['objval'] = hist[0, :]
        history['r_norm'] = hist[1, :]
        history['s_norm'] = hist[2, :]
        history['eps_pri'] = hist[3, :]
        history['eps_dual'] = hist[4, :]
        
        for it in range(1, self.opts['maxiter'] + 1):
            ## update Theta
            B = mu * self.Sigma0 - mu * Lambda - S + L

            d, U = scipy.linalg.eigh(B)

            eigTheta = (-d + np.sqrt(np.square(d) + 4 * mu)) / 2
            # always positive (though might round to 0 due to bad precision)

            Theta = np.dot(np.dot(U, np.diag(eigTheta)), U.T)
            Theta = (Theta + Theta.T) / 2;

            S_old = S
            L_old = L
            
            ## update S and L 
            Gradpartial = Theta - S + L - mu * Lambda
            G = S + tau * Gradpartial
            H = L - tau * Gradpartial
            S = soft_shrink(G, tau * mu * self.lbda, off=self.opts['off'])
            S = (S + S.T) / 2 
            
            d, U = scipy.linalg.eigh(H)
            d = d - tau * mu * self.rho
            d[d < 1e-25] = 0
            eigL = d
            L = np.dot(np.dot(U, np.diag(eigL)), U.T)
            L = (L + L.T) / 2
            
            ## update Lambda 
            resid = Theta - S + L
            Lambda = Lambda - resid / mu
            Lambda = (Lambda + Lambda.T) / 2 
            
            ## diagnostics, reporting, termination checks
            k = it
            
            # ADMM objective
            history['objval'][k]  = self.ADMMobjective(Theta, eigTheta, S,
                                         eigL, off=self.opts['off'])

            Sfronorm = np.linalg.norm(S, 'fro')
            Lfronorm = np.linalg.norm(L, 'fro')
            Tfronorm = np.linalg.norm(Theta, 'fro')
            
            rnorm = np.linalg.norm(resid,'fro')
            snorm = np.sqrt(np.linalg.norm(S - S_old, 'fro') ** 2 +
                   np.linalg.norm(L - L_old, 'fro') ** 2) / mu

            eps_pri = np.sqrt(3 * n ** 2) * ABSTOL + RELTOL * max((
                    Tfronorm ** 2, Sfronorm ** 2 + Lfronorm ** 2))
            eps_dual = np.sqrt(3 * n ** 2) * ABSTOL + RELTOL * np.linalg.norm(Lambda,'fro')

            history['r_norm'][k]  = rnorm
            history['s_norm'][k]  = snorm
        
            history['eps_pri'][k] = eps_pri
            history['eps_dual'][k]= eps_dual
    
            resid = np.linalg.norm(resid,'fro') / max([1, Sfronorm, Lfronorm, Tfronorm]) 
        
            if verb:
                print('%3d\t%10.4f %10.4f %10.4f %10.4f %10.2f %10.2f' %(k, 
                    history['r_norm'][k], history['eps_pri'][k], 
                    history['s_norm'][k], history['eps_dual'][k],
                    history['objval'][k], resid))
    
            obj = history ['objval'][k]  
            
            ## check stop 
            if resid < self.opts['stoptol']:
                break 
            
#            if rnorm < eps_pri and snorm < eps_dual and resid < opts['stoptol']:
#                break
    
            if self.opts['continuation'] and it % self.opts['num_continuation']==0:
                mu = max((mu * eta, self.opts['muf']))

        out = {}
        if resid < self.opts['stoptol']:
            out['message'] = b'CONVERGENCE: resid < stoptol'
        else:
            out['message'] = b'STOP: TOTAL NO. of ITERATIONS EXCEEDS LIMIT'
        
        out['Theta'] = Theta
        out['S'] = S
        out['L'] = L
        out['eigTheta'] = eigTheta
        out['eigL'] = eigL
        out['resid'] = resid
        out['iter'] = it

        obj2 = self.get_objective(S, L)
        out['objdiff'] = np.abs(obj - obj2) # difference of ADMM and real objective
        out['obj'] = obj # always feasible
        
#        print('ADMM_fopt:',obj)
#        print('norm(R-S+L)=', np.linalg.norm(R-S+L))
        if verb:
            print('norm(Theta-S+L)=', np.linalg.norm(resid))
#        a = np.sum(np.sum(np.multiply(R,self.Sigma0)))
#        b = np.sum(np.sum(np.multiply(S-L,self.Sigma0)))
#        c = np.sum(np.sum(np.multiply(R-S+L,self.Sigma0)))
#        print(c)
        
        return out
    
    def preprocess(self, x):
        """unpack flat parameter vector into S, L"""
        S = x[:self.dg **2].reshape((self.dg, self.dg))
        L = x[self.dg **2:].reshape((self.dg, self.dg))
        
        return S, L

    def ADMMobjective(self, Theta, eigTheta, S, eigL, off=False):
        """ADMM objective value, called during ADMM iterations """
        obj = np.sum(np.sum(np.multiply(Theta, self.Sigma0)))
        obj -= np.sum(np.log(eigTheta))
        obj += self.lbda * l1norm(S, off=off) + self.rho * np.sum(eigL)
        
        return obj

    def get_objective(self, S, L, alpha=None, lhonly=False):
        """return objective value
        alpha is a dummy parameter for compatibility with other solvers
        (non-zero alpha not yet implemented here)"""
        Theta = S - L
        obj = np.sum(np.sum(np.multiply(Theta, self.Sigma0)))
        obj -= np.linalg.slogdet(Theta)[1]
        if not lhonly:
            obj += self.lbda * l1norm(S, off=self.opts['off'])
            obj += self.rho * np.trace(L)
        
        return obj

    def _solve_unregularized(self, opts):
        """
        solve unregularized problem: This has an analytical solution 
        if n >= dg
        
        otherwise the problem is degenerate, solver is non-robust in this case
        """
        bnds = self.dg ** 2 * [(-np.inf, np.inf)]
        correctionpairs = min(len(bnds) - 1, 10)

        out = {}
        out['L'] = np.zeros((self.dg, self.dg))
        if self.n >= self.dg: # non-degenerate Sigma0:
            out['S'] = np.linalg.inv(self.Sigma0)
            out['obj'] = self.get_objective(out['S'], out['L'])
            out['iter'] = 1
            out['message'] = b'ANALYTICAL SOLUTION'
            out['resid'] = 0
            out['objdiff'] = 0
        else:
            print("Warning: Solving unregularized problem with degenerate sample covariance... likely inaccurate and non-robust...")
            res = optimize.minimize(self.get_lhvalue_sq, np.eye(self.dg), 
                                    method='L-BFGS-B', 
                                    jac=False, bounds=bnds, 
                                    options={'maxcor': correctionpairs,
                                               'maxiter':opts['maxiter'], 'ftol':opts['stoptol']},
                                    callback=None)

            x = res.x.reshape((self.dg, self.dg))
            out['S'] = np.dot(x, x.T)
            out['obj'] = res.fun
            out['iter'] = res.nit
            out['message'] = res.message
            out['resid'] = np.nan
        
        return out

    def get_lhvalue_sq(self, theta):
        """objective for unregularized problem with singular second-moment matrix
        """
        Theta = theta.reshape((self.dg, self.dg))
        Theta = np.dot(Theta, Theta.T) # positive definite parameter matrix
        return np.sum(np.sum(np.multiply(Theta, self.Sigma0))) -  np.linalg.slogdet(Theta)[1]

    def crossvalidate(self, x):
        """calculate a cross validation score (use after dropping test data) """
        S, L = self.preprocess(x)
   
        # smooth part
        Theta = S - L
        logdet_tiLambda = np.linalg.slogdet(Theta)[1] # logdet of PSD matrix (hopefully also PD)
        trace = np.trace(np.dot(Theta, self.Sigma0))
        
        lval_testdata = trace - logdet_tiLambda

        return -logdet_tiLambda, trace, lval_testdata
        # note: this does not compute individual errors on the nodes (in contrast to PLH cross validation)

    def get_f_vec(self, x):
        """vector objective value (for Benson algorithm) """
        S, L = self.preprocess(x)
   
        # smooth part
        Theta = S - L
        logdet_tiLambda = np.linalg.slogdet(Theta)[1] # logdet of PSD matrix (hopefully also PD)

        fsmooth = np.trace(np.dot(Theta, self.Sigma0)) - logdet_tiLambda

        # l1 part
        l1sum = self.scale_l1 * l1norm(S, off=self.opts['off'])
        
        # nuclear norm
        f_trace = self.scale_nuc * np.sum(np.diag(L))

        return np.array([np.squeeze(fsmooth), l1sum, f_trace])

#    def get_lh(self, nlh=None, x=None):
#        """
#        objective for ADMM (Lambda=S-L, n=dg):
#            <Lambda,Sigma0> - logdet(Lambda)
#        
#        normal Gaussian density:
#            (2pi)^(-n/2) det(Lambda)^(1/2) exp(-1/2 y^T Lambda y)
#        1/n*neg log LH:
#            -1/2 log(2pi) + 1/2<Lambda, Sigma0> - 1/2logdet(Lambda)
#        """
#        if not nlh is None: 
#            return np.exp(-self.dg * 0.5* (nlh - np.log(np.pi)))

    def _certificate(self, S, L, threshold=1e-3):
        """provides with a certificate that (S, L) is a solution
        based on subgradient characterizations
        
        experimental code, not safe to use"""
        assert self.opts['off'] == 0, "off == 1 not implemented"
        Theta = S - L # (negative L !!!)
        # obj = np.sum(np.sum(np.multiply(Theta,Sigma0))) -  np.linalg.slogdet(Theta)[1]
        #obj += alpha*l1norm_off(S) + beta*np.trace(L)
    
        lh_grad = self.Sigma0 - np.linalg.inv(Theta)
        print(self.lbda, self.rho)
        print('grad',lh_grad)

        ## projections on tangent spaces
        errS_t = 0
        suppS = np.where(np.abs(S)>=threshold)
        for i, j in zip(*suppS): # P_Q(grad) = -lbda*sign(S)
#            a = self.lbda*np.sign(S[i,j])
            b = lh_grad[i,j]

            errS_t += np.abs(-lh_grad[i,j] - self.lbda*np.sign(S[i,j]))
        print('>Error sparse tangential components=', errS_t)

        errS_n = 0
        suppSc = np.where(np.abs(S)<threshold)
        for i, j in zip(*suppSc): # ||P_Q^T(grad)||_infty <= lbda
            b = np.max([np.abs(lh_grad[i,j]) - self.lbda, 0])
#            print(lh_grad[i,j], self.lbda, b)
            errS_n += b
        print('>Error sparse normal components=', errS_n)

        ## check P_T(grad) == rho*UU^T
        d, U2 = scipy.linalg.eigh(L)
        # use restricted evd
        nonzeros = np.where(np.abs(d)>threshold)
        U = np.empty((self.dg, len(*nonzeros)))
        for i, ind in enumerate(*nonzeros):
            U[:, i] = U2[:, ind]
        P = np.dot(U, U.T)
        tmp = np.dot(lh_grad, P)
        P_Tgrad = np.dot(P, lh_grad) + tmp - np.dot(P, tmp)
        print('L',L)
        diff = P_Tgrad - self.rho*P
        print('P*rho',self.rho*P)
        print('P_T_grad',P_Tgrad)
#        print(diff)
        errL_t = np.linalg.norm(diff)
        print('>Error low-rank tangential components=', errL_t)
        
        ## check ||P_T^T(grad)||_2 <= rho
        P_Tperp_grad = lh_grad - P_Tgrad
#        I_P = np.eye(self.dg) - P
#        P_Tperp_grad2 = np.dot(np.dot(I_P, lh_grad), I_P)
#        print(P_Tperp_grad2-P_Tperp_grad)
        norm_orth = np.linalg.norm(P_Tperp_grad, 2)
#        e = np.linalg.eigvals(P_Tperp_grad)
#        print(e)
        errL_n = np.max([norm_orth - self.rho, 0])
        print('P_Tperp_grad_',P_Tperp_grad)
        print(norm_orth, self.rho)
        print('>Error low-rank normal components=', errL_n)

    def get_x(self, S, L):
        """pack S, L into a flat parameter vector """
        x = np.empty(self.totalnumberofparams)
        x[:self.dg **2] = S.flatten()
        x[self.dg **2:] =L.flatten()
        
        return x