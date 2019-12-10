# Copyright (c) 2019 Frank Nussbaum (frank.nussbaum@uni-jena.de)
"""
@author: Frank Nussbaum

base class for CG model selection using Huber approximation of sparse norm

Experimental code
"""
import abc
import time
import numpy as np

from scipy.optimize import approx_fprime
from scipy import optimize

from cgmodsel.base_solver import BaseGradSolver, BaseCGSolver

# pylint: disable=W0613 # unused argument

def _huberapprox(array, delta):
    """returns Huber approximation + gradient of group of variables in array"""
    norm = np.sqrt(np.sum(np.multiply(array, array)))

    if norm <= delta:
        return 0.5 / delta * norm * norm, array / delta
    return (norm - delta / 2.0, array / norm)


def _reldiff(array1, array2):
    """relative error between array1 and array2 w.r.t. magnitudes of them
    - for gradient check"""
    diff = array1 - array2
    tmp = np.min([np.abs(array1), np.abs(array2)]) + 1
    # |min(a, b)| + 1 ??? or better use max(|a|, |b|) + 1 ???
    return diff / tmp


######### base class for all Huber model solvers ########################


class HuberBase(BaseGradSolver, BaseCGSolver):
    """
    base class Huber solvers
    """

    def __init__(self):
        super().__init__()
        self._reset()

        self.currentsolution = None

        self.niter = 0  # counter for # inner iters
        self.fcalls = 0  # calls to get_fval_and_grad
        self.total_fcalls = 0  # total # of calls to get_fval_and_grad
        self.outeriter = 0

    @abc.abstractmethod
    def get_bounds(self):
        """get bounds for solver"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_starting_point(self, random=False, seed=-1):
        """get starting point"""
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self, x):
        """preprocess input x"""
        raise NotImplementedError  # implemented in derived classes

    def _reset(self):
        """reset counters"""
        self.niter = 0  # counter for # inner iters
        self.fcalls = 0  # calls to get_fval_and_grad
        self.total_fcalls = 0  # total # of calls to get_fval_and_grad

    def _getfunctioncalls(self):
        """get function calls"""
        self.total_fcalls += self.fcalls  # sum up calls over inner iterations
        fcalls = self.fcalls
        self.fcalls = 0
        return fcalls

    def nocallback(self, x, **kwargs):
        """callback that does basically nothing"""
        self._getfunctioncalls()

    def slimcallback(self, x, delta=None, iter0=False, **kwargs):
        """callback"""
        fcalls = self._getfunctioncalls()
        if not iter0:
            self.niter += 1
        flh, glh = self.get_fval_and_grad(x,
                                          smooth=True,
                                          sparse=False,
                                          verb='-')
        # flh, glh belong to likelihood part of the objective
        freg, greg = self.get_fval_and_grad(x,
                                            delta=delta,
                                            smooth=False,
                                            sparse=True,
                                            verb='-')
        # freg, greg belong to regularization part of the objective

        grad = glh + greg

        self.fcalls = 0
        #        print (self.niter, flh+freg, flh, freg, np.linalg.norm(grad), fcalls)
        print('*it%d f=%f f_lh=%f f_reg=%f gnorm=%f fcalls=%d' %
              (self.niter, flh + freg, flh, freg, np.linalg.norm(grad), fcalls))

    def callback(self,
                 x,
                 approxgrad=1,
                 delta=1,
                 sparse=False,
                 reshapedprinting=0,
                 iter0=False):
        """a callback function that serves primarily for debugging"""
        fcalls = self._getfunctioncalls()
        if not iter0:
            self.niter += 1

        print('*******iter%d, fcalls=%d, delta=%f' %
              (self.niter, fcalls, delta))

        fval, grad = self.get_fval_and_grad(x,
                                            sparse=sparse,
                                            delta=delta,
                                            verb='cb')

        if approxgrad:
            func_handle_f = lambda x: self.get_fval_and_grad(
                x, delta=delta, sparse=sparse, verb='prox')[0]
            eps = np.sqrt(np.finfo(float).eps)  # ~1.49E-08 at my machine
            gprox = approx_fprime(x, func_handle_f, eps)

        if reshapedprinting:
            print('**g_exct_%d' % (self.niter))
            self.print_params(grad)

        if approxgrad:
            diff = grad - gprox
            reldiff = np.empty(grad.size)
            for i in range(grad.size):
                reldiff[i] = _reldiff(grad[i], gprox[i])
            if reshapedprinting:
                print('**rel_diff')
                self.print_params(grad)
            print('graddev=', np.linalg.norm(diff), 'rel_graddev=',
                  np.linalg.norm(reldiff))
        self.fcalls = 0
        print('f%d=' % (self.niter), fval)

    def _solve(self,
               ftol=10E-10,
               callback=None,
               maxiter=1000,
               explicitbounds=None,
               **kwargs):
        """solve the problem starting from recent configuration,
        updates self.currentsolution"""
        self._reset()  # set self.fcalls, self.total_fcalls to 0, etc.

        handle_fg = lambda x: self.get_fval_and_grad(x, **kwargs)
        if explicitbounds is None:
            bnds = self.get_bounds()
        else:
            bnds = explicitbounds  # TODO: assertions

        callback(self.currentsolution, iter0=True)

        correctionpairs = min(len(bnds) - 1, 10)
        res = optimize.minimize(handle_fg,
                                self.currentsolution,
                                method='L-BFGS-B',
                                jac=True,
                                bounds=bnds,
                                options={
                                    'maxcor': correctionpairs,
                                    'maxiter': maxiter,
                                    'ftol': ftol
                                },
                                callback=callback)

        callback(res.x)  # since no call to callback after last solver iteration

        self.currentsolution = res.x

        return res

    def solve(self,
              seed=-1,
              callback=None,
              random=True,
              x_init=None,
              maxiter=1000,
              explicitbounds=None,
              tracktime=False):
        """ solve an unregularized problem"""

        ## starting point
        if x_init is None:
            self.currentsolution = self.get_starting_point(random=random,
                                                           seed=seed)
        else:
            self.currentsolution = x_init

        if callback is None:
            callback = self.nocallback
        if tracktime:
            t_start = time.time()
        if explicitbounds is None:
            probname = 'unregularized problem'
        else:
            probname = 'graph-constrained problem'

        res = self._solve(callback=callback,
                          maxiter=maxiter,
                          explicitbounds=explicitbounds)

        if tracktime:
            t_end = time.time()
            print('Elapsed solver time (%s) = %f secs.' %
                  (probname, t_end - t_start))
        return res

    def solve_sparse(self,
                     innercallback=None,
                     random=False,
                     seed=10,
                     verb=True,
                     x_init=None,
                     delta0=20,
                     maxiter_inner=100,
                     maxiter_outer=10,
                     ftol=10E-10,
                     ftol_inner=10E-12,
                     disp_converg_msgs=False,
                     tracktime=False,
                     earlystopping_inner=True):
        """solve l_{1,2}-norm regularized problem, i.e. min l(x) + norm(x, '2/1').
        The solver uses Huber smoothening of the l2/1 regularization term
        this solves a sequence of smoothened problems that are parametrized by delta,
        i.e. smoothening takes place on the interval (-delta, delta)

        delta... initial smoothening parameter (is stepwise decreased)
        klbda... scaling factor of regularization parameter
        innercallback... optional callback function for inner iterations,
        e.g. methods callback, slimcallback provided by this class
        verb... if output is printed
        seed... seed (if random starting point is used)
        x0.. optional warm start to solution
        maxiter_inner... maximum number of iterations when solving one subproblem
        """

        #        if self.name == 'PW' and self.lbda == 0:  # unregularized problem
        #            return self.solve(callback=innercallback,
        #                              x_init=x_init,
        #                              tracktime=tracktime)

        ## starting point
        if x_init is None:
            if seed != -1:
                self.currentsolution = self.get_starting_point(random=random,
                                                               seed=seed)
            else:
                self.currentsolution = self.get_starting_point(random=False)
        else:
            self.currentsolution = x_init

        if innercallback is None:
            innercallback = self.nocallback

        if tracktime:
            t_start = time.time()

        delta_f = 1
        previousf = np.inf

        self.outeriter = 0
        delta = delta0

        notfinished = True
        while notfinished:
            self.outeriter += 1
            callbackwithsparsity = lambda x, iter0=False: innercallback(
                x, delta=delta, sparse=True, iter0=iter0)
            if earlystopping_inner:
                # scheme with increasing accuracy
                # reduce by 2 to really obtain that many iterations
                maxiters = maxiter_inner * self.outeriter - 2
            else:
                maxiters = maxiter_inner * 10  # TODO(franknu): this is ad hoc

            ## solve problem with current Huber smoothing
            res = self._solve(callback=callbackwithsparsity,
                              maxiter=maxiters,
                              delta=delta,
                              sparse=True,
                              ftol=ftol_inner)

            if disp_converg_msgs:
                print(res.message)

            if verb:
                self._outer_callback(res, delta)

            delta_f = abs(previousf - res.fun)
            # note: function value can increase between outer iters
            #            print(delta_f, res.fun)
            previousf = res.fun
            delta /= 10  # decrease interval for smoothening

            ## termination criteria
            # delta should at least be below 1e-5
            # (terminate after this delta if solver did not reach maxiters)
            # otherwise do further outer iterations (until delta=1e-8 or so)
            # after min iterations also use delta_f<ftol as criterion
            iter1 = 6
            cond_maxiter = self.outeriter >= maxiter_outer
            cond_f = (self.outeriter >= iter1) & (delta_f < ftol)
            # cond_f = True means that f did not change much
            cond_notstopped = (self.outeriter >= iter1) & \
                (not res.message.startswith(b'STOP'))
            if cond_maxiter or cond_f or cond_notstopped:
                notfinished = False
#                print(cond_maxiter, cond_f, cond_notstopped)

        if not res.message.startswith(b'CONV'):
            s = '\x1b[32m%s\x1b[0m%s' % (
                'Warning: Solving may be inaccurate, solver message is: ',
                res.message)
            print(s)


#        self.print_params(self.currentsolution)

        if tracktime:
            t_end = time.time()
            dic = {
                'PW': 'PW l1-regularized problem',
                'SL': 'S+L regularized problem',
                'CLZ': 'CLZ l1-regularized problem'
            }
            print('Elapsed solver time (%s) = %f secs.' %
                  (dic[self.name], t_end - t_start))

        return res

    def _outer_callback(self, res, delta):
        """outer callback"""
        x = res.x

        flh, glh = self.get_fval_and_grad(x,
                                          smooth=True,
                                          sparse=False,
                                          verb='-')
        freg, greg = self.get_fval_and_grad(x,
                                            delta=delta,
                                            smooth=False,
                                            sparse=True,
                                            verb='-')
        gnorm = np.linalg.norm(glh + greg)
        print('#oit%d f=%.6f f_lh=%.4f f_reg=%.4f |g|=%.2f fcalls=%d, de=%f' %
              (self.outeriter, flh + freg, flh, freg, gnorm, self.total_fcalls,
               delta))
        if not res.message.startswith(b'CONV'):
            print(res.message)

    def print_params(self, x):
        """print params"""
        tparams = self.preprocess(x)

        for i in range(len(self.shapes)):
            if np.prod(self.shapes[i][1]) > 0:
                print(self.shapes[i][0])
                print(tparams[i])
