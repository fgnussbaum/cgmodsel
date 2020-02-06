#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2020

"""
import time
import unittest
import numpy as np

from cgmodsel.admm import AdmmCGaussianPW
from cgmodsel.dataops import load_prepare_data  # function to read data

    
#@unittest.skip
class TestPWSolvers(unittest.TestCase):
    """"""
    def setUp(self):
        self.solver = AdmmCGaussianPW()

        opts = {'maxiter':500, 'continuation':1, 'off':1, 'verb':0, 'stoptol':1e-8,
                'lhproxtol':1e-10, 'cont_adaptive': 1, 'use_u':0}
        self.solver.opts.update(opts)
        self.solver.admm_param = 5
        self.ftol = 1e-4
        self.tic = time.time()
        
    def tearDown(self):
        toc = time.time() - self.tic
        print ("%s: %.3f(s)" % (self.id(), toc) )
    
#    @unittest.skip('')
    def test_cg_iris_std(self):
        """Iris mixed dataset standardized, 4 CG, 1 cat variable"""
        Scvx = np.array([[3.889,0,6.3105e-09,-1.6512,2.5069,1.5782],[0,-2.5207,-4.3413e-09,-1.347,3.9524,5.6075],[6.3105e-09,-4.3413e-09,-5.336,1.6439,6.0236,-0.7646],[-1.6512,-1.347,1.6439,-1.8813,-2.3312,0.74026],[2.5069,3.9524,6.0236,-2.3312,-18.7217,10.5061],[1.5782,5.6075,-0.7646,0.74026,10.5061,-12.1741]])
        alpha = np.zeros((1,4))
        fcvx = -1.59442

        refsol = Scvx, alpha, fcvx
        refopts = {'off':1, 'use_u':1, 'use_alpha':0}
        hyperparams = .1
        testname = "Iris_standardized"
        filename = "tests/data/iris_standardized_py.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)  


    def _joint(self, refsol, refopts, hyperparams, filename, testname):
        """joint work for all tests in this class"""
        self.solver.opts.update(refopts)
                
        D, Y, meta = load_prepare_data(filename,
                                       cattype='dummy_red',
                                       standardize=False)
        data = D, Y
        
        ## learn S+L model - PADMM
        self.solver.drop_data(data, meta)
        self.solver.set_regularization_params(hyperparams) 

#        res = self.solver.solve(report=0)
#        f_admm = res['admm_obj']

        Scvx, alpha, fcvx = refsol
        f_cvx = self.solver.get_objective(Scvx, alpha=alpha)
        self.assertAlmostEqual(f_cvx,
                               fcvx,
                               places = 3,
                               msg="%f not equal %f"%(f_cvx, fcvx))
        fdiff = f_admm - fcvx

        self.assertTrue(abs(fdiff) < self.ftol,
                        msg="%s:%f nequal %f"%(testname, fcvx, f_admm))

#        e = self.sparse_norm(mat_s) 
#        _, e2 = self.shrink(mat_s, 0)
#        print(e, e2)


if __name__ == '__main__':
    unittest.main()