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
        print ("%s: %.3f(s)" % (self.id(), toc))


#    @unittest.skip('')
    def test_bin1(self):
        """unit test u, off, binary data"""
        Scvx = np.array([[-1.8425,0.49746,0.43169],[0.49746,-1.3499,0.98679],[0.43169,0.98679,-2.0592]])
        alpha = np.empty((0,0))
        fcvx= 1.93996
        refsol = Scvx, alpha, fcvx
        refopts = {'off':1, 'use_u':1}
        hyperparams = .1
        testname = "DChain_s12d3(la=0.1)"
        filename = "tests/data/py_D_s12d3l0.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)

#    @unittest.skip('')
    def test_bin2(self):
        """unit test u, off, binary data"""
        Scvx = np.array([[-0.97913,2.9899e-09,1.1052e-09],[2.9899e-09,-0.44376,0.34489],[1.1052e-09,0.34489,-1.0643]])
        alpha = np.empty((0,0))
        fcvx = 2.02611

        refsol = Scvx, alpha, fcvx
        refopts = {'off':1, 'use_u':1}
        hyperparams = .5
        testname = "DChain_s12d3(la=0.5)"
        filename = "tests/data/py_D_s12d3l0.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)

#    @unittest.skip('')
    def test_cat1(self):
        """categorical data"""
        Scvx = np.array([[2.8422,0,0.94356,-0.81473,-0.8702,-0.99998],[0,-1.9862,1.2576,1.9049,-0.14209,-0.089824],[0.94356,1.2576,0.30823,0,-1.1237,-0.46153],[-0.81473,1.9049,0,2.4119,-0.76095,-2.311],[-0.8702,-0.14209,-1.1237,-0.76095,0.68253,0],[-0.99998,-0.089824,-0.46153,-2.311,0,2.2938]])
        alpha = np.empty((0,0))
        fcvx = 2.7248

        refsol = Scvx, alpha, fcvx
        refopts = {'off':1, 'use_u':1}
        hyperparams = .1
        testname = "Catchain_d3(la=0.1)"
        filename = "tests/data/Catchain_d3.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)

#    @unittest.skip('')
    def test_g1(self):
        """g continuous variables"""
        Scvx = np.array([[-0.99462,-0.38101,-0.018378],[-0.38101,-0.90362,-0.33924],[-0.018378,-0.33924,-0.9072]])
        alpha = np.array([[-0.072406],[-0.23669],[-0.32267]])
        fcvx = 1.60211
        refsol = Scvx, alpha, fcvx
        refopts = {'off':1,  'use_alpha':1}
        hyperparams = .1
        testname = "GChain_s12d3"
        filename = "tests/data/py_G_s12d3l0.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)   
    
#    @unittest.skip('')
    def test_cg1(self):
        """mixed dataset with 3 cat and 3 CG variables"""
        Scvx = np.array([[-3.0709,0.14249,0.093442,-0.98859,0.00232,-0.22641],[0.14249,-2.659,-8.2508e-11,-0.15589,-0.74287,-5.5561e-09],[0.093442,-8.2508e-11,-4.7627,-0.29749,-0.065125,-1.0433],[-0.98859,-0.15589,-0.29749,-1.0275,-0.40192,-0.16874],[0.00232,-0.74287,-0.065125,-0.40192,-1.1376,-0.34575],[-0.22641,-5.5561e-09,-1.0433,-0.16874,-0.34575,-1.0119]])
        alpha = np.array([[0.060857],[-0.19066],[-0.0032914]])
        fcvx = 2.78867
        refsol = Scvx, alpha, fcvx
        refopts = {'off':1, 'use_u':1, 'use_alpha':1}
        hyperparams = .1
        testname = "Dbl1Chain_s12d3_0L"
        filename = "tests/data/py_CG_s12d3l0.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)   

#    @unittest.skip('')
    def test_cg_iris_std(self):
        """Iris mixed dataset standardized, 4 CG, 1 cat variable"""
        Scvx = np.array([[3.889,0,6.3105e-09,-1.6512,2.5069,1.5782],[0,-2.5207,-4.3413e-09,-1.347,3.9524,5.6075],[6.3105e-09,-4.3413e-09,-5.336,1.6439,6.0236,-0.7646],[-1.6512,-1.347,1.6439,-1.8813,-2.3312,0.74026],[2.5069,3.9524,6.0236,-2.3312,-18.7217,10.5061],[1.5782,5.6075,-0.7646,0.74026,10.5061,-12.1741]])
        alpha = np.array([[-1.2986e-08],[0.9994],[-2.153],[-2.3953]])
        fcvx = -1.59442

        refsol = Scvx, alpha, fcvx
        refopts = {'off':1, 'use_u':1, 'use_alpha':1}
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
        
        ## learn model
        self.solver.drop_data(data, meta)
        self.solver.set_regularization_params(hyperparams) 

        res = self.solver.solve(report=0)

        f_admm = res['admm_obj']
#        f_admm2 = self.solver.get_objective(res['solution'][0])
#        print(f_admm2)

        Scvx, alpha, fcvx = refsol
#        f_cvx = self.solver.get_objective(Scvx, alpha=alpha)
#        self.assertAlmostEqual(f_cvx,
#                               fcvx,
#                               places = 3,
#                               msg="%f not equal %f"%(f_cvx, fcvx))
        fdiff = f_admm - fcvx

        self.assertTrue(abs(fdiff) < self.ftol,
                        msg="%s:%f nequal %f"%(testname, fcvx, f_admm))


if __name__ == '__main__':
    unittest.main()