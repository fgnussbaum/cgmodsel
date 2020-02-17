#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
import time
import unittest
#import logging # https://stackoverflow.com/questions/284043/outputting-data-from-unit-test-in-python


import numpy as np

from cgmodsel.map import MAP
from cgmodsel.dataops import load_prepare_data  # function to read data

#from tests.test_SL import TestSLSolvers

class TestMAP(unittest.TestCase):

    def setUp(self):
        self.solver = MAP()
        self.tol = 10E-3
        self.tic = time.time()
        
    def tearDown(self):
        toc = time.time() - self.tic
        print ("%s: %.3f(s)" % (self.id(), toc) )

#    @unittest.skip('')
    def test_cat1(self):
        refsol = (np.array([[[0.26,0.1],[0.14,0.11]],[[0.11,0.05],[0.07,0.16]]]), np.array([]), np.array([]))
        testname = "DChain_s12d3"
        filename = "tests/data/py_D_s12d3l0.csv"

        self._joint(refsol, filename, testname)

#    @unittest.skip('')
    def test_g1(self):
        mu = np.array([-0.01,-0.14,-0.3])
        Sigma = np.array([[1.26,-0.6,0.19],[-0.6,1.6,-0.58],[0.19,-0.58,1.34]])
        refsol = np.array([]), mu, Sigma
        testname = "GChain_s12d3"
        filename = "tests/data/py_G_s12d3l0.csv"

        self._joint(refsol, filename, testname)

#    @unittest.skip('')
    def test_cg1(self):
        p = np.array([[[0.46,0.06],[0.16,0.04]],[[0.12,0.06],[0.08,0.02]]])
        mus = np.array([[[[0.09,-0.21,0.06],[-0.04,0.39,-1.04]],[[0.43,-0.93,0.39],[0.08,-1.74,-0.8]]],[[[-0.62,-0.21,-0.18],[-1.23,1.1,-0.79]],[[-1.31,0.14,-0.01],[0.12,-1.3,-1.07]]]])
        Sigma = np.array([[1.04,-0.29,-0.04],[-0.29,0.9,-0.32],[-0.04,-0.32,1.06]])
        refsol = p, mus, Sigma
        testname = "DblChain_s12d3"
        filename = "tests/data/py_CG_s12d3l0.csv"

        self._joint(refsol, filename, testname)
        
#    @unittest.skip
#    def test_iris(self):
#        p = np.array([[[0.46,0.06],[0.16,0.04]],[[0.12,0.06],[0.08,0.02]]])
#        mus = np.array([[[[0.09,-0.21,0.06],[-0.04,0.39,-1.04]],[[0.43,-0.93,0.39],[0.08,-1.74,-0.8]]],[[[-0.62,-0.21,-0.18],[-1.23,1.1,-0.79]],[[-1.31,0.14,-0.01],[0.12,-1.3,-1.07]]]])
#        Sigma = np.array([[1.04,-0.29,-0.04],[-0.29,0.9,-0.32],[-0.04,-0.32,1.06]])
#        refsol = p, mus, Sigma
#        testname = "GChain_s12d3"
#        filename = "tests/data/py_G_s12d3l0.csv"
#
#        self._joint(refsol, filename, testname)

    def _joint(self, refsol, filename, testname):
        """joint work for all tests in this class"""
#        print(testname)
        D, Y, meta = load_prepare_data(filename,
                                       cattype='flat',
                                       standardize=False)
        data = D, Y
#        print(meta)
        self.solver.drop_data(data, meta)
        mparams_learned = self.solver.fit_fixed_covariance()
        p, mus, Sigmas = mparams_learned

        success = True
        for x_name, x_ref, x_learned in zip(('p', 'mus', 'Sigmas'),
                                            refsol,
                                            mparams_learned):
            shape = x_ref.shape
            if len(shape) > 0  and np.prod(shape) > 0:
#                print(x_ref, x_learned)
                diff = np.sum(np.abs(x_ref-x_learned))
                diff /= np.prod(shape)
                try:
                    self.assertTrue((diff < self.tol),
                        msg="%s:diff=%f for %s"%(testname, diff, x_name))
                except AssertionError as e:
                    success = False
                    print(e, '\n')
                    print(x_ref)
                    print(x_learned)
        self.assertTrue(success)
            


if __name__ == '__main__':
    unittest.main()