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
from cgmodsel.admm_pwsl import AdmmCGaussianSL
#, AdmmGaussianSL
from cgmodsel.dataops import load_prepare_data  # function to read data


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
            
    
#@unittest.skip
class TestSLSolvers(unittest.TestCase):
    """"""
    def setUp(self):
        self.solver = AdmmCGaussianSL()

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
    def test_bin1(self):
        """unit test no u, binary data"""
        Scvx = np.array([[0,7.2413e-13,-1.6943e-13],[7.2413e-13,0,0.38863],[-1.6943e-13,0.38863,0]])
        Lcvx = np.array([[1.8426e-13,1.6744e-13,1.2959e-14],[1.6744e-13,7.4865e-13,1.9285e-13],[1.2959e-14,1.9285e-13,1.8283e-13]])
        fcvx = 2.0615
        refsol = Scvx, Lcvx, fcvx
        refopts = {'off':1, 'use_u':0}
        hyperparams = .2, .2
        testname = "DChain_s12d3_1L"
        filename = "tests/data/py_D_s12d3l1.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)

#    @unittest.skip('')
    def test_bin2(self):
        Scvx = np.array([[-2.0148,7.8604e-14,-5.2736e-15],[7.8604e-14,-2.0109,4.14e-13],[-5.2736e-15,4.14e-13,-2.6093]])
        Lcvx = np.array([[0.24652,0.51234,0.43764],[0.51234,1.0648,0.90954],[0.43764,0.90954,0.77693]])
        fcvx = 1.9644
        refsol = Scvx, Lcvx, fcvx
        refopts = {'off':1, 'use_u':1}
        hyperparams = .2, .2
        testname = "DChain_s12d3_1L"
        filename = "tests/data/py_D_s12d3l1.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)

#    @unittest.skip('')
    def test_cat1(self):
        """categorical data"""
        Scvx = np.array([[1.7508,-0.48957,0.45206,-1.446,-3.2523e-12,-3.0942e-12],[-0.48957,-1.9733,1.1795,1.6568,2.6379e-13,1.4208e-12],[0.45206,1.1795,-0.2606,-0.58319,-0.61309,0.18918],[-1.446,1.6568,-0.58319,1.5069,-0.10094,-1.3125],[-3.2523e-12,2.6379e-13,-0.61309,-0.10094,0.41389,-1.2358],[-3.0942e-12,1.4208e-12,0.18918,-1.3125,-1.2358,1.6213]])
        Lcvx = np.array([[1.1531,0.48957,0.72742,0.89106,-1.0996,-1.2864],[0.48957,0.23665,0.3202,0.43173,-0.47376,-0.58422],[0.72742,0.3202,0.46336,0.58319,-0.69641,-0.82651],[0.89106,0.43173,0.58319,0.78764,-0.86253,-1.0647],[-1.0996,-0.47376,-0.69641,-0.86253,1.0503,1.2358],[-1.2864,-0.58422,-0.82651,-1.0647,1.2358,1.4854]])
        fcvx = 2.6968
        refsol = Scvx, Lcvx, fcvx
        refopts = {'off':1, 'use_u':1}
        hyperparams = .1, .1
        testname = "Catchain_d3"
        filename = "tests/data/Catchain_d3.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)

#    @unittest.skip('')
    def test_g1(self):
        """g continuous variables"""
        Scvx = np.array([[-1.3526,-0.0066189,1.7174e-12,9.5791e-14,-9.142e-14],[-0.0066189,-1.2408,-0.34376,-0.052764,-0.0067642],[1.7174e-12,-0.34376,-1.2603,-0.52609,-0.0541],[9.5791e-14,-0.052764,-0.52609,-1.1567,-0.55651],[-9.142e-14,-0.0067642,-0.0541,-0.55651,-0.98124]])
        Lcvx = np.array([[0.19457,-0.19506,0.17953,0.059501,-0.043471],[-0.19506,0.19556,-0.17999,-0.059653,0.043583],[0.17953,-0.17999,0.16566,0.054904,-0.040112],[0.059501,-0.059653,0.054904,0.018196,-0.013294],[-0.043471,0.043583,-0.040112,-0.013294,0.0097127]])
        fcvx = 2.3089
        refsol = Scvx, Lcvx, fcvx
        refopts = {'off':1,  'use_alpha':0}
        hyperparams = .1, 0.2
        testname = "GChain_s30d5_1L"
        filename = "tests/data/py_G_s30n5.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)   
    
#    @unittest.skip('')
    def test_g2(self):
        """12 continuous variables"""
        Scvx = np.array([[-3.7115,-2.5806e-10,-2.0324e-10,-3.8522e-11,5.1024e-11,-1.1406e-11,4.8437e-11,-1.4763e-10,-9.0311e-11,-2.3685e-11,-1.0376e-10,4.6053e-11],[-2.5806e-10,-4.6524,-2.1598e-10,-3.4537e-11,9.4106e-11,-1.271e-11,1.6364e-11,-2.2308e-10,3.386e-11,-1.9998e-11,-1.8232e-10,-6.7391e-11],[-2.0324e-10,-2.1598e-10,-4.6485,-1.1153e-10,1.3213e-11,-9.5399e-11,-7.686e-11,-2.8856e-10,2.9572e-11,-3.3279e-11,-2.3707e-10,-1.215e-10],[-3.8522e-11,-3.4537e-11,-1.1153e-10,-3.9274,-1.9501e-10,-2.613e-10,-2.6643e-10,-1.8369e-10,-9.3189e-11,-5.5339e-11,-8.982e-11,-5.4114e-11],[5.1024e-11,9.4106e-11,1.3213e-11,-1.9501e-10,-3.7127,-2.1063e-10,-2.0458e-10,-2.3064e-12,-2.0605e-10,-6.8329e-11,7.2147e-11,7.6432e-11],[-1.1406e-11,-1.271e-11,-9.5399e-11,-2.613e-10,-2.1063e-10,-4.1732,-2.3781e-10,-1.1774e-10,-1.474e-10,-9.6883e-11,-3.991e-11,-2.4058e-12],[4.8437e-11,1.6364e-11,-7.686e-11,-2.6643e-10,-2.0458e-10,-2.3781e-10,-3.6583,-1.503e-10,-1.9503e-11,-2.6123e-11,-9.2994e-11,-1.0074e-10],[-1.4763e-10,-2.2308e-10,-2.8856e-10,-1.8369e-10,-2.3064e-12,-1.1774e-10,-1.503e-10,-3.3107,3.8178e-11,-3.8612e-11,-2.4148e-10,-1.6124e-10],[-9.0311e-11,3.386e-11,2.9572e-11,-9.3189e-11,-2.0605e-10,-1.474e-10,-1.9503e-11,3.8178e-11,-4.0242,-4.2419e-11,1.3172e-10,2.345e-10],[-2.3685e-11,-1.9998e-11,-3.3279e-11,-5.5339e-11,-6.8329e-11,-9.6883e-11,-2.6123e-11,-3.8612e-11,-4.2419e-11,-3.0636,-1.0591e-10,-2.8751e-11],[-1.0376e-10,-1.8232e-10,-2.3707e-10,-8.982e-11,7.2147e-11,-3.991e-11,-9.2994e-11,-2.4148e-10,1.3172e-10,-1.0591e-10,-3.6648,-1.9186e-10],[4.6053e-11,-6.7391e-11,-1.215e-10,-5.4114e-11,7.6432e-11,-2.4058e-12,-1.0074e-10,-1.6124e-10,2.345e-10,-2.8751e-11,-1.9186e-10,-3.6827]])
        Lcvx = np.array([[0.5763,-0.26802,-0.10532,0.12371,0.086309,-0.27992,0.23969,-0.2695,-0.031901,0.019665,0.059873,0.22915],[-0.26802,1.2739,-0.86502,0.04751,0.08103,0.20984,0.13013,-0.012254,-0.17143,0.090891,-0.37338,0.11523],[-0.10532,-0.86502,1.3975,-0.55748,0.32747,0.051082,-0.22606,-0.18164,0.057971,-0.088818,0.066716,-0.011853],[0.12371,0.04751,-0.55748,1.2306,-0.41033,-0.37977,-0.2975,-0.039183,-0.044385,0.12113,-0.015667,-0.010973],[0.086309,0.08103,0.32747,-0.41033,1.1162,-0.77142,0.11388,-0.0098364,-0.081704,-0.0094082,0.0637,0.14165],[-0.27992,0.20984,0.051082,-0.37977,-0.77142,1.5443,-0.3174,-0.18043,-0.077988,-0.17963,-0.2453,0.063682],[0.23969,0.13013,-0.22606,-0.2975,0.11388,-0.3174,0.74352,-0.16057,-0.10933,0.050196,0.29978,-0.31863],[-0.2695,-0.012254,-0.18164,-0.039183,-0.0098364,-0.18043,-0.16057,0.78829,0.13964,0.14805,-0.27996,-0.041755],[-0.031901,-0.17143,0.057971,-0.044385,-0.081704,-0.077988,-0.10933,0.13964,0.41041,-0.028612,-0.058597,0.3474],[0.019665,0.090891,-0.088818,0.12113,-0.0094082,-0.17963,0.050196,0.14805,-0.028612,0.1612,-0.27141,0.056868],[0.059873,-0.37338,0.066716,-0.015667,0.0637,-0.2453,0.29978,-0.27996,-0.058597,-0.27141,1.0239,-0.67082],[0.22915,0.11523,-0.011853,-0.010973,0.14165,0.063682,-0.31863,-0.041755,0.3474,0.056868,-0.67082,1.032]])
        fcvx = -0.371591
        refsol = Scvx, Lcvx, fcvx
        refopts = {'off':1,  'use_alpha':0}
        hyperparams = 1, 0.2
        testname = "GChain_s12d12_1L"
        filename = "tests/data/py_G_s12d12l1.csv"
        self._joint(refsol, refopts, hyperparams, filename, testname)   
    
#    @unittest.skip('')
    def test_cg2(self):
        """mixed dataset with 3 cat and 3 CG variables"""
        Scvx = np.array([[-3.5971,1.2036e-12,3.6304e-13,-0.5854,-5.4106e-15,-1.3942e-10],[1.2036e-12,-2.9271,-2.5679e-13,-4.5677e-13,-0.84596,1.5184e-13],[3.6304e-13,-2.5679e-13,-5.2814,-1.5548e-12,-0.13043,-0.90272],[-0.5854,-4.5677e-13,-1.5548e-12,-1.2995,-0.39755,-0.31256],[-5.4106e-15,-0.84596,-0.13043,-0.39755,-1.1135,-0.3467],[-1.3942e-10,1.5184e-13,-0.90272,-0.31256,-0.3467,-1.0855]])
        Lcvx = np.array([[0.47022,0.20023,0.35059,-0.36113,-0.0020348,-0.19258],[0.20023,0.085257,0.14928,-0.15377,-0.00086643,-0.082002],[0.35059,0.14928,0.26139,-0.26925,-0.0015171,-0.14358],[-0.36113,-0.15377,-0.26925,0.27734,0.0015627,0.1479],[-0.0020348,-0.00086643,-0.0015171,0.0015627,8.8052e-06,0.00083335],[-0.19258,-0.082002,-0.14358,0.1479,0.00083335,0.078872]])
        fcvx = 2.7949
        refsol = Scvx, Lcvx, fcvx
        refopts = {'off':1, 'use_u':1, 'use_alpha':0}
        hyperparams = .1, .2
        testname = "Dbl1Chain_s12d3_0L"
        filename = "tests/data/py_CG_s12d3l0.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)   

#    @unittest.skip('')
    def test_cg3(self):
        """mixed dataset with 6 cat and 6 CG variables"""
        Scvx = np.array([[-6.8033,0.15632,3.3951e-13,-4.0276e-13,-4.9405e-15,2.0073e-13,-1.0887e-12,0.035074,2.0272e-13,5.3912e-13,-3.5003e-13,1.113e-14],[0.15632,-3.8193,5.9441e-13,1.4522e-13,-3.2012e-13,8.0491e-14,1.4028e-13,-0.19938,5.1387e-13,-2.2052e-13,4.9811e-14,0.042576],[3.3951e-13,5.9441e-13,-5.1017,0.7239,4.6019e-13,2.2587e-13,0.014329,7.0737e-14,-4.9616e-13,0.22519,0.17927,1.748e-13],[-4.0276e-13,1.4522e-13,0.7239,-3.7529,1.1457,-1.2931e-12,7.4718e-14,0.44977,6.0074e-13,-3.6415e-13,4.4797e-14,2.2538e-14],[-4.9405e-15,-3.2012e-13,4.6019e-13,1.1457,-3.0973,1.984e-13,-0.081073,-0.011972,-5.3291e-15,-5.8051e-14,-1.138,-5.9819e-13],[2.0073e-13,8.0491e-14,2.2587e-13,-1.2931e-12,1.984e-13,-3.4529,1.8573e-12,0.5622,0.2389,1.5765e-14,-0.030681,-5.107e-13],[-1.0887e-12,1.4028e-13,0.014329,7.4718e-14,-0.081073,1.8573e-12,-3.5648,-1.8816e-12,-0.050723,-0.2577,-3.0975e-13,-0.49405],[0.035074,-0.19938,7.0737e-14,0.44977,-0.011972,0.5622,-1.8816e-12,-2.3818,-0.90699,-0.3568,3.1808e-13,-9.2102e-14],[2.0272e-13,5.1387e-13,-4.9616e-13,6.0074e-13,-5.3291e-15,0.2389,-0.050723,-0.90699,-2.8076,-0.23169,4.947e-13,-1.9428e-12],[5.3912e-13,-2.2052e-13,0.22519,-3.6415e-13,-5.8051e-14,1.5765e-14,-0.2577,-0.3568,-0.23169,-3.2426,4.5852e-14,-1.5039e-12],[-3.5003e-13,4.9811e-14,0.17927,4.4797e-14,-1.138,-0.030681,-3.0975e-13,3.1808e-13,4.947e-13,4.5852e-14,-2.0912,-0.59865],[1.113e-14,0.042576,1.748e-13,2.2538e-14,-5.9819e-13,-5.107e-13,-0.49405,-9.2102e-14,-1.9428e-12,-1.5039e-12,-0.59865,-3.444]])
        Lcvx = np.array([[1.0666,0.55819,0.62756,-0.23423,0.097006,0.52042,-0.2589,0.14424,-0.031059,0.61736,-0.22707,0.16571],[0.55819,1.2518,1.0665,0.54348,-0.086298,0.17982,0.40827,-0.05312,0.084308,-0.18774,0.0017826,0.65164],[0.62756,1.0665,1.4719,0.44792,-0.036709,0.34941,0.653,0.012484,-0.27278,0.25655,0.064469,0.36245],[-0.23423,0.54348,0.44792,0.75951,-0.11058,-0.22997,0.33783,0.072628,0.2258,-0.60824,0.35697,0.29138],[0.097006,-0.086298,-0.036709,-0.11058,0.1497,0.27833,0.018075,-0.032437,0.1379,0.1033,-0.10462,-0.25625],[0.52042,0.17982,0.34941,-0.22997,0.27833,0.69393,0.14988,-0.087247,0.13291,0.40428,-0.28975,-0.33144],[-0.2589,0.40827,0.653,0.33783,0.018075,0.14988,0.85251,-0.27294,-0.15007,-0.15973,-0.0012633,0.0084292],[0.14424,-0.05312,0.012484,0.072628,-0.032437,-0.087247,-0.27294,0.22922,0.033579,0.055706,0.16759,0.0097768],[-0.031059,0.084308,-0.27278,0.2258,0.1379,0.13291,-0.15007,0.033579,0.58661,-0.41743,0.030971,-0.11877],[0.61736,-0.18774,0.25655,-0.60824,0.1033,0.40428,-0.15973,0.055706,-0.41743,0.89456,-0.26125,-0.22317],[-0.22707,0.0017826,0.064469,0.35697,-0.10462,-0.28975,-0.0012633,0.16759,0.030971,-0.26125,0.31326,0.078023],[0.16571,0.65164,0.36245,0.29138,-0.25625,-0.33144,0.0084292,0.0097768,-0.11877,-0.22317,0.078023,0.70581]])
        fcvx = 3.5787
        refsol = Scvx, Lcvx, fcvx
        refopts = {'off':1, 'use_u':1, 'use_alpha':0}
        hyperparams = .1, .2
        testname = "DblChain_s12d6_1L"
        filename = "tests/data/py_CG_s12d6l1.csv"

        self._joint(refsol, refopts, hyperparams, filename, testname)  


    @unittest.skip("takes long, afterwards solutions possibly inaccurate")
    def test_cg_iris_nonstd(self):
        """Iris mixed dataset non-standardized, 4 CG, 1 cat variable"""
        Scvx = np.array([[-16.1577,0,8.9976e-12,-2.6581e-11,3.4837e-11,-2.293e-11],[0,-43.9749,-1.4577e-11,1.5432e-12,2.5736e-11,4.3881e-11],[8.9976e-12,-1.4577e-11,-12.5917,9.5113e-11,8.1424e-11,-6.3193e-11],[-2.6581e-11,1.5432e-12,9.5113e-11,-13.7409,-6.4073e-11,6.311e-11],[3.4837e-11,2.5736e-11,8.1424e-11,-6.4073e-11,-17.5328,6.0962e-11],[-2.293e-11,4.3881e-11,-6.3193e-11,6.311e-11,6.0962e-11,-25.2529]])
        Lcvx = np.array([[2.0241,1.5539,0.46729,-2.1833,3.9121,1.044],[1.5539,4.588,-0.68339,-1.7168,6.2163,5.3576],[0.46729,-0.68339,8.0645,5.3003,2.8592,-0.46043],[-2.1833,-1.7168,5.3003,6.7483,-2.0269,-0.65204],[3.9121,6.2163,2.8592,-2.0269,11.7355,6.5982],[1.044,5.3576,-0.46043,-0.65204,6.5982,6.7164]])
        fcvx = -1.85807
        refsol = Scvx, Lcvx, fcvx
        refopts = {'off':1, 'use_u':1, 'use_alpha':0}
        hyperparams = 1, .2
        testname = "Iris_original"
        filename = "tests/data/iris_nonstandardized_py.csv"
    
        self._joint(refsol, refopts, hyperparams, filename, testname)  
        
#    @unittest.skip('')
    def test_cg_iris_std(self):
        """Iris mixed dataset standardized, 4 CG, 1 cat variable"""
        Scvx = np.array([[1.7844,0,-4.5795e-12,-3.4451e-12,1.524e-11,-1.4277e-11],[0,-2.4463,-2.9251e-12,4.8737e-12,1.5004e-12,1.9114e-11],[-4.5795e-12,-2.9251e-12,-10.2241,6.7609e-11,6.6978e-11,-6.0842e-11],[-3.4451e-12,4.8737e-12,6.7609e-11,-5.0205,-6.3651e-11,5.6963e-11],[1.524e-11,1.5004e-12,6.6978e-11,-6.3651e-11,-29.2807,6.2954e-11],[-1.4277e-11,1.9114e-11,-6.0842e-11,5.6963e-11,6.2954e-11,-19.0419]])
        Lcvx = np.array([[0.35912,-0.0094744,0.52016,-0.79346,1.4929,0.14674],[-0.0094744,1.0013,0.10838,0.13763,2.7186,2.8959],[0.52016,0.10838,5.2482,1.4515,4.3536,0.46048],[-0.79346,0.13763,1.4515,3.2601,-1.9061,-0.047246],[1.4929,2.7186,4.3536,-1.9061,14.5723,8.5554],[0.14674,2.8959,0.46048,-0.047246,8.5554,8.4627]])
        fcvx = -1.12586
        refsol = Scvx, Lcvx, fcvx
        refopts = {'off':1, 'use_u':1, 'use_alpha':0}
        hyperparams = 1, .2
        testname = "Iris_standardized"
        filename = "tests/data/iris_standardized_py.csv"

        # solving this takes a little longer, but likely below 10s
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
#        print(self.solver)
        self.solver.solve(report = 0)

        model = self.solver.get_canonicalparams()

        Scvx, Lcvx, fcvx = refsol
        f_cvx = self.solver.get_objective(Scvx, Lcvx)
        self.assertAlmostEqual(f_cvx,
                               fcvx,
                               places = 3,
                               msg="%f not equal %f"%(f_cvx, fcvx))
        
        params_sl = model.get_params_sl(padded = False)
        f_admm = self.solver.get_objective(*params_sl)

        fdiff = f_admm -  fcvx

        self.assertTrue(abs(fdiff) < self.ftol,
                        msg="%s:%f nequal %f"%(testname, fcvx, f_admm))

#        e = self.sparse_norm(mat_s) 
#        _, e2 = self.shrink(mat_s, 0)
#        print(e, e2)


if __name__ == '__main__':
    unittest.main()