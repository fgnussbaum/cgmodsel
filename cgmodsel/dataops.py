 # -*- coding: utf-8 -*-
"""
Copyright: Frank Nussbaum (frank.nussbaum@uni-jena.de)
(2019)

IO operations for loading data
"""

import time
import numpy as np
import pandas as pd

import csv

###############################################################################
# loading data and retrieving meta info
###############################################################################

def load_data_from_csv(filename, drop=[], verb=False):
    """read data from csv file <filename> into a panda dataframe and return the data frame"""
    t = time.time()
    
    # following f returns true if colname shall not be dropped
    f = lambda colname: not colname in drop 
    data = pd.read_csv(filename, index_col=None, usecols=f, skipinitialspace=0)
    
#    print(data.columns)
#    for col in data:
#        print(col)
    
    if verb:
        print('Data loading time:', time.time()-t)
    return data


def get_meta_data(data, verb=False, catuniques=None,
                  codedcnames=False, detectindexcols=True):
    """ extracts and returns dictionary of meta data for <data>
    particularly, detect categorical/numerical columns
    
    data ...        panda dataframe
    catuniques ...  either dictionary with levels of discrete variable
                    (this might be useful if certain discrete labels are
                    unobserved)
                    or list of levels, if the same for all discrete variables
                    
    codedcnames...  if True, recognize categorical variables by prefix 'X' in 
                    column name
    detectindexcols ... if True, look for columns with elements in {0,..., k}
                        and interpret such a columns as a discrete variable
    """
    t = time.time()

    # TODO: rather detect column types at loading time? dtype option in pd.read_csv
    categoricals = []
    numericals = []
    levels = {}
    sizes = []
    dc = 0
    dg = 0
    
    if catuniques is None:
        catuniquesprovided = False
        catuniques = {} # build catuniques automatically
    else:
        catuniquesprovided = True
        if type(catuniques) is type([]):
            tmp = catuniques
            catuniques = {}
            for col in data.columns:
                catuniques[col] = tmp

        
#    print(catuniques)
    for colname in data:
        column = data[colname]
#        print(column.dtype.name)
        
        categorical1 = column.dtype.name == "category" or \
          column.dtype.name == "bool" or column.dtype == "object" or \
           colname in categoricals or (colname[0]=='X' and codedcnames)
   
        detected = False    
        if not categorical1 and detectindexcols and column.dtype.name == "int64": 
            uniques = sorted(column.unique())
            if len(uniques) == 1 or (len(uniques) == 2 and uniques[1] == uniques[0]+1):
                detected = True
                
        if detected or categorical1:
            
            uniques = sorted(column.unique())

            if catuniquesprovided:
                assert colname in catuniques.keys(), "Incompatible catuniques provided, new cat column %s" %(colname)
                for label in uniques:
                    assert label in catuniques[colname], "Imcompatible catuniques provided, new label %s"%(label)
            else:
                catuniques[colname] = uniques
                
            categoricals.append(colname)
            sizes.append(len(catuniques[colname]))
            dc += 1
        else:
            numericals.append(colname)
            dg += 1
    assert dc == len(sizes), "Incompatible catuniques provided: detected %d cat cols but %d were given"%(dc, len(sizes))

#    categoricals = sorted(categoricals) # better leave the order as in the data set - easier indexing
#    print(categoricals,  numericals)
    
    ##### ** dictionary for translating cat data to indices **
    dcatval2ind={} # dictionary probably best for retrieving index from catvalue
    for colname in categoricals:
        dcatval2ind[colname] = {}
        for j, v in enumerate(catuniques[colname]):
            dcatval2ind[colname][v] = j
            # e.g. dval2ind['discretevarname']['discreteval0'] = 0

    ##### cumulative levels of categorical data **
    Lcum = np.cumsum([0] + sizes)

    ##### ** store everything in a dictionary **
    meta = {}
    meta['dc'] = dc
    meta['dg'] = dg
    meta['n'] = data.shape[0]
    
    meta['catnames'] = categoricals
    meta['contnames'] = numericals
    
    meta['dcatval2ind'] = dcatval2ind
    meta['Lcum'] = Lcum
    
    meta['sizes'] = sizes
#    meta['sizes_red'] = [size - 1 for size in sizes]
    
    meta['catuniques'] = catuniques
    
    if verb:
        print('Data Meta Processing time:', time.time()-t)
    return meta

###############################################################################
# wrapper
###############################################################################

def load_prepare_data(datasource, drop=[], verb=False, standardize=False,
                      cattype='dummy',  shuffle=False,
                      shuffleseed=10, **kwargs):
    """
    datasource ... either filename of csv-file (load data!) or panda data frame
    drop       ... (optional) if loading data, specifies columns not to load
    """
    if type(datasource) is type(""): # filename
        data = load_data_from_csv(datasource, drop=drop)
    else:
        data = datasource
    meta = get_meta_data(data, **kwargs)

    if verb and type(datasource) is type(""):
        print('Filename:', datasource)
        print('Loaded a dataset with %d samples, %d discrete and %d continuous variables.' % (meta['n'], meta['dc'], meta['dg']))
        print('Discrete Variables (at most 20): %s'%(meta['catnames'][:20]))
        print('Continuous Variables (at most 20): %s\n'%(meta['contnames'][:20]))
    
    if shuffle:
        data = data.sample(frac=1, random_state=shuffleseed)

    if meta['dg'] > 0:
        Y = data[meta['contnames']].values
    else:
        Y = np.empty((meta['n'],0))

    # transform discrete variables to indicator data/ flat index etc.
    if meta['dc'] > 0:
        D = prepare_cat_data(data[meta['catnames']], meta, cattype=cattype)
    else:
        D = np.empty((meta['n'],0))

    if standardize:
        # recommended to avoid exp overflow
        means, sigmas = standardize_continuous_data(Y)

    return D, Y, meta # TODO: return means and sigmas?

def load_prepare_split_data(filename, splittingfactor=1, 
                            cattype='dummy',
                            standardize=False, write_csv=False,
                            splittingseed=10, 
                            verb=True, **kwargs):
    """
    load data from specified filename
    
    drop ... list of columns to be dropped

    splittingfactor ... ratio to split between training and test data
    
    standardize ... if True standardize continuous data: 
                    useful to avoid exp-overflow in certain data sets
    
    returns: tuple (Dtrain, Ytrain, Dtest, Ytest, meta) where
    D denotes discrete data (dummy encoded) and Y continuous data
    with corresponding suffixes for training and test data,
    and meta is a dictionary containing information about the data.
    """
    ##### ** load and prepare cts and cat data ** 
    if splittingfactor < 1:
        shuffle = True
    else:
        shuffle = False
    D, Y, meta = load_prepare_data(filename, verb=verb, 
                                    standardize=False, cattype=cattype,
                                    shuffle=shuffle, **kwargs)

    ## split data for training and validation
    n = meta['n']
    ntrain = int(splittingfactor * n);

    Dtrain = D[:ntrain, :]; Dtest = D[ntrain:, :]
    Ytrain = Y[:ntrain, :]; Ytest = Y[ntrain:, :]
    
    if standardize:
        
        t1 = time.time()
        means, sigmas = standardize_continuous_data(Ytrain)
        # standardize test data using means/sigmas from training data
        standardize_continuous_data(Ytest, (means, sigmas))
        meta['means'] = means
        meta['sigmas'] = sigmas
        print("Standardized training data...(%fms)"%(time.time()-t1))
        
    if write_csv:
        # write training data to file
        filename_train = filename[:-4]+'_train.csv'
        filename_test = filename[:-4]+'_test.csv'
        print("Writing training data to file(s)... %s"%(filename_train))
        if cattype == 'dummy':
            from cgmodsel.utils import dummy_to_index            
            Xtrain = dummy_to_index(Dtrain, meta['sizes'])
            Xtest= dummy_to_index(Dtest, meta['sizes'])
        elif cattype == 'index':
            Xtrain = Dtrain
            Xtest = Dtest
        
        writeCSV(filename_train, Xtrain, Ytrain, meta)
        writeCSV(filename_test, Xtest, Ytest, meta)

    return Dtrain, Ytrain, Dtest, Ytest, meta


def split_traintest(filename, splittingfactor, **kwargs):
    """wrapper for  load_prepare_split_data, writes split data to csv"""
    Dtrain, Ytrain, Dtest, Ytest, meta = \
      load_prepare_split_data(filename, splittingfactor=splittingfactor, 
                            write_csv=True, cattype='index', **kwargs)
    
    meta['ntrain']  = int(splittingfactor * meta['n']);
    return meta

def load_traintest_datasets(filename_trunk, verb=True, **kwargs):
    """load train and test data from csv files filename_trunk%s.csv, where %s is _train or _test """
    filename_train = filename_trunk + '_train.csv'
    filename_test = filename_trunk + '_test.csv'
    name = filename_trunk.split('/')[-1]

    Dtrain, Ytrain, meta = load_prepare_data(filename_train, **kwargs)
    Dtest, Ytest, meta_test = load_prepare_data(filename_test, **kwargs)
    meta['ntrain'] = meta['n']
    meta['n'] += meta_test['n']
    if verb:
        print('Name:', name)
        print('Loaded a datasets, %d discrete and %d continuous variables.' % (meta['dc'], meta['dg']))
        print('Training data has %d samples, test data has %d samples'%(meta['n'], meta_test['n']))
        print('Discrete Variables (at most 20): %s'%(meta['catnames'][:20]))
        print('Continuous Variables (at most 20): %s\n'%(meta['contnames'][:20]))
    
    return Dtrain, Ytrain, Dtest, Ytest, meta

###############################################################################
# data preprocessing
###############################################################################

def dummy_from_X(X, meta, red=False):
    """
    convert discrete data X into dummy coded version
    meta ... dictionary of meta data for data X, must contain attribute Lcum
            (cumulative levels/delimiters for concatenated indicator variables)
    red ... if True, leave out indicator variable for first level
    """
    n, dc = X.shape
    catcumlevels = meta['Lcum']
    nr = catcumlevels[-1]

    if red:
        D = np.zeros((n, nr - dc), dtype=np.int64) 
        for i in range(n):
            for j in range(dc):
                if X[i,j] > 0:
                    D[i, catcumlevels[j] -j + X[i,j] - 1] = 1

    else:
        D = np.zeros((n, nr), dtype = np.int64) 
    
        for i in range(n):
            for j in range(dc):
                D[i, catcumlevels[j] + X[i,j] ] = 1
    return D

def prepare_cat_data(data, meta, verb=False, cattype='dummy'):
    """
    preprocess discrete/categorical data
    input:
    data  ... panda data frame
    meta ...  meta information for panda data frame (such as number of dicrete/continuous cols)
    cattype ... 
      'dummy' ... store discrete data as dummy encoded variables
      'dummy_red' ... store discrete data as dummy (leave out first col)
      'index' ... store discrete data as index vectors
      'index+1' ... store discrete data as index vectors, index shifted by one
      'flat'  ... store flattened (linear) index of discrete data (required for MAP-estimators)

    output:
    D  ... matrix of discrete data (dummy encoded indicator vectors, or indices)"""
    t = time.time()
    n, d = data.shape
    
    dc = meta['dc']
    assert d  == dc
    
    catcols = meta['catnames']
    dcatval2ind = meta['dcatval2ind']
    catcumlevels = meta['Lcum']
    nr = catcumlevels[-1]

    if dc == 0:
        return np.array([])

    ##### ** generate indicator vectors for categorical data **
    if cattype == 'dummy':
        # store dummy encoded discrete variables (i.e. indicator vectors)
        D = np.zeros((n, nr), dtype=np.int64) # Store indicator vectors for each data point herein, TODO: sparsity?

        for i in range(n): # TODO: this data transformation is very costly
            for j in range(dc):
    #            print('%d-th row, value %d'%(i, j), data.iloc[i, j])
    #            print('colname', catcols[j])
    #            print('catname', data.iloc[indices[i], j])
    
                # look up position of j-th variable value
                posj = dcatval2ind[catcols[j]][data.iloc[i, j]] 
                D[i, catcumlevels[j] + posj ] = 1
    elif cattype == 'dummy_red':
        # store dummy encoded discrete variables (apart from 0-th level)
        D = np.zeros((n, nr - dc), dtype=np.int64) 
        for i in range(n): # TODO: code redundancy
            for j in range(dc):
                posj = dcatval2ind[catcols[j]][data.iloc[i, j]] # position of j. variable value
                if posj != 0:
                    D[i, catcumlevels[j]-j + posj-1] = 1
    elif cattype == 'index' or cattype == 'index+1':
        # just store tuples of indices for each categorical variable
        D = np.empty((n, dc), dtype=np.int64)
        if cattype == 'index':
            shift = 0
        else:
            shift = 1
        for i in range(n):
            for j in range(dc):
                D[i, j] = dcatval2ind[catcols[j]][data.iloc[i, j]] + shift
    elif cattype == 'flat': # just store tuples of indices for each categorical variable
        D = np.empty( n, dtype=np.int64)
        ind = np.zeros(dc, dtype = np.int64)
        L = meta['sizes']
        for i in range(n):
            for j in range(dc):
                ind[j] = dcatval2ind[catcols[j]][data.iloc[i, j]]
            D[i] = np.ravel_multi_index(ind, L)

    if verb:
        print('Data preprocessing time:', time.time() - t)
    return D

def standardize_continuous_data(Y, meanssigmas=None):
    """standardize the continous data in Y and return means and standard deviations """
    n, dg = Y.shape
    if meanssigmas is None:
        means = Y.sum(axis = 0) /n
        sigmas = np.sqrt((Y**2).sum(axis=0)/n - means**2)
    else:
        means, sigmas = meanssigmas
    for s in range(dg):
        Y[:, s] = (Y[:, s] - means[s] * np.ones(n)) / sigmas[s]

    return means, sigmas
    
###############################################################################
# write data to file
###############################################################################

def writeCSV(filename, X, Y, meta, method='rpl_numericalcats', prefix='val'):
    """
    formatlab:  if true leave out column names in first row of csv file
                and store only numerical values for categorical variables
    
    filename:   name for csv where data is stored
    
    method:     is in {rpl_numericalcats} where
    
      rpl_numericalcats replaces any integer value <d> of a categorical 
          variable by dummy string val<d>
      for_matlab produces a file that can be read from MATLAB
    
    meta:       dictionary containing meta information as usual
    
    required options in meta are
      dc          # of discrete variables
      dg          # of Gaussian variables
    
    optional options in meta are
      dl:         if >0, then the last dl Gaussian columns are dropped, 
                  in particular they are eliminated permanently from
                  the Gaussian data Y
      gaussnames: column names for Gaussian variables
      catnames:   column names for Gaussian variables
    """
    ## column names
    dc = meta['dc']
    if 'catnames' in meta:
        catcols = meta['catnames'][:]
        assert(len(catcols) == dc)
    else:
        catcols = ['X%d'%(i) for i in range(dc)]

    dg = meta['dg'] 
    if 'gaussnames' in meta:
        gausscols = meta['gaussnames'][:]
        assert(len(gausscols) == dg)
    else:
        gausscols = ['g%d'%(i) for i in range(dg)]
    
    ## drop columns? (aka unobserved variables)
    if 'dl' in meta: 
        dl = meta['dl']
        if dl>0:
            assert dl <= dg
            gausscols = gausscols[:-dl]
            colindices = list(range(dc+dg-dl, dc+dg))
            Y = np.delete(Y, colindices, axis=1) # note: this modifies Y
    
    if dc > 0:
        n, _ = X.shape
    if dg > 0:
        n, _ = Y.shape
    if dc >0 and dg > 0:
        assert (X.shape[0] == Y.shape[0])
    ## write data         
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(catcols + gausscols)
        
        if method == 'formatlab': # do not use strings
            # in MATLAB use csvread(filename, 1)
            for i in range(n):
                l = list(X[i, :]+1) + list(Y[i, :])
                writer.writerow(l)
        elif method == 'rpl_numericalcats':
            # replace numerical values in categorical columns
            for i in range(n):
                l = [prefix +str(e) for e in X[i, :]] + list(Y[i, :])
                writer.writerow(l)
        else:
            raise Exception('No writing method given')
    
    
