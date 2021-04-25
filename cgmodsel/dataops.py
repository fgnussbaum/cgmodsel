# -*- coding: utf-8 -*-
"""
Copyright: Frank Nussbaum (frank.nussbaum@uni-jena.de)
(2019)

IO operations for loading data
"""

import time
import csv

import numpy as np
import pandas as pd

###############################################################################
# loading data and retrieving meta info
###############################################################################


def load_data_from_csv(filename: str, drop=(), names=None, verb: bool = False):
    """read data from csv file into a panda dataframe
    
    Args:
        filename (str): filename.
        drop (tuple): column names to exclude from loading.
        names (iterable, optional): names for the loaded columns.
        verb (bool): whether to print stats.
        
    Returns:
        pd.dataframe: data.
    """
    tic = time.time()

    # following f returns true if colname shall not be dropped
    func = lambda colname: not colname in drop
    data = pd.read_csv(filename,
                       index_col=None,
                       usecols=func,
                       names=names,
                       skipinitialspace=0)

    #    print(data.columns)
    #    for col in data:
    #        print(col)

    if verb:
        print('Data loading time:', time.time() - tic)
    return data


def get_meta_data(data,
                  verb: bool = False,
                  catuniques=None,
                  categoricals=None,
                  coded_colnames: bool = False,
                  detectindexcols: bool = True,
                  **kwargs):
    """Extracts and returns dictionary of meta data for provided data.
    
    Args:
        data (pd.dataframe): data
        catuniques: either dictionary with levels of discrete variable
                    (this might be useful if certain discrete labels are
                    unobserved)
                    or list of levels, if the same for all discrete variables.
        categoricals: optional list of column names for discrete variables.
        coded_colnames (bool): if True, recognize categorical variables by 
            prefix 'X' in column name.
        detectindexcols (bool): if True, look for columns with elements 
            in {0,..., k} and interpret such a columns as a discrete variable.
    
    Returns:
        dict: dictionary with meta information.
    """
    tic = time.time()

    # TODO(franknu): rather detect column types at loading time?
    # TODO(franknu): use heuristic to diff between cat and numerical cols
    # e.g., #uniques < 10% of columns
    # also, there exists a dtype option in pd.read_csv
    if categoricals is None:
        categoricals = []
        infer_categoricals = True
    else:
        for categorical in categoricals:
            if not categorical in data:
                print('No column %s loaded, removing it from provided list of categoricals'%categorical)
                categoricals.remove(categorical)
        infer_categoricals = False
    numericals = []
    sizes = []
    n_cat = 0
    n_cg = 0

    if catuniques is None:
        catuniques_provided = False
        catuniques = {}  # build catuniques automatically
    else:
        catuniques_provided = True
        if isinstance(catuniques, list):
            tmp = catuniques
            catuniques = {}
            for col in data.columns:
                catuniques[col] = tmp

    for colname in data:
        column = data[colname]

        if infer_categoricals:
            categorical1 = (column.dtype.name == "category" or 
                            column.dtype.name == "bool" or 
                            column.dtype == "object" or 
                            colname in categoricals or
                            (colname[0] == 'X' and coded_colnames))

            categorical2 = False
            if (not categorical1 and detectindexcols
                and column.dtype.name == "int64"):
                uniques = sorted(column.unique())
                if len(uniques) == 1 or \
                    (len(uniques) == 2 and uniques[1] == uniques[0] + 1):
                    categorical2 = True
                    
            is_categorical = categorical1 or categorical2
            if is_categorical:
                categoricals.append(colname)
        else:
            is_categorical = colname in categoricals

        if is_categorical:
            uniques = sorted(column.unique())

            if catuniques_provided:
                assert colname in catuniques.keys(), \
                    "Incompatible catuniques, new cat column %s" %(colname)
                for label in uniques:
                    assert label in catuniques[colname], \
                        "Imcompatible catuniques, new label %s"%(label)
            else:
                catuniques[colname] = uniques
            sizes.append(len(catuniques[colname]))
            n_cat += 1
        else:
            numericals.append(colname)
            n_cg += 1
    assert n_cat == len(
        sizes
    ), "Incompatible catuniques: detected %d cat cols but %d were given" % (
        n_cat, len(sizes))

    #    print(categoricals,  numericals)

    cat_in_indexform = True
    for colname in categoricals:
        nlevels = len(catuniques[colname])
        for i in range(nlevels):
            if not i in catuniques[colname]:
                cat_in_indexform = False
    ##### ** dictionary for translating cat data to indices **
    catval2ind = {
    }  # dictionary probably best for retrieving index from catvalue
    for colname in categoricals:
        catval2ind[colname] = {}
        for j, val in enumerate(catuniques[colname]):
            catval2ind[colname][val] = j
            # e.g. dval2ind['discretevarname']['discreteval0'] = 0

    ##### ** store everything in a dictionary **
    meta = {}
    meta['n_cat'] = n_cat
    meta['n_cg'] = n_cg
    meta['n_data'] = data.shape[0]

    meta['categoricals'] = categoricals
    meta['numerical'] = numericals

    meta['catval2ind'] = catval2ind
    meta['cat_in_indexform'] = cat_in_indexform
    meta['cat_glims'] = np.cumsum([0] + sizes)

    meta['sizes'] = sizes

    meta['catuniques'] = catuniques

    if verb:
        print('Meta data processing time: %.1f'%(time.time() - tic))
    return meta


###############################################################################
# wrapper
###############################################################################


def load_prepare_data(datasource,
                      drop=(),
                      verb: bool = False,
                      standardize: bool = False,
                      cattype: str = 'dummy',
                      names=None,
                      shuffle: bool = False,
                      shuffleseed: int = 10,
                      **kwargs):
    """
    
    Args:
        datasource: either filename of csv-file (load data!) or panda data frame.
        drop: (optional) if loading data, specifies columns not to load.
        standardize (bool): whether to standardize continuous data.
        cattype (str): encoding of the discrete variables (dummy).
        names: optional list of column names.
        shuffle (bool): whether to shuffle rows before loading.
        shuffleseed (int): seed for shuffling.
        
    Returns:
        tuple: cat_data (np.array), cont_data (np.array),
        meta information about the data (dict).
    """
    t1 = time.time()
    if isinstance(datasource, str):  # filename
        data = load_data_from_csv(datasource, names=names, drop=drop)
    else:
        data = datasource
    t2 = time.time()
    if verb:
        print("Data loading time: %.1f"%(t2-t1))
    meta = get_meta_data(data, verb=verb, **kwargs)

    if verb and isinstance(datasource, str):
        print('Filename:', datasource)
        print(
            'Loaded a dataset with %d samples, %d discrete and %d continuous variables.'
            % (meta['n_data'], meta['n_cat'], meta['n_cg']))
        print('Discrete Variables (at most 20): %s' % (meta['categoricals'][:20]))
        print('Continuous Variables (at most 20): %s\n' %
              (meta['numerical'][:20]))

    if shuffle:
        if verb:
            print('Randomly shuffling data...')
        data = data.sample(frac=1, random_state=shuffleseed)

    if meta['n_cg'] > 0:
        cont_data = data[meta['numerical']].values
    else:
        cont_data = np.empty((meta['n_data'], 0))

    # transform discrete variables to indicator data/ flat index etc.
    if meta['n_cat'] > 0:
        cat_data = prepare_cat_data(data[meta['categoricals']],
                                    meta,
                                    cattype=cattype)
    else:
        cat_data = np.empty((meta['n_data'], 0))

    if standardize:
        # recommended to avoid exp overflow
        # means, sigmas =
        if verb:
            print('Standardizing continuous data...')
        standardize_continuous_data(cont_data)

    return cat_data, cont_data, meta
    # TODO(franknu): return means and sigmas?


def load_prepare_split_data(filename: str,
                            splittingfactor=1,
                            cattype: str = 'dummy',
                            standardize: bool = False,
                            write_csv: bool = False,
                            verb: bool = True,
                            **kwargs):
    """
    Load data from specified filename.

    Args:
        drop: optional list of columns to be dropped.
        splittingfactor (int): ratio to split between training and test data.
        standardize (bool): if True standardize continuous data:
                    useful to avoid exp-overflow in certain data sets.

    Returns: 
        tuple: (Dtrain, cont_datatrain, Dtest, cont_datatest, meta), where
        D denotes discrete data (dummy encoded) and cont_data continuous data
        with corresponding suffixes for training and test data,
        and meta is a dictionary containing information about the data.
    """
    ##### ** load and prepare cts and cat data **
    shuffle = (splittingfactor < 1)

    cat_data, cont_data, meta = load_prepare_data(filename,
                                                  verb=verb,
                                                  standardize=False,
                                                  cattype=cattype,
                                                  shuffle=shuffle,
                                                  **kwargs)

    ## split data for training and validation
    n_data = meta['n_data']
    ntrain = int(splittingfactor * n_data)

    cat_datatrain = cat_data[:ntrain, :]
    cat_datatest = cat_data[ntrain:, :]
    cont_datatrain = cont_data[:ntrain, :]
    cont_datatest = cont_data[ntrain:, :]

    if standardize:

        tic = time.time()
        means, sigmas = standardize_continuous_data(cont_datatrain)
        # standardize test data using means/sigmas from training data
        standardize_continuous_data(cont_datatest, (means, sigmas))
        meta['means'] = means
        meta['sigmas'] = sigmas
        print("Standardized training data...(%fms)" % (time.time() - tic))

    if write_csv:
        # write training data to file
        filename_train = filename[:-4] + '_train.csv'
        filename_test = filename[:-4] + '_test.csv'
        print("Writing training data to file(s)... %s" % (filename_train))
        if cattype == 'dummy':
            from cgmodsel.utils import dummy_to_index
            cat_data_train_w = dummy_to_index(cat_datatrain, meta['sizes'])
            cat_data_test_w = dummy_to_index(cat_datatest, meta['sizes'])
        elif cattype == 'index':
            cat_data_train_w = cat_datatrain
            cat_data_test_w = cat_datatest

        write_to_csv(filename_train, cat_data_train_w, cont_datatrain, meta)
        write_to_csv(filename_test, cat_data_test_w, cont_datatest, meta)

    return cat_datatrain, cont_datatrain, cat_datatest, cont_datatest, meta


def split_traintest(filename: str, splittingfactor, **kwargs):
    """wrapper for load_prepare_split_data, writes split data to csv.
    
    Args:
        filenames (str): filename.
        splittingfactor (int): train/test split ration in [0,1].
    
    Returns:
        dict: dictionary of meta information about the data.
    """
    _, _, _, _, meta = \
      load_prepare_split_data(filename, splittingfactor=splittingfactor,
                              write_csv=True, cattype='index', **kwargs)

    meta['ntrain'] = int(splittingfactor * meta['n_data'])
    return meta


def load_traintest_datasets(filename_trunk: str, verb: bool = True, **kwargs):
    """load train/ test data from csv files, compare load_prepare_split_data.
    
    Note:
        Filenames must be of form 
        filename_trunk%s.csv, where %s is _train or _test
        
    """
    filename_train = filename_trunk + '_train.csv'
    filename_test = filename_trunk + '_test.csv'
    name = filename_trunk.split('/')[-1]

    cat_datatrain, cont_datatrain, meta = load_prepare_data(
        filename_train, **kwargs)
    cat_datatest, cont_datatest, meta_test = load_prepare_data(
        filename_test, **kwargs)
    meta['ntrain'] = meta['n_data']
    meta['n_data'] += meta_test['n_data']
    if verb:
        print('Name:', name)
        print('Loaded a datasets, %d discrete and %d continuous variables.' %
              (meta['n_cat'], meta['n_cg']))
        print('Training data has %d samples, test data has %d samples' %
              (meta['n_data'], meta_test['n_data']))
        print('Discrete Variables (at most 20): %s' % (meta['categoricals'][:20]))
        print('Continuous Variables (at most 20): %s\n' %
              (meta['numerical'][:20]))

    return cat_datatrain, cont_datatrain, cat_datatest, cont_datatest, meta


###############################################################################
# data preprocessing
###############################################################################


def dummy_from_cat_data(cat_data, meta: dict, red: bool = False):
    """
    Convert discrete data cat_data into dummy-coded version.
    
    Args:
        meta (dict): dictionary of meta data for data cat_data,
            must contain attribute cat_glims
            (cumulative levels/delimiters for concatenated indicator variables)
        red (bool): if True, leave out indicator variable for first level
        
    Returns:
        np.array: categorical data
    """
    n_data, n_cat = cat_data.shape
    cat_glims = meta['cat_glims']
    ltot = cat_glims[-1]

    if red:
        cat_data = np.zeros((n_data, ltot - n_cat), dtype=np.int64)
        for i in range(n_data):
            for j in range(n_cat):
                if cat_data[i, j] > 0:
                    cat_data[i, cat_glims[j] - j + cat_data[i, j] - 1] = 1

    else:
        cat_data = np.zeros((n_data, ltot), dtype=np.int64)

        for i in range(n_data):
            for j in range(n_cat):
                cat_data[i, cat_glims[j] + cat_data[i, j]] = 1
    return cat_data


def prepare_cat_data(data,
                     meta: dict,
                     verb: bool = False,
                     cattype: str = 'dummy'):
    """
    Preprocess discrete/categorical data.
    
    Args:
        data (pd.dataframe): data.
        meta (dict):  meta information for panda data frame
        (such as number of dicrete/continuous cols).
        cattype (str): encoding of discrete data, can be
            'dummy' ... store discrete data as dummy encoded variables
            'dummy_red' ... store discrete data as dummy (leave out first col)
            'index' ... store discrete data as index vectors
            'index+1' ... store discrete data as index vectors, index shifted by one
            'flat'  ... store flattened (linear) index of discrete data
            (required for MAP-estimators).
        verb (bool): whether to prints stats.

    Returns:
        np.array: matrix of discrete data
        (dummy encoded indicator vectors, or indices)
    """
    tic = time.time()
    n_data, n_cat = data.shape

    assert meta['n_cat'] == n_cat

    if n_cat == 0:
        return np.array([])

    catcols = meta['categoricals']
    catval2ind = meta['catval2ind']
    cat_glims = meta['cat_glims']
    ltot = cat_glims[-1]

    ##### ** generate indicator vectors for categorical data **
    if cattype == 'dummy':
        # store dummy encoded discrete variables (i.e. indicator vectors)
        cat_data = np.zeros((n_data, ltot), dtype=np.int64)
        # store indicator vectors herein, TODO(franknu): sparsity?

        for i in range(n_data):
            # TODO(franknu): this data transformation is very costly
            for j in range(n_cat):
                #            print('%d-th row, value %d'%(i, j), data.iloc[i, j])
                #            print('colname', catcols[j])
                #            print('catname', data.iloc[indices[i], j])

                # look up position of j-th variable value
                posj = catval2ind[catcols[j]][data.iloc[i, j]]
                cat_data[i, cat_glims[j] + posj] = 1
    elif cattype == 'dummy_red':
        # store dummy encoded discrete variables (apart from 0-th level)
        cat_data = np.zeros((n_data, ltot - n_cat), dtype=np.int64)
        if meta['cat_in_indexform']:
            for i in range(n_data):
                row = data.iloc[i, :]
                for j in range(n_cat):
                    cat_data[i, cat_glims[j] - j + row[j] - 1] = 1
        else:
            for i in range(n_data):  # TODO: code redundancy
                for j in range(n_cat):
                    posj = catval2ind[catcols[j]][data.iloc[i, j]]
                    # posj: position of j. variable value
                    if posj != 0:
                        cat_data[i, cat_glims[j] - j + posj - 1] = 1
    elif cattype in ('index', 'index+1'):
        # just store tuples of indices for each categorical variable
        cat_data = np.empty((n_data, n_cat), dtype=np.int64)
        if cattype == 'index':
            shift = 0
        else:
            shift = 1
        for i in range(n_data):
            for j in range(n_cat):
                cat_data[i, j] = catval2ind[catcols[j]][data.iloc[i, j]] + shift
    elif cattype == 'flat':  # just store tuples of indices for each categorical variable
        cat_data = np.empty(n_data, dtype=np.int64)
        ind = np.zeros(n_cat, dtype=np.int64)
        sizes = meta['sizes']
        for i in range(n_data):
            for j in range(n_cat):
                ind[j] = catval2ind[catcols[j]][data.iloc[i, j]]
            cat_data[i] = np.ravel_multi_index(ind, sizes)

    if verb:
        print('Data preprocessing time:', time.time() - tic)
    return cat_data


def standardize_continuous_data(cont_data, meanssigmas=None):
    """Standardize continous data.
    
    Args: 
        cont_data (np.array): continuous data.
        meanssigmas (optional): if provided, standardize with these means
            and sigmas instead of empirical quantities from the dataset.
    
    Returns:
        tuple: means and standard deviations
    """
    n_data, n_cg = cont_data.shape
    if meanssigmas is None:
        means = cont_data.sum(axis=0) / n_data
        sigmas = np.sqrt((cont_data**2).sum(axis=0) / n_data - means**2)
    else:
        means, sigmas = meanssigmas
    for s in range(n_cg):
        if sigmas[s] == 0:
            print('Warning(standardize_continuous_data):',
                  'Variable %d seems to have zero variance, skipping'%s)
            continue
        cont_data[:, s] = (cont_data[:, s] -
                           means[s] * np.ones(n_data)) / sigmas[s]

    return means, sigmas


###############################################################################
# write data to file
###############################################################################


def write_to_csv(filename: str,
                 cat_data,
                 cont_data,
                 meta: dict,
                 method: str = 'rpl_numericalcats',
                 prefix: str = 'val'):
    """Write data to file.

    Args:    
        filename (str): name for csv where to store data.
        cat_data (np.array): categorical data.
        cont_data (np.array): continuous data.
        meta (dict): meta information about the data
            required options in meta are n_cat (# of discrete variables),
            n_cg (# of Gaussian variables), optional options in meta are
            n_latent (if >0, then the last dl Gaussian columns are dropped,
            they are eliminated permanently from
            the continuous data cont_data), 
            gaussnames (column names for Gaussian variables), 
            categoricals (column names for Gaussian variables),
        method (str): method for storing discrete data, can be
            rpl_numericalcats (replaces any integer value <d> of a categorical
            variable by dummy string val<d>) or for_matlab 
            (produces a file that can be read from MATLAB).
    """
    ## column names
    n_cat = meta['n_cat']
    if 'categoricals' in meta:
        catcols = meta['categoricals'][:]
        assert len(catcols) == n_cat
    else:
        catcols = ['X%d' % (i) for i in range(n_cat)]

    n_cg = meta['n_cg']
    if 'gaussnames' in meta:
        gausscols = meta['gaussnames'][:]
        assert len(gausscols) == n_cg
    else:
        gausscols = ['g%d' % (i) for i in range(n_cg)]

    ## drop columns? (aka unobserved variables)
    if 'n_latent' in meta:
        n_latent = meta['n_latent']
        if n_latent > 0:
            assert n_latent <= n_cg
            gausscols = gausscols[:-n_latent]
            colindices = list(range(n_cat + n_cg - n_latent, n_cat + n_cg))
            cont_data = np.delete(cont_data, colindices, axis=1)
            # note: this modifies cont_data

    if n_cat > 0:
        n_data, _ = cat_data.shape
    if n_cg > 0:
        n_data, _ = cont_data.shape
    if n_cat > 0 and n_cg > 0:
        assert cat_data.shape[0] == cont_data.shape[0]
    ## write data
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

        writer.writerow(catcols + gausscols)

        if method == 'formatlab':  # do not use strings
            # in MATLAB use csvread(filename, 1)
            for i in range(n_data):
                line = list(cat_data[i, :] + 1) + list(cont_data[i, :])
                writer.writerow(line)
        elif method == 'rpl_numericalcats':
            # replace numerical values in categorical columns
            for i in range(n_data):
                line = [prefix + str(val) for val in cat_data[i, :]]
                line += list(cont_data[i, :])
                writer.writerow(line)
        else:
            raise Exception('No writing method given')
