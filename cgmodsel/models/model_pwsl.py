# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
import pickle

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from cgmodsel.models.base import BaseModelPW
from cgmodsel.models.base import unpad, pad, _split_theta

# pylint: disable=R0914 # too many local variable
# pylint: disable=R0913 # too many arguments


##################################
class ModelPWSL(BaseModelPW):
    """
    A class for parameters of sparse + low-rank models
    
    Note:
        Model to store parameters of pairwise CG model,
        where some of the CG variables have been marginalized out.
        Density is given by
        
        p(x,y) ~ exp(1/2 (C_x, y)^T (S+L) (C_x y) + u^T C_x + alpha^T y)

        Here, C_x is a dummy representation of x,
        u are univariate discrete parameters,
        alpha are univariate cont. parameters,
        L>=0 describes indirect interactions,
        S describes direct pairwise interactions given by
        S= (Q & R^T \\ R & -Lambda)
        with
        discrete-discrete interactions Q,
        discrete-cont. interactions R,
        and cont.-cont. interactions Lambda.

        This class uses padded parameters (that is,
        parameters for 0-th levels are included, however,
        one might want them to be constrained to 0 for identifiability reasons)

    Warning:
        Attention to the special case of only Gaussians observed variables,
        uses as above
        
        p(y) ~ exp(1/2  y^T (S+L) y + alpha^T y),
        
        i.e., the pairwise parameters S have inverted sign
        compared to the regular precision matrix 
        as used in Chandrasekaran et al. (2011):
        p(y) ~ exp(-1/2 y^T (S-L) y + alpha^T y).
    """
    
    name = 'SL'  # model name

    def __init__(self,
                 sl_params: tuple = None,
                 meta: dict = None,
                 infile: str = None,
                 annotations: dict = {},
                 in_padded: bool = True):
        """
        Args:
            sl_params (tuple): parameters (u, Q, R, alpha, Lambda, L).
            meta (dict): dictionary of meta info (similar to meta dictionary
                 from loading data).
            infile (str): filename of a PW model file. Loading model from file
                 has the highest priority.
            annotations (dict, optional): pass annotations to the model.
            in_padded (bool): whether parameters are padded
        """

        if not infile is None:
            # load model from file
            params = pickle.load(open(infile, "rb"))
            assert params['type'] == self.name, \
            "Wrong model type (is %s, should be %s)"%(params['type'], self.name)
            sl_params = params['params']
            meta = params['meta']
            annotations.update(params['annotations'])

        assert (not sl_params is None) and (not meta is None), "Incomplete data"

        *pw_params, mat_l = sl_params
        BaseModelPW.__init__(self,
                             pw_params,
                             meta,
                             annotations=annotations,
                             in_padded=in_padded)
        if not in_padded:
            self.mat_l = pad(mat_l, self.meta['sizes'])
        else:
            self.mat_l = mat_l  # low-rank matrix

    def __str__(self):
        """a string representation for the model"""
        mat_s, mat_l, _, _ = self.get_params_sl(padded=False)

        string = '<pwsl_model S:' + str(mat_s) + \
            '\nL(from Theta=S+L):\n' + str(mat_l) + \
            '\nalpha:' + str(self.alpha.T) + '\nu:' + str(self.vec_u.T)+'>'
        if self.annotations.keys():
            string += "\nAnnotations: " + str(self.annotations)
        return string

    def get_stats(self, threshold=1e-2, **kwargs):
        """Calculate number of edges of S and rank of L.
        
        Args:
            threshold (float): cutoff for edges and principal components.
            
        Returns:
            tuple: number of edges (int), rank (int)
        """
        spec = np.linalg.eigvals(self.mat_l)
        rank = len(*np.where(spec > threshold))

        #        print('ModelPWSL.get_stats: maximal singular value:', np.max(spec))

        graph = self.get_graph(threshold=threshold, **kwargs)
        graph = graph.astype(int)
        no_edges = int(np.sum(graph) / 2)

        return no_edges, rank

    def get_groupnorm_theta(self, diagonal=True, norm=True):
        """return group norms of Theta (observed pw interactions)
        
        Args:
            diagonal (bool): whether to include diagonal.
            norm (bool): if True, aggregate groups using Euclidean norm.
            
        Returns:
            np.array: matrix with aggregated groups.
        """
        theta_marg, unew = self.get_theta(padded=True)
        # theta_marg has 0 discrete block diagonal

        ltot = self.meta['ltot']

        mat_q = theta_marg[:ltot, :ltot]
        if diagonal:
            mat_q += 2 * np.diag(unew)
        mat_r = theta_marg[ltot:, :ltot]
        mat_lbda = -theta_marg[ltot:, ltot:]

        return self._get_group_mat(mat_q,
                                   mat_r,
                                   mat_lbda,
                                   diagonal=diagonal,
                                   norm=norm)

    def compare(self, other, disp=False, threshold=1e-2, **kwargs):
        """Compare this model with another S+L model.

        Args:
            other: another Model_PSWL model instance 
                to which this model is to be compared 
                (typically some 'true' model).
            threshold: threshold for counting edges, non-zero singular values.
            disp (bool): whether to plot differences.

        Note:
            kwargs are passed to _get_group_mat 
            and get_groupnorm_theta methods.

        Returns: 
            tuple: plus_edges (additional edges in model self), minus_edges
            (missing edges from model other), diff_rank
            = rank(L_self) - rank(L_other)
        """
        ## S
        self_normmat = self.get_group_mat(**kwargs)
        other_normmat = other.get_group_mat(**kwargs)

        s_graph = self.get_graph(threshold=threshold)
        o_graph = other.get_graph(threshold=threshold)

        diff = np.subtract(s_graph.astype(int), o_graph.astype(int))
        s_plus = np.maximum(diff, 0)  # self - other
        o_plus = np.maximum(-diff, 0)  # other - self
        plus_edges = np.sum(s_plus) / 2
        minus_edges = np.sum(o_plus) / 2

        ## L
        mat_l_self = unpad(self.mat_l, self.meta['sizes'])
        mat_l_other = unpad(other.mat_l, other.meta['sizes'])

        s_spec = np.linalg.eigvals(mat_l_self)
        o_spec = np.linalg.eigvals(mat_l_other)

        rank_o = len(*np.where(o_spec > threshold))
        rank_s = len(*np.where(s_spec > threshold))
        diff_rank = rank_s - rank_o

        #        np.sort(s_spec); np.sort(o_spec)
        #        print('org spec',s_spec, rank_s)
        #        print('learned_spec',o_spec, rank_o)

        ## Theta
        theta_normmat = self.get_groupnorm_theta(**kwargs)
        theta_o_normmat = other.get_groupnorm_theta(**kwargs)

        if disp:  # plotting etc.
            string = 'Diffs(s-o); edges+=%d, edge-=%d, rank=%d'
            print(string % (plus_edges, minus_edges, diff_rank))

            #            vmin = min((np.min(self_normmat), np.min(other_normmat)))
            #            vmax = max((np.max(self_normmat), np.max(other_normmat)))
            #
            #            vmin = min((np.min(Ls), np.min(Lo)))
            #            vmax = max((np.max(Ls), np.max(Lo)))

            vmin = min((np.min(theta_normmat), np.min(theta_o_normmat)))
            vmax = max((np.max(theta_normmat), np.max(theta_o_normmat)))

            # https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar

            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 21))
            plt.subplots_adjust(left=None,
                                bottom=None,
                                right=None,
                                top=.9,
                                wspace=.2,
                                hspace=.25)

            #        fig.suptitle('Sparse matrix S', fontsize=18, y=0.88)
            cbarsize = .75
            axes[0, 0].matshow(self_normmat,
                               interpolation='nearest',
                               vmin=vmin,
                               vmax=vmax)
            axes[0, 0].set_title('S original', y=1.1)
            im1 = axes[0, 1].matshow(other_normmat,
                                     interpolation='nearest',
                                     vmin=vmin,
                                     vmax=vmax)
            axes[0, 1].set_title('S learned', y=1.1)
            fig.colorbar(im1, ax=[axes[0, 0], axes[0, 1]], shrink=cbarsize)

            #        fig.suptitle('Low rank matrix L', fontsize=18, y=0.88)
            axes[1, 0].matshow(mat_l_self,
                               interpolation='nearest',
                               vmin=vmin,
                               vmax=vmax)
            axes[1, 0].set_title('L original', y=1.1)
            im2 = axes[1, 1].matshow(mat_l_other,
                                     interpolation='nearest',
                                     vmin=vmin,
                                     vmax=vmax)
            axes[1, 1].set_title('L learned', y=1.1)
            fig.colorbar(im2, ax=[axes[1, 0], axes[1, 1]], shrink=cbarsize)

            axes[2, 0].matshow(theta_normmat,
                               interpolation='nearest',
                               vmin=vmin,
                               vmax=vmax)
            axes[2, 0].set_title('Theta original', y=1.1)
            im3 = axes[2, 1].matshow(theta_o_normmat,
                                     interpolation='nearest',
                                     vmin=vmin,
                                     vmax=vmax)
            axes[2, 1].set_title('Theta learned', y=1.1)
            fig.colorbar(im3, ax=[axes[2, 0], axes[2, 1]], shrink=cbarsize)

            plt.show()

        return plus_edges, minus_edges, diff_rank

    def get_incoherence(self):
        """Calculate incoherence.
        
        Note:
            coh(L) = max_i ||P_{U(L)} e_i||_2,
            where P_{U(L)} projects onto the row/column space U(L) of L
        
        Returns:
            float: incoherence value.
        """
        lambdas, mat_u = np.linalg.eigh(self.mat_l)
        threshold = 1e-5  # only use eigvals that differ significantly from 0
        ind = np.where(lambdas > threshold)

        # calculate projection P_{U(L)}
        proj_u = np.dot(mat_u[:, list(*ind)], mat_u[:, list(*ind)].T)

        dim = self.meta['ltot'] + self.meta['n_cg']
        maxproj = -np.inf
        for i in range(dim):
            colnorm = np.linalg.norm(proj_u[:, i])  # norm of P_{U(L)}e_i
            if colnorm > maxproj:
                maxproj = colnorm
        return maxproj

    def get_lambdamax(self):
        """return maximal singular value of low-rank matrix"""
        spec = np.linalg.eigvals(self.mat_l)
        spec = np.real(spec)
        return np.max(spec)

    def get_grpnorm_s(self, addu=0):
        """return group l12-norm of sparse component S
        
        Args:
            addu: set to 1.0 if univariate effects (diagonal)
            shall be included in calculation, set to 0 otherwise
        
        Returns:
            float: l_{1,2}-norm.
        """
        mat_q = self.mat_q.copy()
        if addu:
            mat_q += 2 * addu * np.diag(self.vec_u)
        mat_s = self._get_group_mat(mat_q,
                                    self.mat_r,
                                    self.mat_lbda,
                                    diagonal=0,
                                    norm=True,
                                    aggr=True)
        return np.sum(mat_s)

    def get_theta(self, padded=False):
        """get marginal interaction parameters.
        
        Args:
            padded (bool): if true, return 0-padded parameter matrices.

        Returns:
            tuple: (Theta = S + L, u), where discrete block diagonal of Theta
            is set to zero and discrete univariate effects are in u
        """
        mat_s, mat_l, vec_u, _ = self.get_params_sl(padded=padded, clean_l=True)

        theta = mat_s + mat_l  # marginal pw interaction matrix

        return theta, vec_u

    def get_params_sl(self, padded=False, clean_l=False):
        """
        
        Args:
            padded (bool): whether to return padded parameter values.
            clean_l: if set to true, discrete effects on diagonal of d are
                transferred to u, and discrete block diagonal of L is set
                to zero
        
        Returns:
            tuple: (S, L, u, alpha)
            
        Note:
            Density then is
            p(x,y) ~ exp(1/2 (C_x, y)^T (S+L) (C_x y) + u^T C_x + alpha^T y).
        """
        glims = self.meta['cat_glims']
        sizes = self.meta['sizes']
        mat_s = self.get_pw_mat(padded=padded)
        mat_l = self.mat_l.copy(
        )  # contains univariate discrete effects on diagonal
        if clean_l:
            u_from_l = np.diag(mat_l)[:self.meta['ltot']].copy()
            vec_u = self.vec_u + 0.5 * u_from_l

            for r in range(
                    self.meta['n_cat']):  # set block-diagonal of L to zero
                mat_l[glims[r]:glims[r+1], glims[r]:glims[r+1]] = \
                    np.zeros((sizes[r], sizes[r]))
        else:
            vec_u = self.vec_u
        if not padded:
            mat_l = unpad(mat_l, sizes)
            vec_u = unpad(vec_u, sizes)

        return mat_s, mat_l, vec_u, self.alpha

    def get_params(self):
        """return a list of the class parameters"""
        return (self.vec_u, self.mat_q, self.mat_r, self.alpha, 
                self.mat_lbda, self.mat_l)

    def get_meanparams(self):
        """
        wrapper for _get_meanparams from base class
        
        Note:
            Q, R, Lambda are obtained from the compound matrix Theta = S + L.
        
        Returns:
            tuple: mean parameters (p, mus, Sigmas).
        """
        theta, vec_u = self.get_theta(padded=True)
        mat_q, mat_r, mat_lbda = _split_theta(theta, self.meta['ltot'])

        glims = self.meta['cat_glims']
        sizes = self.meta['sizes']

        mat_q = mat_q.copy()
        for r in range(self.meta['n_cat']):  # set block-diagonal to zero
            mat_q[glims[r]:glims[r+1], glims[r]:glims[r+1]] = \
                np.zeros((sizes[r], sizes[r]))

        pwparams = vec_u, mat_q, mat_r, self.alpha, mat_lbda

        return self._get_meanparams(pwparams)

    def to_pwmodel(self):
        """ return model converted to pairwise model (class ModelPW)"""

        theta, vec_u = self.get_theta(padded=True)

        mat_q, mat_r, mat_lbda = _split_theta(theta, self.meta['ltot'])
        pw_params = vec_u, mat_q, mat_r, self.alpha, mat_lbda

        from cgmodsel.models.model_pw import ModelPW
        return ModelPW(pw_params, self.meta)

    def plot_sl(self,
                diagonal=False,
                usegraph=False,
                norm=True,
                diagcutoff=None,
                addu=True,
                normalize_l=False,
                l_abs=False,
                aggr=True,
                notitle=False,
                samescale=True,
                symscale=True,
                save=False,
                cbarscale=1.0,
                plottype='bw'):
        """Plot S+L decomposition.
        
        Note:
            several options are available to make the plots look nice.
            Experimental code.
        """
        cmap = cm.seismic
        plus = True
        if plottype == 'bw':
            aggr = True
            l_abs = 1
            normalize_l = False
            samescale = False
            symscale = False
            cmap = cm.Greys
        elif plottype == 'pn':
            aggr = True
            norm = False
            l_abs = False
            symscale = True
            samescale = False
            usegraph = False
            cmap = cm.seismic

        if usegraph:
            mat_s = self.get_graph(threshold=1e-1)
            mat_s = np.array(mat_s, dtype=np.float)
        else:
            mat_q = self.mat_q.copy()
            if addu and self.meta['n_cat'] > 0:
                scale = 1
                mat_q += scale * 2 * np.diag(self.vec_u)
            mat_s = self._get_group_mat(mat_q,
                                        self.mat_r,
                                        self.mat_lbda,
                                        diagonal=diagonal,
                                        norm=norm,
                                        aggr=aggr)

        if normalize_l:  # perform same aggregations on L as on S
            mat_l_q, mat_l_r, mat_l_lbda = _split_theta(self.mat_l,
                                                        self.meta['ltot'])
            mat_l = self._get_group_mat(mat_l_q,
                                        mat_l_r,
                                        mat_l_lbda,
                                        diagonal=diagonal,
                                        norm=norm,
                                        aggr=aggr)
        else:
            mat_l = unpad(self.mat_l, self.meta['sizes'])
            if l_abs:
                mat_l = np.abs(mat_l)

        if not diagcutoff is None:
            for i in range(mat_s.shape[0]):
                mat_s[i, i] = np.sign(mat_s[i, i]) * \
                    min((diagcutoff, np.abs(mat_s[i, i])))

        svmin = np.min(mat_s)
        svmax = np.max(mat_s)
        _smax = max((svmax, 0))
        _smin = max((0, -svmin))
        srange = max((_smax, _smin))

        lvmin = np.min(mat_l)
        lvmax = np.max(mat_l)
        _lmax = max((lvmax, 0))
        _lmin = max((0, -lvmin))
        lrange = max((_lmax, _lmin))

        if symscale:
            svmin = -srange
            svmax = srange
            lvmin = -lrange
            lvmax = lrange

        if samescale:
            vmin = min((lvmin, svmin))
            vmax = max((lvmax, svmax))

            vmin *= cbarscale
            vmax *= cbarscale

            lvmin = svmin = vmin
            lvmax = svmax = vmax

        ## plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

        plt.subplots_adjust(wspace=.1)
        off = 1.01
        fsize = 20

        notitle = True
        #        fig.suptitle('Parameters S+L model %s'%(caption), fontsize=18, y=0.8)
        axes.flat[0].matshow(mat_s,
                             cmap=cmap,
                             vmin=svmin,
                             vmax=svmax,
                             interpolation='nearest')
        if not notitle:
            axes.flat[0].set_title('S', y=off, fontsize=fsize)
#        if not samecolorbar:
#            fig.colorbar(im0, ax=axes[0], shrink = .5)

        axes.flat[1].matshow(mat_l,
                             interpolation='nearest',
                             cmap=cmap,
                             vmin=lvmin,
                             vmax=lvmax)
        if not notitle:
            axes.flat[1].set_title('L', y=off, fontsize=fsize)
#        if not samecolorbar:
#            fig.colorbar(im1, ax=axes[1], shrink = .5)

#        axes[0].set_axis_off()
        for i in [0, 1]:
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
            axes[i].set_xticks([])
            axes[i].set_yticks([])
#        plt.axis('off')

        if plus:
            trans = fig.transFigure

            fig.text(0.485,
                     0.47,
                     '+',
                     transform=trans,
                     color='black',
                     fontsize=75)

        if save:
            plt.savefig('SLdecomp.pdf', bbox_inches='tight')
        plt.show()

    def repr_graphical(self,
                       diagonal=False,
                       caption='',
                       jointboundaries=True,
                       samecolorbar=True,
                       save=False,
                       norm=False,
                       printspec=False):
        """another plotting function for the model
        
        Note:
            This is experimental code.
        """
        if not norm:
            #            cmap = cm.coolwarm
            cmap = cm.seismic
        else:
            #            cmap = cm.binary
            #            cmap = cm.gist_yarg
            cmap = cm.Greys

        s_normmat = self._get_group_mat(self.mat_q,
                                        self.mat_r,
                                        self.mat_lbda,
                                        diagonal=diagonal,
                                        norm=norm)
        svmin = np.min(s_normmat)
        svmax = np.max(s_normmat)

        #        print(np.round(S_normmat,2 ))

        split_l = _split_theta(self.mat_l, self.meta['ltot'])
        mat_l = self._get_group_mat(*split_l, diagonal=diagonal, norm=norm)

        lvmin = np.min(mat_l)
        lvmax = np.max(mat_l)

        theta_normmat = self.get_groupnorm_theta(diagonal=diagonal, norm=norm)
        tvmin = np.min(theta_normmat)
        tvmax = np.max(theta_normmat)

        ## joint boundaries
        if not norm:
            max_val = max((np.abs(svmin), svmax))
            svmin = -max_val
            svmax = max_val
            max_val = max((np.abs(lvmin), lvmax))
            lvmin = -max_val
            lvmax = max_val
            max_val = max((np.abs(tvmin), tvmax))
            tvmin = -max_val
            tvmax = max_val
        else:
            vmin = 0

        vmin = min([svmin, lvmin, tvmin])
        vmax = max([svmax, lvmax, tvmax])

        if jointboundaries:
            svmin = vmin
            lvmin = vmin
            tvmin = vmin
            svmax = vmax
            lvmax = vmax
            tvmax = vmax
        else:
            string = "ModelPWSL.repr_graphical: not using joint boundaries, cbar is for Theta"
            print(string)

        ## plot
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 10))

        fig.suptitle('Parameters S+L model (%s)' % (caption),
                     fontsize=18,
                     y=0.75)
        im0 = axes.flat[0].matshow(s_normmat,
                                   interpolation='nearest',
                                   vmin=svmin,
                                   vmax=svmax,
                                   cmap=cmap)
        axes.flat[0].set_title(r'$S$', y=1.1)
        if not samecolorbar:
            fig.colorbar(im0, ax=axes[0], shrink=.5)

        im1 = axes.flat[1].matshow(mat_l,
                                   interpolation='nearest',
                                   vmin=lvmin,
                                   vmax=lvmax,
                                   cmap=cmap)
        axes.flat[1].set_title(r'$L$', y=1.1)
        if not samecolorbar:
            fig.colorbar(im1, ax=axes[1], shrink=.5)

        im2 = axes.flat[2].matshow(theta_normmat,
                                   interpolation='nearest',
                                   vmin=tvmin,
                                   vmax=tvmax,
                                   cmap=cmap)
        axes.flat[2].set_title(r'$\Theta=S+L$', y=1.1)
        if not samecolorbar or (not jointboundaries):
            fig.colorbar(im2, ax=axes[2], shrink=.5)
        else:
            fig.colorbar(im2, ax=axes[0:3], shrink=.5)

        if save:
            fig.savefig('pwsl_model.pdf', bbox_inches='tight')
        plt.show()

        if printspec:
            spec = np.linalg.eigvals(self.mat_l)
            self_spec = np.real(spec)  # issues complex warning
            print('spec(>.1):', np.around(self_spec[self_spec > .01], 2))

    def sample(self, n: int, gibbs_iter: int = 10):
        """Sample from the model.
        
        Note:
            Calls the Gibbs samples from Model_PW,
            hence the interace is the same.

        Args:
            n (int): number of samples to be generated. 
            gibbs_iter (int): steps of the Markov Chain.
        
        Returns:
            tuple: cat_data (np.array), cont_data (np.array).
        """
        return self.to_pwmodel().sample(n, gibbs_iter=gibbs_iter)
