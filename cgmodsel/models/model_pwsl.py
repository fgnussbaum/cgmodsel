
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
from cgmodsel.models.model_base import Model_PW_Base
from cgmodsel.models.model_base import unpad, pad, _split_Theta

import matplotlib.pyplot as plt
import numpy as np

import pickle, os

from matplotlib import cm

##################################
class Model_PWSL(Model_PW_Base):
    """
    class for parameters of distribution
    p(x,y) ~ exp(1/2 (C_x, y)^T (S+L) (C_x y) + u^T C_x + alpha^T y)
    
    this class uses padded parameters (that is, parameters for 0-th levels are
    included, however, one might want them to be constrained to 0 for 
    identifiability reasons)
    
    here:
    [C_x .. dummy representation of x]
    u .. univariate discrete parameters
    alpha .. univariate cont. parameters
    L >= 0 .. indirect interactions
    S .. direct interaction given by
        S = (Q & R^T \\ R & -Lambda)
    with 
    Q .. discrete-discrete interactions
    R .. discrete-cont. interactions
    Lambda .. cont-cont interactions
    
    initialize with tuple sl = (u, Q, R, alpha, Lambda, L)
    
    Attention to the special case of only Gaussians observed variables:
    uses as above
        p(y) ~ exp(1/2  y^T (S+L) y + alpha^T y),
    i.e. S has inverted sign
    instead of the representation in Chandrasekaran et al. (2011)
        p(y) ~ exp( -1/2 y^T (S-L) y + alpha^T y)
    
    """

    def __init__(self, sl=None, meta=None, infile = None, annotations = {},
                 in_padded = True): 
        
        if not infile is None:
            sl, meta, annotations2 = self.load(infile)
            annotations.update(annotations2)

        assert (not sl is None) and (not meta is None), "Incomplete data"
        
        *pw, L = sl
        Model_PW_Base.__init__(self, pw, meta, annotations=annotations,
                               in_padded=in_padded)
        if not in_padded:
            self.Lmat = pad(L, self.sizes)
        else:
            self.Lmat = L # low-rank matrix
            
        self.name = 'SL' # model name
        
    def __str__(self): 
        """a string representation for the model"""
        S, L, _, _ = self.get_paramsSL(padded=False)
        
        s ='<pwsl_model S:' + str(S) + '\nL(from Theta=S+L):\n' + str(L) + \
        '\nalpha:' + str(self.alpha.T) + '\nu:' + str(self.u.T)+'>'
        if len(self.annotations.keys())>0:
            s += "\nAnnotations: " + str(self.annotations)
        return s 

    def get_stats(self, threshold=1e-2, **kwargs):
        """"return number of edges of S and rank of L
        threshold ... cutoff for edges and principal components
        """
        spec = np.linalg.eigvals(self.Lmat)
        rank = len(*np.where(spec > threshold))
        
#        print('Model_PWSL.get_stats: maximal singular value:', np.max(spec))
        
        graph = self.get_graph(threshold=threshold, **kwargs)
        graph = graph.astype(int)
        no_edges = int(np.sum(graph) / 2)
        
        return no_edges, rank
        
    def get_groupnorm_theta(self, diagonal=True, norm=True):
        """ return group norms of Theta (observed pw interactions)"""
        ThetaMarg, unew = self.get_theta(padded=True)
        # ThetaMarg has 0 discrete block diagonal
        Q = ThetaMarg[:self.Ltot, :self.Ltot]
        if diagonal:
            Q += 2 * np.diag(unew)
        R = ThetaMarg[self.Ltot:, :self.Ltot]
        Lambda = -ThetaMarg[self.Ltot:, self.Ltot:]

        return self._get_group_mat(Q, R, Lambda, diagonal=diagonal, norm=norm)
        
    def compare(self, other, disp=False, threshold=1e-2, **kwargs):
        """
        compare this model with other S+L model (of the same class)
        
        other ... other model to which self model shall be compared
                    (typically some 'true' model)
        
        threshold ... threshold: when does an edge count as an edge/
                        the same threshold is used for singular values
        
        disp ... whether to plot differences
        
        kwargs are passed to _get_group_mat and get_groupnorm_theta methods
                (diagonal = True/False)
        
        output: plus_edges (additional edges in model self)
                minus_edges (missing edges from model other)
                diff_rank = rank(L_self) - rank(L_other)
        
        """
        ## S
        self_normmat = self._get_group_mat(self.Q, self.R, self.Lambda, **kwargs)
        other_normmat = other._get_group_mat(other.Q, other.R, other.Lambda, **kwargs)

        s_graph = self.get_graph(threshold=threshold, disp=0)        
        o_graph = other.get_graph(threshold=threshold, disp=disp)
        
        diff = np.subtract(s_graph.astype(int), o_graph.astype(int))
        s_plus = np.maximum(diff, 0) # self - other
        o_plus = np.maximum(-diff, 0) # other - self
        plus_edges = np.sum(s_plus) / 2
        minus_edges = np.sum(o_plus) / 2
        
        ## L
        Ls = unpad(self.Lmat, self.sizes)
        Lo = unpad(other.Lmat, other.sizes)
        
        s_spec = np.linalg.eigvals(Ls) # TODO: assert non-complex, scipy.linalg
        o_spec = np.linalg.eigvals(Lo)

        rank_o = len(*np.where(o_spec > threshold))
        rank_s = len(*np.where(s_spec > threshold))
        diff_rank = rank_s - rank_o

#        np.sort(s_spec); np.sort(o_spec)
#        print('org spec',s_spec, rank_s)
#        print('learned_spec',o_spec, rank_o)

        ## Theta
        theta_normmat = self.get_groupnorm_theta(**kwargs)
        theta_o_normmat = other.get_groupnorm_theta(**kwargs)

        if disp: # plotting etc.
            print('Diffs(s-o); edges+=%d, edge-=%d, rank=%d'%(plus_edges, minus_edges, diff_rank))
        
#            vmin = min((np.min(self_normmat), np.min(other_normmat)))
#            vmax = max((np.max(self_normmat), np.max(other_normmat)))
#            
#            vmin = min((np.min(Ls), np.min(Lo)))
#            vmax = max((np.max(Ls), np.max(Lo)))
    
            vmin = min(( np.min(theta_normmat), np.min(theta_o_normmat)))
            vmax = max(( np.max(theta_normmat), np.max(theta_o_normmat)))
            
            # https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 21))
    #        plt.subplots_adjust(left=None, bottom=None, right=None, top=.72, wspace=.2, hspace=.05)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=.9, wspace=.2, hspace=.25)
    
    #        fig.suptitle('Sparse matrix S', fontsize=18, y=0.88)
            cbarsize = .75
            axes[0, 0].matshow(self_normmat, interpolation='nearest', vmin=vmin, vmax=vmax)
            axes[0,0].set_title('S original', y=1.1)
            im = axes[0,1].matshow(other_normmat, interpolation='nearest', vmin=vmin, vmax=vmax)
            axes[0,1].set_title('S learned', y=1.1)
            fig.colorbar(im, ax=[axes[0,0], axes[0,1]], shrink = cbarsize)
    
    #        fig.suptitle('Low rank matrix L', fontsize=18, y=0.88)
            axes[1,0].matshow(Ls, interpolation='nearest', vmin=vmin, vmax=vmax)
            axes[1,0].set_title('L original',y=1.1)
            im2 = axes[1,1].matshow(Lo, interpolation='nearest', vmin=vmin, vmax=vmax)
            axes[1,1].set_title('L learned', y=1.1)
            fig.colorbar(im2, ax=[axes[1,0], axes[1,1]], shrink = cbarsize)
            
    #        fig.suptitle('Marginal interaction matrix Theta', fontsize=18, y=0.88)
            axes[2,0].matshow(theta_normmat, interpolation='nearest', vmin=vmin, vmax=vmax)
            axes[2,0].set_title('Theta original', y=1.1)
            im3 = axes[2,1].matshow(theta_o_normmat, interpolation='nearest', vmin=vmin, vmax=vmax)
            axes[2,1].set_title('Theta learned', y=1.1)
            fig.colorbar(im3, ax=[axes[2,0], axes[2,1]], shrink = cbarsize)
    
            plt.show()

        return plus_edges, minus_edges, diff_rank

    def get_incoherence(self):
        """
        calculate incoherence coh(L) = max_i ||P_{U(L)} e_i||_2,
        where P_{U(L)} projects onto the row/column space U(L) of L 
        """
        lambdas, U = np.linalg.eigh(self.Lmat)
        threshold = 1e-5 # only use eigenvalues that differ significantly from 0
        ind = np.where(lambdas > threshold)
        
        PU = np.dot(U[:, list(*ind)], U[:, list(*ind)].T) # projection P_{U(L)}
        
        m = self.Ltot+self.dg
        maxproj = - np.inf
        for i in range(m):
            colnorm = np.linalg.norm(PU[:, i]) # norm of P_{U(L)}e_i
            if colnorm > maxproj:
                maxproj = colnorm
        return maxproj
    
    def get_lambdamax(self):
        """return maximal singular value of low-rank matrix"""
        spec = np.linalg.eigvals(self.Lmat)
        spec = np.real(spec)
        return np.max(spec)

    def get_grpnormS(self, addu=0):
        """return group l21-norm of sparse component S
        
        addu ... set to 1.0 if univariate effects (diagonal) shall be included
        in calculation, set to 0 otherwise"""
        Q = self.Q.copy()
        if addu:
            Q += 2 * addu * np.diag(self.u)
        S = self._get_group_mat(Q, self.R, self.Lambda,
                                    diagonal=0, norm=True, aggr=True)
        return np.sum(S)

    def get_theta(self, padded=False):
        """
        padded: if true, return 0-padded parameter matrices
        
        returns (Theta = S + L, u)
        where discrete block diagonal of Theta is set to zero
        and discrete univariate effects are in u
        """
        S, L, u, _ = self.get_paramsSL(padded=padded, cleanL=True)

        Theta = S + L # marginal pw interaction matrix
        
        return Theta, u
        
    def get_paramsSL(self, padded=False, cleanL=False):
        """
        returns (S, L, u, alpha) from the pw CG density
            p(x,y) ~ exp(1/2 (C_x, y)^T (S+L) (C_x y) + u^T C_x + alpha^T y)
        
        cleanL: if set to true, discrete effects on diagonal of d are trans-
                ferred to u, and discrete block diagonal of L is set to zero

        padded: if true, return 0-padded parameter matrices
        """

        S = self.get_pw_mat(padded=padded)
        L = self.Lmat.copy() # contains univariate discrete effects on diagonal
        if cleanL:
            u_from_L = np.diag(L)[:self.Ltot].copy()
            u = self.u + 0.5 * u_from_L

            for r in range(self.dc): # set block-diagonal of L to zero
                L[self.Lcum[r]:self.Lcum[r+1], self.Lcum[r]:self.Lcum[r+1]] = \
                    np.zeros((self.sizes[r], self.sizes[r]))
        else:
            u = self.u
        if not padded:
            L = unpad(L, self.sizes)
            u = unpad(u, self.sizes)

        return S, L, u, self.alpha

    def get_params(self):
        """return a list of the class parameters"""
        return self.u, self.Q, self.R, self.alpha, self.Lambda, self.Lmat

    def get_meanparams(self):
        """
        wrapper for _get_meanparams from base class Model_PW_Base
        Q, R, Lambda are obtained from the compound matrix Theta = S + L
        """
        Theta, u = self.get_theta(padded=True)
        Q, R, Lambda = _split_Theta(Theta, self.Ltot)

        Q = Q.copy()
        for r in range(self.dc): # set block-diagonal to zero
            Q[self.Lcum[r]:self.Lcum[r+1], self.Lcum[r]:self.Lcum[r+1]] = np.zeros((self.sizes[r], self.sizes[r]))

        pwparams = u, Q, R, self.alpha, Lambda
        
        return self._get_meanparams(pwparams)

    def to_pwmodel(self):
        """ return model converted to pairwise model (class Model_PW)"""

        Theta, u = self.get_theta(padded = True)

        Q, R, Lambda = _split_Theta(Theta, self.Ltot)
        pw = u, Q, R, self.alpha, Lambda

        meta = {'dg':self.dg, 'dc':self.dc, 'sizes':self.sizes}
        
        from cgmodsel.model_pw import Model_PW
        return Model_PW(pw, meta)
    
    def plotSL(self, diagonal=False, usegraph=False, norm=True,
               samecolorbar=False, caption='', normalize=True,
               diagcutoff=None, addu=True, normalizeL=False,
               Labs=False, aggr=True, notitle=False,
               samescale = True, symscale = True,
               cbarscale = 1.0, plottype = 'bw'):
        """
        plot S+L decomposition
        several options are available to make the plots look nice
        experimental code
        """
        cmap = cm.seismic
        plus = True
        if plottype == 'bw':
            aggr = True
            Labs = 1; normalizeL = False
            samescale = False; symscale = False
            cmap = cm.Greys
        elif plottype == 'pn':
            aggr = True
            norm = False
            Labs = False
            symscale = True; samescale = False
            usegraph = False
            cmap = cm.seismic
        
        if usegraph:
            S = self.get_graph(threshold=1e-1, disp=0)
            S = np.array(S, dtype=np.float)
        else:
            Q = self.Q.copy()
            if addu and self.dc > 0:
                c = 1
                Q += c*2 * np.diag(self.u)
            S = self._get_group_mat(Q, self.R, self.Lambda,
                                    diagonal=diagonal, norm=norm, aggr=aggr)

        if normalizeL: # perform same aggregations on L as on S
            LQ, LR, LLambda = _split_Theta(self.Lmat, self.Ltot)
            L = self._get_group_mat(LQ, LR, LLambda,
                                    diagonal=diagonal, norm=norm, aggr=aggr)
        else:
            L = unpad(self.Lmat, self.sizes)
            if Labs:
                L = np.abs(L)
        
        if not diagcutoff is None:
            for i in range(S.shape[0]):
                S[i, i] = np.sign(S[i,i]) * min((diagcutoff, np.abs(S[i,i])))
        
        Svmin = np.min(S); Svmax = np.max(S)
        _Smax = max((Svmax, 0)); _Smin = max((0, -Svmin))
        Sm = max((_Smax, _Smin))
        
        Lvmin = np.min(L); Lvmax = np.max(L)
        _Lmax = max((Lvmax, 0)); _Lmin = max((0, -Lvmin))
        Lm = max((_Lmax, _Lmin))
        
        if symscale:
            Svmin = -Sm
            Svmax = Sm
            Lvmin = -Lm
            Lvmax = Lm
        
        if samescale:
            vmin = min((Lvmin, Svmin))
            vmax = max((Lvmax, Svmax))

            vmin *= cbarscale
            vmax *= cbarscale

            Lvmin = Svmin = vmin
            Lvmax = Svmax = vmax
        
        ## plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

        plt.subplots_adjust(wspace = .1)
        off = 1.01; fsize = 20

        notitle = True
#        fig.suptitle('Parameters S+L model %s'%(caption), fontsize=18, y=0.8)
        im0 = axes.flat[0].matshow(S,cmap = cmap, vmin=Svmin, vmax=Svmax, interpolation='nearest')
        if not notitle:
            axes.flat[0].set_title('S', y=off, fontsize = fsize)
#        if not samecolorbar:
#            fig.colorbar(im0, ax=axes[0], shrink = .5)
        
        im1 = axes.flat[1].matshow(L, interpolation='nearest', cmap = cmap,
                       vmin=Lvmin, vmax=Lvmax)
        if not notitle:
            axes.flat[1].set_title('L', y=off, fontsize = fsize)
#        if not samecolorbar:
#            fig.colorbar(im1, ax=axes[1], shrink = .5)

#        axes[0].set_axis_off()
        for i in [0,1]:
            axes[i].set_xticklabels([]) 
            axes[i].set_yticklabels([]) 
            axes[i].set_xticks([])
            axes[i].set_yticks([])
#        plt.axis('off')
        
        if plus:
#            import matplotlib.transforms as transforms
#            from matplotlib.lines import Line2D
#            trans = transforms.blended_transform_factory(fig.transFigure, axes.transAxes)
            trans = fig.transFigure
#            a= .49; b = .535
#            lineh = Line2D([a, b], [0.5, 0.5], color='black', transform=trans)
#            linev = Line2D([(a+b)/2, (a+b)/2], [0.44, 0.56], color='black', transform=trans)
#            fig.lines.append(lineh)
#            fig.lines.append(linev)
            
            fig.text(0.485, 0.47, '+', transform=trans, color='black', fontsize=75)


        plt.savefig('plots/SLdecomp.pdf', bbox_inches='tight')
        plt.show()
        
    def repr_graphical(self, diagonal=False, caption='', jointboundaries=True,
                       samecolorbar=True, save=True, norm=False,
                       symboundaries=True, printspec=False):
        """
        another plotting function for the model
        experimental code
        """
        if not norm:
            cmap = cm.coolwarm
            cmap = cm.seismic
        else:
            cmap = cm.binary
            cmap = cm.gist_yarg
            cmap = cm.Greys
            
        S_normmat = self._get_group_mat(self.Q, self.R, self.Lambda,
                                            diagonal=diagonal, norm=norm)
        Svmin = np.min(S_normmat); Svmax = np.max(S_normmat)

#        print(np.round(S_normmat,2 ))

        LQ, LR, LLambda = _split_Theta(self.Lmat, self.Ltot)
        L = self._get_group_mat(LQ, LR, LLambda,
                                    diagonal=diagonal, norm=norm)      
        
        Lvmin = np.min(L); Lvmax = np.max(L)

        theta_normmat = self.get_groupnorm_theta(diagonal=diagonal, norm=norm)
        Tvmin = np.min(theta_normmat); Tvmax = np.max(theta_normmat)
     
        ## joint boundaries
        if not norm:
            m = max((np.abs(Svmin), Svmax))
            Svmin = -m; Svmax = m
            m = max((np.abs(Lvmin), Lvmax))
            Lvmin = -m; Lvmax = m
            m = max((np.abs(Tvmin), Tvmax))
            Tvmin = -m; Tvmax = m
        else:
            vmin = 0

        vmin = min([Svmin, Lvmin, Tvmin]) 
        vmax = max([Svmax, Lvmax, Tvmax])   
        
        if jointboundaries:
            Svmin = vmin; Lvmin = vmin; Tvmin = vmin
            Svmax = vmax; Lvmax = vmax; Tvmax = vmax
        else:
            print("Model_PWSL.repr_graphical: not using joint boundaries, cbar is for Theta")
    
        ## plot
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 10))
        
        fig.suptitle('Parameters S+L model (%s)'%(caption), fontsize=18, y=0.75)
        im0 = axes.flat[0].matshow(S_normmat, interpolation='nearest',
                       vmin=Svmin, vmax=Svmax, cmap = cmap)
        axes.flat[0].set_title(r'$S$', y=1.1)
        if not samecolorbar:
            fig.colorbar(im0, ax=axes[0], shrink = .5)
        
        im1 = axes.flat[1].matshow(L, interpolation='nearest',
                       vmin=Lvmin, vmax=Lvmax, cmap =cmap)
        axes.flat[1].set_title(r'$L$', y=1.1)
        if not samecolorbar:
            fig.colorbar(im1, ax=axes[1], shrink = .5)

        im2 = axes.flat[2].matshow(theta_normmat, interpolation='nearest',
                       vmin=Tvmin, vmax=Tvmax, cmap =cmap)
        axes.flat[2].set_title(r'$\Theta=S+L$', y=1.1)
        if not samecolorbar or (not jointboundaries):
            fig.colorbar(im2, ax=axes[2], shrink = .5)
        else:
            fig.colorbar(im2, ax=axes[0:3], shrink = .5)
        
        if save:
            fig.savefig('plots/pwsl_model.pdf', bbox_inches='tight')
        plt.show()
        
        if printspec:
            spec = np.linalg.eigvals(self.Lmat)
            self_spec = np.real(spec) # issues complex warning
            print('spec(>.1):', np.around(self_spec[self_spec>.01],2))

###############################################################################
# Model IO
###############################################################################
    def update_annotations(self, **kwargs):
        """add anotations to the model by specifying keyword args"""
        self.annotations.update(kwargs)
        
    def save(self, outfile=None, foldername="savedmodels", trial=None):
        """save model to file in folder <foldername>
        
        """
        params = list(self.get_params()) + [self.dc, self.dg, self.sizes, self.annotations]

        if not os.path.exists(foldername):
            os.mkdir(foldername)
            print("Directory ", foldername,  " Created ")
        
        if outfile is None: # try to construct filename from annotations and 
            if trial == -1:
                outfile = "d%dgen"%(self.dc+self.dg)
            else:
                try:
                    n = self.annotations['n']
    #                seed = self.annotations['seed']
                except:
                    raise("No filename provided and could not construct filename")
                outfile = "d%dn%d"%(self.dc + self.dg, n)
            if not trial is None:
                outfile += "_t%d"%(trial)
            outfile = "%s/%s.npy"%(foldername, outfile)
        pickle.dump(params, open(outfile, "wb"))
        
        return outfile

    def load(self, infile):
        """load a model from file"""
        params = pickle.load(open(infile, "rb"))

        pwslparams = params[:-4]
        meta = {'dc':params[-4], 'dg':params[-3], 'sizes':params[-2]}
        annotations = params[-1]
        
        return pwslparams, meta, annotations