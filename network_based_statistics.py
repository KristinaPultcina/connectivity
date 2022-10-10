#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:46:56 2022

@author: pultsinak
"""

from __future__ import division
import numpy as np
from scipy import stats
from bct.algorithms import get_components

import mne
import pandas as pd
import os
import os.path as op
import mne_connectivity
import xarray
import copy
from mne.externals.h5io import write_hdf5
import statsmodels.stats.multitest as mul


from scipy.cluster.hierarchy import fclusterdata
from scipy.stats import ttest_1samp
from bct.utils import BCTParamError, get_rng
from bct.algorithms import get_components
from bct.due import due, BibTeX


mne.viz.set_3d_options(antialias=(False))
data_path = '/net/server/data/Archive/prob_learn/vtretyakova/ICA_cleaned'
os.environ['SUBJECTS_DIR'] = '/net/server/data/Archive/prob_learn/freesurfer'
subjects_dir = '/net/server/data/Archive/prob_learn/freesurfer'
334,335,059

subjects_nt = ['P023', 'P053', 'P022', 'P016', 'P055', 
               'P019', 'P004', 'P039', 'P008', 'P047',
                'P063', 'P032', 'P017', 'P044']

subjects_aut=  ['P301', 'P304', 'P307',
                    'P321', 'P323', 'P325', 'P327', 'P328',
                    'P329', 'P341']


labels = mne.read_labels_from_annot(subject="fsaverage", parc='HCPMMP1',subjects_dir=subjects_dir)
### remove median wall (indices 0,1)
labels.pop(0)
label_names = [label.name for label in labels] 



######## function from the https://github.com/aestrivex/bctpy.git 
def nbs_bct(x, y, thresh, k=1000, tail='both', paired=False, verbose=False, seed=None):
   
    rng = get_rng(seed)

    def ttest2_stat_only(x, y, tail):
        t = np.mean(x) - np.mean(y)
        n1, n2 = len(x), len(y)
        s = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1)
                     * np.var(y, ddof=1)) / (n1 + n2 - 2))
        denom = s * np.sqrt(1 / n1 + 1 / n2)
        if denom == 0:
            return 0
        if tail == 'both':
            return np.abs(t / denom)
        if tail == 'left':
            return -t / denom
        else:
            return t / denom

    def ttest_paired_stat_only(A, B, tail):
        n = len(A - B)
        df = n - 1
        sample_ss = np.sum((A - B)**2) - np.sum(A - B)**2 / n
        unbiased_std = np.sqrt(sample_ss / (n - 1))
        z = np.mean(A - B) / unbiased_std
        t = z * np.sqrt(n)
        if tail == 'both':
            return np.abs(t)
        if tail == 'left':
            return -t
        else:
            return t

    if tail not in ('both', 'left', 'right'):
        raise BCTParamError('Tail must be both, left, right')

    ix, jx, nx = x.shape
    iy, jy, ny = y.shape

    if not ix == jx == iy == jy:
        raise BCTParamError('Population matrices are of inconsistent size')
    else:
        n = ix

    if paired and nx != ny:
        raise BCTParamError('Population matrices must be an equal size')

    # only consider upper triangular edges
    ixes = np.where(np.triu(np.ones((n, n)), 1))

    # number of edges
    m = np.size(ixes, axis=1)

    # vectorize connectivity matrices for speed
    xmat, ymat = np.zeros((m, nx)), np.zeros((m, ny))

    for i in range(nx):
        xmat[:, i] = x[:, :, i][ixes].squeeze()
    for i in range(ny):
        ymat[:, i] = y[:, :, i][ixes].squeeze()
    del x, y

    # perform t-test at each edge
    t_stat = np.zeros((m,))
    for i in range(m):
        if paired:
            t_stat[i] = ttest_paired_stat_only(xmat[i, :], ymat[i, :], tail)
        else:
            t_stat[i] = ttest2_stat_only(xmat[i, :], ymat[i, :], tail)

    # threshold
    ind_t, = np.where(t_stat > thresh)

    if len(ind_t) == 0:
        raise BCTParamError("Unsuitable threshold")

    # suprathreshold adjacency matrix
    adj = np.zeros((n, n))
    adj[(ixes[0][ind_t], ixes[1][ind_t])] = 1
    # adj[ixes][ind_t]=1
    adj = adj + adj.T

    a, sz = get_components(adj)

    # convert size from nodes to number of edges
    # only consider components comprising more than one node (e.g. a/l 1 edge)
    ind_sz, = np.where(sz > 1)
    ind_sz += 1
    nr_components = np.size(ind_sz)
    sz_links = np.zeros((nr_components,))
    for i in range(nr_components):
        nodes, = np.where(ind_sz[i] == a)
        sz_links[i] = np.sum(adj[np.ix_(nodes, nodes)]) / 2
        adj[np.ix_(nodes, nodes)] *= (i + 2)

    # subtract 1 to delete any edges not comprising a component
    adj[np.where(adj)] -= 1

    if np.size(sz_links):
        max_sz = np.max(sz_links)
    else:
        # max_sz=0
        raise BCTParamError('True matrix is degenerate')
    print('max component size is %i' % max_sz)

    # estimate empirical null distribution of maximum component size by
    # generating k independent permutations
    print('estimating null distribution with %i permutations' % k)

    null = np.zeros((k,))
    hit = 0
    for u in range(k):
        # randomize
        if paired:
            indperm = np.sign(0.5 - rng.rand(1, nx))
            d = np.hstack((xmat, ymat)) * np.hstack((indperm, indperm))
        else:
            d = np.hstack((xmat, ymat))[:, rng.permutation(nx + ny)]

        t_stat_perm = np.zeros((m,))
        for i in range(m):
            if paired:
                t_stat_perm[i] = ttest_paired_stat_only(
                    d[i, :nx], d[i, -nx:], tail)
            else:
                t_stat_perm[i] = ttest2_stat_only(d[i, :nx], d[i, -ny:], tail)

        ind_t, = np.where(t_stat_perm > thresh)

        adj_perm = np.zeros((n, n))
        adj_perm[(ixes[0][ind_t], ixes[1][ind_t])] = 1
        adj_perm = adj_perm + adj_perm.T

        a, sz = get_components(adj_perm)

        ind_sz, = np.where(sz > 1)
        ind_sz += 1
        nr_components_perm = np.size(ind_sz)
        sz_links_perm = np.zeros((nr_components_perm))
        for i in range(nr_components_perm):
            nodes, = np.where(ind_sz[i] == a)
            sz_links_perm[i] = np.sum(adj_perm[np.ix_(nodes, nodes)]) / 2

        if np.size(sz_links_perm):
            null[u] = np.max(sz_links_perm)
        else:
            null[u] = 0

        # compare to the true dataset
        if null[u] >= max_sz:
            hit += 1

        if verbose:
            print(('permutation %i of %i.  Permutation max is %s.  Observed max'
                   ' is %s.  P-val estimate is %.3f') % (
                u, k, null[u], max_sz, hit / (u + 1)))
        elif (u % (k / 10) == 0 or u == k - 1):
            print('permutation %i of %i.  p-value so far is %.3f' % (u, k,
                                                                     hit / (u + 1)))

    pvals = np.zeros((nr_components,))
    # calculate p-vals
    for i in range(nr_components):
        pvals[i] = np.size(np.where(null >= sz_links[i])) / k

    return pvals, adj, null






rounds = [1, 2, 3, 4, 5, 6]

comp1_per_sub = np.zeros(shape=(68,68,len(subjects_nt)))
comp2_per_sub = np.zeros(shape=(68,68,len(subjects_aut)))
                         
for ind, subj in enumerate(subjects_nt):
    for r in rounds:
        print(subj)
        try:
            risk_fb_cur_negative_nt = xarray.open_dataset('/net/server/data/Archive/prob_learn/pultsinak/connectivity/plv/csd_1500_1900_avg_into_fb/{0}_risk_fb_cur_negative.netcdf'.format(subj),engine ="netcdf4")
            risk_fb_cur_negative_nt= xarray.Dataset.to_array(risk_fb_cur_negative_nt)
            risk_fb_cur_negative_nt = xarray.DataArray.to_numpy(risk_fb_cur_negative_nt)
            risk_fb_cur_negative_nt =  risk_fb_cur_negative_nt + risk_fb_cur_negative_nt.T - np.diag(np.diag(risk_fb_cur_negative_nt))

            np.fill_diagonal(risk_fb_cur_negative_nt,0)
        
        except (OSError):
        
            print('This file not exist')
    

    comp1_per_sub[:, :,ind]= risk_fb_cur_negative_nt
    
for ind, subj in enumerate(subjects_aut):
    for r in rounds:

        try:
            
            risk_fb_cur_negative_autists = xarray.open_dataset('/net/server/data/Archive/prob_learn/pultsinak/connectivity/plv/csd_1500_1900_avg_into_fb/{0}_risk_fb_cur_negative.netcdf'.format(subj),engine ="netcdf4")
            risk_fb_cur_negative_autists= xarray.Dataset.to_array(risk_fb_cur_negative_autists)
            risk_fb_cur_negative_autists = xarray.DataArray.to_numpy(risk_fb_cur_negative_autists)

            risk_fb_cur_negative_autists =  risk_fb_cur_negative_autists + risk_fb_cur_negative_autists.T - np.diag(np.diag(risk_fb_cur_negative_autists))
            print(risk_fb_cur_negative_autists)
            np.fill_diagonal(risk_fb_cur_negative_autists,0)
            print(risk_fb_cur_negative_autists)
        
        #risk_fb_cur_negative=np.nan_to_num(risk_fb_cur_negative, nan=1.0)
        except (OSError):
        
            print('This file not exist')
    comp2_per_sub[:, :,ind]= risk_fb_cur_negative_autists

pvals, adj, null= nbs_bct(comp2_per_sub,comp1_per_sub,thresh=1.5, k=5000, tail='both', paired=False, verbose=False, seed=None)  










