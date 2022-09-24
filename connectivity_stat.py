#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:25:25 2022

@author: pultsinak
"""

import mne
import numpy as np
import pandas as pd
import os
import os.path as op
import mne_connectivity
import xarray

from mne.externals.h5io import write_hdf5


from scipy import stats
from scipy.cluster.hierarchy import fclusterdata
from scipy.stats import ttest_1samp


mne.viz.set_3d_options(antialias=(False))



data_path = '/net/server/data/Archive/prob_learn/vtretyakova/ICA_cleaned'
os.environ['SUBJECTS_DIR'] = '/net/server/data/Archive/prob_learn/freesurfer'
subjects_dir = '/net/server/data/Archive/prob_learn/freesurfer'

subjects = pd.read_csv('/home/pultsinak/Рабочий стол/subj_list.csv')['subj_list'].tolist()
subjects.remove('P062') 
subjects.remove('P052') 
subjects.remove("P032")
subjects.remove('P045') 


rounds = [1, 2, 3, 4, 5, 6]
freq_range = "beta_16_30"
trial_type = ['risk']
feedback = ['positive', 'negative']


fsaverage = mne.setup_source_space(subject = "fsaverage", spacing="ico5", add_dist=False)
labels = mne.read_labels_from_annot(subject="fsaverage", parc='aparc',subjects_dir=subjects_dir)
labels.pop(68)
label_names = [label.name for label in labels]    



############## avereging with including feedback type ###################
for subj in subjects:
    for t in trial_type:
        emp= np.empty((0,68,68))
        for fb in feedback:
            
            for r in rounds:
                try:
                
                    x_pos = xarray.open_dataset('/net/server/data/Archive/prob_learn/pultsinak/connectivity/csd_1500_1900/{0}_run{1}_{2}_fb_cur_positive.netcdf'.format(subj,r,t),engine ="netcdf4")
                    v_pos= xarray.Dataset.to_array(x_pos) # make array to convert to numpy
                    
                    con_array_pos = xarray.DataArray.to_numpy(v_pos)
                    con_array_pos= con_array_pos.reshape(1,68,68)
                
                    x_neg = xarray.open_dataset('/net/server/data/Archive/prob_learn/pultsinak/connectivity/csd_1500_1900/{0}_run{1}_{2}_fb_cur_negative.netcdf'.format(subj,r,t),engine ="netcdf4")
                    v_neg = xarray.Dataset.to_array(x_neg)
                    con_array_neg = xarray.DataArray.to_numpy(v_neg)
                    con_array_neg= con_array_neg.reshape(1,68,68)
                
                except (OSError): 
                    print('This file not exist')
    
            
        con_pos = np.vstack([emp, con_array_pos])
        con_mean_pos=con_pos.mean(axis=0)
        panda_df_pos = pd.DataFrame(data = con_mean_pos, 
                                index =  label_names,
                                columns = label_names)
        xr_pos = panda_df_pos.to_xarray()

        #con_matrix = mne_connectivity.Connectivity(con_with_baseline,freqs, times, n_nodes = con.n_nodes, names=label_names, method=con_methods)
        xr_pos.to_netcdf("/net/server/data/Archive/prob_learn/pultsinak/connectivity/csd_1500_1900_avg_into_fb/{0}_{1}_fb_cur_positive.netcdf".format(subj,t),mode='w')
        
        con_neg = np.vstack([emp, con_array_neg])
        con_mean_neg =con_neg.mean(axis=0)
        panda_df_neg = pd.DataFrame(data = con_mean_neg, 
                                index =  label_names,
                                columns = label_names)
        xr_neg = panda_df_neg.to_xarray()

        #con_matrix = mne_connectivity.Connectivity(con_with_baseline,freqs, times, n_nodes = con.n_nodes, names=label_names, method=con_methods)
        xr_neg.to_netcdf("/net/server/data/Archive/prob_learn/pultsinak/connectivity/csd_1500_1900_avg_into_fb/{0}_{1}_fb_cur_negative.netcdf".format(subj,t),mode='w')
        

        
########## Ttest ##############
risk_fb_cur_positive = xarray.open_dataset('/net/server/data/Archive/prob_learn/pultsinak/connectivity/csd_1500_1900_avg_into_fb/P002_risk_fb_cur_negative.netcdf',engine ="netcdf4")

comp1_per_sub = np.zeros(shape=(len(subjects), 68, 68))
comp2_per_sub = np.zeros(shape=(len(subjects), 68,68))

for ind, subj in enumerate(subjects):
    
    
    try:     
            
        risk_fb_cur_positive = xarray.open_dataset('/net/server/data/Archive/prob_learn/pultsinak/connectivity/csd_1500_1900_avg_into_fb/{0}_risk_fb_cur_positive.netcdf'.format(subj),engine ="netcdf4")
        risk_fb_cur_positive= xarray.Dataset.to_array(risk_fb_cur_positive)
        risk_fb_cur_positive = xarray.DataArray.to_numpy(risk_fb_cur_positive)   
        comp1_per_sub[ind, :, :]= risk_fb_cur_positive
            
        risk_fb_cur_negative = xarray.open_dataset('/net/server/data/Archive/prob_learn/pultsinak/connectivity/csd_1500_1900_avg_into_fb/{0}_risk_fb_cur_negative.netcdf'.format(subj),engine ="netcdf4")
        risk_fb_cur_negative= xarray.Dataset.to_array(risk_fb_cur_negative)
        risk_fb_cur_negative = xarray.DataArray.to_numpy(risk_fb_cur_negative)
        comp2_per_sub[ind, :, :]= risk_fb_cur_negative

    except (OSError): 
        print('This file not exist')
    t, pval = stats.ttest_rel(comp2_per_sub, comp1_per_sub,axis=0)
    print(pval.min(), pval.mean(), pval.max())
        # t-test condition vs zero 
    t_pos,pval_pos_fb= stats.ttest_1samp(comp1_per_sub,0, axis=0)
    t_neg,pval_neg_fb= stats.ttest_1samp(comp2_per_sub,0, axis=0)
        

#replace nan values from pvalue array
pval_pos_fb = np.nan_to_num(pval_pos_fb, nan=1.0)

data_pval = []
for p in pval_pos_fb:
    data = 1 - 10*p
    data_pval.append(data)
data_pval = np.array(data_pval)

data_pval[data_pval<0] = 0





###### Create connectivity circle ##########

lh_labels = [name for name in label_names if name.endswith('lh')]

# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels]

# Save the plot order and create a circular layout
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)
label_colors = [label.color for label in labels]
node_angles = mne.viz.circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])



fig = mne.viz.plot_connectivity_circle(data_pval,label_names,indices=None,n_lines=10,
                             node_angles=node_angles, node_colors=label_colors)


fig[0].savefig("/net/server/data/Archive/prob_learn/pultsinak/connectivity/plots/1500_1900_lp_positive_vs_negative_fb.png", facecolor='black')

 