#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:12:18 2022

@author: pultsinak
"""
import mne
import conpy
from mne.externals.h5io import write_hdf5
import numpy as np
import pandas as pd
import os
import os.path as op
from mne.time_frequency import csd_multitaper
import mne_connectivity
import xarray

mne.viz.set_3d_options(antialias=False)

from mne.connectivity import spectral_connectivity
from mne.minimum_norm import apply_inverse_epochs

data_path = '/net/server/data/Archive/prob_learn/vtretyakova/ICA_cleaned'
os.environ['SUBJECTS_DIR'] = '/net/server/data/Archive/prob_learn/freesurfer'
subjects_dir = '/net/server/data/Archive/prob_learn/freesurfer'


subjects = pd.read_csv('/home/pultsinak/Рабочий стол/subj_list.csv')['subj_list'].tolist()
subjects.remove('P062') 
subjects.remove('P052') 
subjects.remove("P032")
subjects.remove('P045') 

# list of the participants with autism how all cond
subjects = ['P301', 'P304', 'P307', 'P332', 'P312', 'P313', 'P314', 'P316', 'P318', 'P320', 'P321', 
                    'P322', 'P323', 'P325', 'P326', 'P327', 'P328', 'P329', 'P331', 'P333', 'P334', 'P335', 
                    'P336', 'P308', 'P340', 'P341', 'P338', 'P342', 'P324']


rounds = [1, 2, 3, 4, 5, 6]
freq_range = "beta_16_30"
trial_type = ['norisk','risk',]
feedback = ['positive', 'negative']

time_bandwidth = 4
L_freq = 16
H_freq = 31
L_freq = 16
H_freq = 31
f_step = 2
n_cycles=2
freqs = np.arange(16, 31, 2)

period_start = -1.750
period_end = 2.750

baseline = (-0.35, -0.05)



snr = 3.0
lambda2=1.0 / snr ** 2
def read_events_N(events_file):    
    with open(events_file, "r") as f:
        events_raw = np.fromstring(f.read().replace("[", "").replace("]", "").replace("'", ""), dtype=int, sep=" ")
        h = events_raw.shape[0]
        events_raw = events_raw.reshape((h//3, 3))
        return events_raw                       

def make_stc_epochs(subj, r, cond, fb, data_path, baseline, bem, src):

	
    events_pos = read_events_N("/net/server/data/Archive/prob_learn/data_processing/fix_cross_mio_corr/{0}_run{1}_norisk_fb_cur_positive_fix_cross.txt".format(subj, r)) 
    

        # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводим shape к виду (N,3)
    if events_pos.shape == (3,):
        events_pos = events_pos.reshape(1,3)
        
    # download marks of negative feedback      
    
    events_neg = read_events_N("/net/server/data/Archive/prob_learn/data_processing/fix_cross_mio_corr/{0}_run{1}_norisk_fb_cur_negative_fix_cross.txt".format(subj, r))
    
    
    # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводим shape к виду (N,3)
    if events_neg.shape == (3,):
        events_neg = events_neg.reshape(1,3) 
    
    #объединяем негативные и позитивные фидбеки для получения общего бейзлайна по ним, и сортируем массив, чтобы времена меток шли в порядке возрастания    
    events = np.vstack([events_pos, events_neg])
    events = np.sort(events, axis = 0) 
    
    #events, which we need
    events_response = read_events_N('/net/server/data/Archive/prob_learn/data_processing/fix_cross_mio_corr/{0}_run{1}_{2}_fb_cur_{3}.txt'.format(subj, r, cond, fb))
    
    # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводи shape к виду (N,3)
    if events_response.shape == (3,):
        events_response = events_response.reshape(1,3)
    
	           
    raw_fname = op.join(data_path, '{0}/run{1}_{0}_raw_ica.fif'.format(subj, r))

    raw_data = mne.io.Raw(raw_fname, preload=True)
        
    sfreq = raw_data.info['sfreq']
    print("################## FRQ ##################")
    print(sfreq)
    
    picks = mne.pick_types(raw_data.info, meg = True, eog = True)
		    
	# Forward Model
    trans = '/net/server/data/Archive/prob_learn/freesurfer/{0}/mri/T1-neuromag/sets/{0}-COR.fif'.format(subj)
        
	   	    
    #epochs for baseline
    # baseline = None, чтобы не вычитался дефолтный бейзлайн

    epochs_bl = mne.Epochs(raw_data, events, event_id = None, tmin = -1.0, tmax = 1.0, baseline = None, picks = picks, preload = True)
    cov = mne.compute_covariance(epochs=epochs_bl, method='auto', tmin=-0.35, tmax = -0.05)
    
    #epochs_bl.resample(100)
    print(epochs_bl)
    ####### ДЛЯ ДАННЫХ ##############
    # baseline = None, чтобы не вычитался дефолтный бейзлайн
    epochs = mne.Epochs(raw_data, events_response, event_id = None, tmin = 1.500, 
		                tmax = 1.900, baseline = None, picks = picks, preload = True)

     
    fwd = mne.make_forward_solution(info=epochs.info, trans=trans, src=src, bem=bem)	                
    inv = mne.minimum_norm.make_inverse_operator(raw_data.info, fwd, cov, loose=0.2) 	                
		       
    #epochs.resample(100) 
    evoked= epochs.average()
    #print(evoked_bl.data)

    
    #усредняем по времени

    
    stc_epo_list = mne.minimum_norm.apply_inverse_epochs(epochs.pick('meg'), inv, lambda2, method="sLORETA",pick_ori="normal", nave=evoked.nave) 


    return (stc_epo_list)



def make_baseline (subj, r, cond, fb, data_path, baseline, bem, src):

    #bands = dict(beta=[L_freq, H_freq])
    events_pos = read_events_N("/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/fix_cross_mio_corr/{0}_run{1}_norisk_fb_cur_positive_fix_cross.txt".format(subj, r)) 
    

        # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводим shape к виду (N,3)
    if events_pos.shape == (3,):
        events_pos = events_pos.reshape(1,3)
        
    # download marks of negative feedback      
    
    events_neg = read_events_N("/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/fix_cross_mio_corr/{0}_run{1}_norisk_fb_cur_negative_fix_cross.txt".format(subj, r))
    
    
    # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводим shape к виду (N,3)
    if events_neg.shape == (3,):
        events_neg = events_neg.reshape(1,3) 
    
    #объединяем негативные и позитивные фидбеки для получения общего бейзлайна по ним, и сортируем массив, чтобы времена меток шли в порядке возрастания    
    events = np.vstack([events_pos, events_neg])
    events = np.sort(events, axis = 0) 
    
    #events, which we need
    events_response = read_events_N('/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/events_by_cond_mio_corrected/{0}_run{1}_{2}_fb_cur_{3}.txt'.format(subj, r, cond, fb))
    
    # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводи shape к виду (N,3)
    if events_response.shape == (3,):
        events_response = events_response.reshape(1,3)
    
	           
    raw_fname = op.join(data_path, '{0}/run{1}_{0}_raw_ica.fif'.format(subj, r))

    raw_data = mne.io.Raw(raw_fname, preload=True)
    picks = mne.pick_types(raw_data.info, meg = True, eog = True)
		    
	# Forward Model
    trans = '/net/server/mnt/Archive/prob_learn/freesurfer/{0}/mri/T1-neuromag/sets/{0}-COR.fif'.format(subj)
        
	   	    
    #epochs for baseline
    # baseline = None, чтобы не вычитался дефолтный бейзлайн
    epochs_bl = mne.Epochs(raw_data, events, event_id = None, tmin = -1.0, tmax = 1.0, baseline = None, picks = picks, preload = True)
    cov = mne.compute_covariance(epochs=epochs_bl, method='auto', tmin=-0.35, tmax = -0.05)
     
    epochs_bl.resample(100)
    evoked_bl=epochs_bl.average()
    #print(len(epochs_bl))
  
    fwd = mne.make_forward_solution(info=epochs_bl.info, trans=trans, src=src, bem=bem)	                
    inv = mne.minimum_norm.make_inverse_operator(raw_data.info, fwd, cov, loose=0.2) 	                
		       
    #epochs.resample(100)
    stc_epo_list = mne.minimum_norm.apply_inverse_epochs(epochs_bl.pick('meg'), inv, lambda2, method="sLORETA",pick_ori="normal", nave=evoked_bl.nave) 

    return (stc_epo_list)

fmin = 16.
fmax = 32.
#cwt_freqs = np.arange(fmin, fmax, 2)
bem = mne.read_bem_solution('/net/server/data/Archive/prob_learn/data_processing/bem/P301_bem.h5', verbose=None)

con_methods = ['pli']

for subj in subjects:
    
    bem = mne.read_bem_solution('/net/server/data/Archive/prob_learn/data_processing/bem/{0}_bem.h5'.format(subj), verbose=None)
    
    
    src = mne.setup_source_space(subject =subj, spacing='ico5', add_dist=False ) # by default - spacing='oct6' (4098 sources per hemisphere)
    
    
    for r in rounds:
        for cond in trial_type:
            for fb in feedback:
                print(subj,r,cond,fb)

                try:
                    
                
                    #stc_fsaverage = mne.read_source_estimate("/net/server/data/Archive/prob_learn/data_processing/beta_16_30_sources/sLoreta_may_2022/Average_Epo_stc_morphed/{0}_run{1}_{2}_fb_cur_{3}_fsaverage".format(subj,r,cond,fb), subject='fsaverage')
                    stc_epo_list = make_stc_epochs(subj, r, cond, fb, data_path, baseline, bem, src)
                    stc_baseline = make_baseline(subj, r, cond, fb, data_path, baseline, bem, src)
                    
                    
                    fsaverage = mne.setup_source_space(subject = "fsaverage", spacing="ico5", add_dist=False)
                    labels = mne.read_labels_from_annot(subject="fsaverage", parc='aparc',subjects_dir=subjects_dir)
                    labels.pop(68)
                    
                    label_names = [label.name for label in labels]
                    
                    
                    stc_fsaverage_epo_list = []
                    
                    for s in range(len(stc_epo_list)):
                        morph = mne.compute_source_morph(stc_epo_list[s], subject_from=subj, subject_to='fsaverage')
                        stc_fsaverage = morph.apply(stc_epo_list[s])
                        #stc_fsaverage_epo_list.append(stc_fsaverage)
                        
                        stc_fsaverage_epo_list.append(stc_fsaverage)
                        
                    stc_fsaverage_baseline = []
                    for b in range(len(stc_baseline)):
                        morph_baseline = mne.compute_source_morph(stc_baseline[b], subject_from=subj, subject_to='fsaverage')
                        stc_fsaverage_b = morph_baseline.apply(stc_baseline[b])
                        #stc_fsaverage_epo_list.append(stc_fsaverage)
                        
                        stc_fsaverage_baseline.append(stc_fsaverage_b)
                        
                    label_ts = mne.extract_label_time_course(stc_fsaverage_epo_list, labels, src=fsaverage, mode='mean_flip')
                    label_ts_bs = mne.extract_label_time_course(stc_fsaverage_baseline, labels, src=fsaverage, mode='mean_flip')    
                   
                    
                       
                    con = mne_connectivity.spectral_connectivity_epochs(label_ts, names=label_names, method=con_methods, mode='multitaper',sfreq=1000,  fmin=fmin, fmax=fmax, faverage=True, mt_adaptive=True,n_jobs=1)
                    #con.save("/net/server/data/Archive/prob_learn/pultsinak/connectivity/csd_900_300/{0}_run{1}_{2}_fb_cur_{3}.nc".format(subj,r,cond,fb))
                    conmat = con.get_data(output='dense')[:, :, 0] # if matrix 68*68 needed
                    con_bs = mne_connectivity.spectral_connectivity_epochs(label_ts_bs,names=label_names, method=con_methods, mode='multitaper',sfreq=1000,  fmin=fmin, fmax=fmax, faverage=True, mt_adaptive=True,n_jobs=1)
                    
                    con_bs_data=con_bs.get_data()
                    mean_bs= np.mean(con_bs_data)
                    sd_bs = np.std(con_bs_data)
                    
                    con_with_baseline= (np.abs(conmat)-mean_bs)/sd_bs
                    panda_df = pd.DataFrame(data = conmat, 
                                            index =  label_names,
                                            columns = label_names)
                    xr = panda_df.to_xarray()

                    #con_matrix = mne_connectivity.Connectivity(con_with_baseline,freqs, times, n_nodes = con.n_nodes, names=label_names, method=con_methods)
                    xr.to_netcdf("/net/server/data/Archive/prob_learn/pultsinak/connectivity/csd_1500_1900_autists/{0}_run{1}_{2}_fb_cur_{3}.netcdf".format(subj,r,cond,fb),mode='w')
                    
                    
                except (OSError):
                    print('This file not exist')
                    
            
                    