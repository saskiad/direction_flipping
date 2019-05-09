#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:19:22 2018

@author: saskiad
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
sys.path.append(r'/Users/saskiad/visual_coding_2p_analysis/visual_coding_2p_analysis')
import core

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
manifest_file = core.get_manifest_path()
print manifest_file
boc = BrainObservatoryCache(manifest_file=manifest_file)

analysis_path = r'/Volumes/programs/braintv/workgroups/cortexmodels/michaelbu/ObservatoryPlatformPaperAnalysis/event_analysis_files_2018_09_25'

cell_id = 676753199
fig_save_path = r'/Users/saskiad/Documents/Data/CAM/Direction tuning TF/receptive field plots'
cells = np.load(r'/Users/saskiad/Documents/Data/CAM/Direction tuning TF/reversing_cells.npy')
for cell_id in cells:    
    try:
        lsn_name = 'locally_sparse_noise'
        session_id =  boc.get_ophys_experiments(cell_specimen_ids=[cell_id], stimuli=[lsn_name])[0]['id']
        print "session id: ", session_id
        save_file = os.path.join(analysis_path, 'LocallySparseNoise', str(session_id)+"_lsn_events_analysis.h5")
        sweep_response = pd.read_hdf(save_file, 'sweep_events')
    except:
        lsn_name = 'locally_sparse_noise_8deg'
        session_id = boc.get_ophys_experiments(cell_specimen_ids=[cell_id], stimuli=[lsn_name])[0]['id']
        print "session id: ", session_id
        save_file = os.path.join(analysis_path, 'LocallySparseNoise', str(session_id)+"_lsn_events_analysis.h5")
        sweep_response = pd.read_hdf(save_file, 'sweep_events_8deg')
    
    stim_table, numbercells, specimen_ids = core.get_stim_table(session_id, lsn_name)
    LSN = core.get_stimulus_template(session_id, lsn_name)
    dataset = boc.get_ophys_experiment_data(session_id)
    cell_index= dataset.get_cell_specimen_indices(cell_specimen_ids=[cell_id])[0]
    
    x_shape = LSN.shape[1]
    y_shape = LSN.shape[2]
    
    plt.figure(figsize=(24,20))
    vmax=0
    vmin=0
    for yp in range(y_shape):
        for xp in range(x_shape):
            sp_pt = (yp*x_shape)+xp+1
            on_frame = np.where(LSN[:,xp,yp]==255)[0]
            off_frame = np.where(LSN[:,xp,yp]==0)[0]
            subset_on = sweep_response[stim_table.frame.isin(on_frame)][str(cell_index)]
            subset_off = sweep_response[stim_table.frame.isin(off_frame)][str(cell_index)]
            ax = plt.subplot(y_shape,x_shape,sp_pt)
            ax.plot(subset_on.mean()[14:49], color='r', lw=2)
            ax.plot(subset_off.mean()[14:49], color='b', lw=2)
            ax.axvspan(14, 21 ,ymin=0, ymax=1, facecolor='gray', alpha=0.3)
            vmax = np.where(np.amax(subset_on.mean())>vmax, np.amax(subset_on.mean()), vmax)
            vmax = np.where(np.amax(subset_off.mean())>vmax, np.amax(subset_off.mean()), vmax)
            vmin = np.where(np.amin(subset_on.mean())<vmin, np.amin(subset_on.mean()), vmin)
            vmin = np.where(np.amin(subset_off.mean())<vmin, np.amin(subset_off.mean()), vmin)
            ax.set_xticks([])
            ax.set_yticks([])
    for i in range(1,sp_pt+1):
        ax = plt.subplot(y_shape,x_shape,i)
        ax.set_ylim(vmin, vmax)
    plt.tight_layout()
    plt.suptitle("Cell " + str(cell_id), fontsize=20)
    plt.subplots_adjust(top=0.9)
    filename = 'Traces LSN Cell_'+str(cell_id)+'.png'
    fullfilename = os.path.join(fig_save_path, filename) 
    plt.savefig(fullfilename)   
    plt.close() 


#def plot_LSN_Traces(self):
#    print "Plotting LSN traces for all cells"
#    xtime = np.arange(-1*self.interlength/self.acquisition_rate, (self.sweeplength+self.interlength)/self.acquisition_rate, 1/self.acquisition_rate)
#    while len(xtime)>len(self.sweep_response['0'][0]):#(9*self.sweeplength):
#        xtime = np.delete(xtime, -1)
#    for nc in range(self.numbercells):
#        if np.mod(nc,100)==0:
#            print "Cell #", str(nc)
#        plt.figure(nc, figsize=(24,20))
#        vmax=0
#        vmin=0
#        one_cell = self.sweep_response[str(nc)]
#        for yp in range(16):
#            for xp in range(28):
#                sp_pt = (yp*28)+xp+1
#                on_frame = np.where(self.LSN[:,yp,xp]==255)[0]
#                off_frame = np.where(self.LSN[:,yp,xp]==0)[0]
#                subset_on = one_cell[self.stim_table.Frame.isin(on_frame)]
#                subset_off = one_cell[self.stim_table.Frame.isin(off_frame)]
##                subset_on = one_cell.iloc[on_frame]
##                subset_off = one_cell.iloc[off_frame]
#                ax = plt.subplot(16,28,sp_pt)
#                ax.plot(xtime, subset_on.mean(), color='r', lw=2)
#                ax.plot(xtime, subset_off.mean(), color='b', lw=2)
#                ax.axvspan(0, self.sweeplength/self.acquisition_rate ,ymin=0, ymax=1, facecolor='gray', alpha=0.3)
#                vmax = np.where(np.amax(subset_on.mean())>vmax, np.amax(subset_on.mean()), vmax)
#                vmax = np.where(np.amax(subset_off.mean())>vmax, np.amax(subset_off.mean()), vmax)
#                vmin = np.where(np.amin(subset_on.mean())<vmin, np.amin(subset_on.mean()), vmin)
#                vmin = np.where(np.amin(subset_off.mean())<vmin, np.amin(subset_off.mean()), vmin)
#                ax.set_xticks([])
#                ax.set_yticks([])
#        for i in range(1,sp_pt+1):
#            ax = plt.subplot(16,28,i)
#            ax.set_ylim(vmin, vmax)
#        
#        plt.tight_layout()
#        plt.suptitle("Cell " + str(nc+1), fontsize=20)
#        plt.subplots_adjust(top=0.9)
#        filename = 'Traces LSN Cell_'+str(nc+1)+'.png'
#        fullfilename = os.path.join(self.savepath, filename) 
#        plt.savefig(fullfilename)   
#        plt.close()          



# for each position plot on and off
# save figure

