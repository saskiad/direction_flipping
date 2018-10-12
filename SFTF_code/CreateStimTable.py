# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:24:20 2017

@author: saskiad
"""

import pandas as pd
import numpy as np
import os
import h5py
import cPickle as pickle
from sync import Dataset

def get_files(exptpath):
    for f in os.listdir(exptpath):
        if f.endswith('.pkl'):
            logpath = os.path.join(exptpath, f)
            print "Stimulus log:", f
        if f.endswith('sync.h5'):
            syncpath = os.path.join(exptpath, f)
            print "Sync file:", f
    return logpath, syncpath

def get_sync(syncpath):
#    head,tail = os.path.split(syncpath)    
    d = Dataset(syncpath)
    sample_freq = d.meta_data['ni_daq']['counter_output_freq']
    
    #microscope acquisition frames    
#    ophys_start = d.get_rising_edges('2p_acquiring')/sample_freq
    twop_vsync_fall = d.get_falling_edges('vsync_2p')/sample_freq
#    twop_vsync_fall = twop_vsync_fall[np.where(twop_vsync_fall > ophys_start)[0]]
    twop_diff = np.ediff1d(twop_vsync_fall)
    acquisition_rate = 1/np.mean(twop_diff)
    
#    stimulus frames
    stim_vsync_fall = d.get_falling_edges('vsync_stim')[1:]/sample_freq          #eliminating the DAQ pulse    
    stim_vsync_diff = np.ediff1d(stim_vsync_fall)
    dropped_frames = np.where(stim_vsync_diff>0.033)[0]
    dropped_frames = stim_vsync_fall[dropped_frames]
    long_frames = np.where(stim_vsync_diff>0.1)[0]
    long_frames = stim_vsync_fall[long_frames]
    print "Dropped frames: " + str(len(dropped_frames)) + " at " + str(dropped_frames)
    print "Long frames(>0.1 s): " + str(len(long_frames)) + " at " + str(long_frames) 
    
    try:
        #photodiode transitions
        photodiode_rise = d.get_rising_edges('photodiode')/sample_freq
    
        #test and correct for photodiode transition errors
        ptd_rise_diff = np.ediff1d(photodiode_rise)
        short = np.where(np.logical_and(ptd_rise_diff>0.1, ptd_rise_diff<0.3))[0]
        medium = np.where(np.logical_and(ptd_rise_diff>0.5, ptd_rise_diff<1.5))[0]
        for i in medium:
            if set(range(i-2,i)) <= set(short):
                ptd_start = i+1
#            elif set(range(i+1,i+3)) <= set(short):
#                ptd_end = i
        ptd_end = np.where(photodiode_rise>stim_vsync_fall.max())[0][0] - 1
    
        if ptd_start > 3:
            print "Photodiode events before stimulus start.  Deleted."
        
        ptd_errors = []
        while any(ptd_rise_diff[ptd_start:ptd_end] < 1.8):
            error_frames = np.where(ptd_rise_diff[ptd_start:ptd_end]<1.8)[0] + ptd_start
            print "Photodiode error detected. Number of frames:", len(error_frames)
            photodiode_rise = np.delete(photodiode_rise, error_frames[-1])
            ptd_errors.append(photodiode_rise[error_frames[-1]])
            ptd_end-=1
            ptd_rise_diff = np.ediff1d(photodiode_rise)
            
        #calculate monitor delay
        first_pulse = ptd_start
        delay_rise = np.empty((ptd_end - ptd_start,1))    
        for i in range(ptd_end+1-ptd_start-1):     
            delay_rise[i] = photodiode_rise[i+first_pulse] - stim_vsync_fall[(i*120)+60]
        
        delay = np.mean(delay_rise[:-1])  
        delay_std = np.std(delay_rise[:-1])
        print "Delay:", round(delay, 4)
        print "Delay std:", round(delay_std, 4)
        if delay_std>0.005:
            print "Sync error needs to be fixed"
#            delay = 0.005   #this appears to be the delay on research rig
            delay = 0.0351
            print "Using assumed delay:", round(delay,4)
    except Exception as e:
        print e
        print "Process without photodiode signal"
        delay = 0.0351
        print "Assumed delay:", round(delay, 4)
            
    #adjust stimulus time with monitor delay
    stim_time = stim_vsync_fall + delay
    
    #convert stimulus frames into twop frames
    twop_frame = np.empty((len(stim_time),1))
    for i in range(len(stim_time)):
        crossings = np.nonzero(np.ediff1d(np.sign(twop_vsync_fall - stim_time[i]))>0)
        try:
            twop_frame[i] = crossings[0][0]
        except:
            twop_frame[i]=np.NaN
            if i > 100:
                print i
                print "Acquisition ends before stimulus"            
    return twop_frame, acquisition_rate
    

'''create sync table'''
def get_sync_table(logpath, twop_frames):        
    print "Loading stimulus log from:", logpath
    f = open(logpath, 'rb')
    data = pickle.load(f)
    f.close()
#    sync_table_full = pd.DataFrame()
    sync_dict = {}
    for a in range(len(data['stimuli'])):
        stim_name = data['stimuli'][a]['stim_path'].split('\\')[-1].split('.')[0]
        print stim_name            
        sweep_order = data['stimuli'][a]['sweep_order']
        sweep_frames = data['stimuli'][a]['sweep_frames']
        display_sequence = data['stimuli'][a]['display_sequence']   #in seconds
        display_sequence += data['pre_blank_sec']   #in seconds
        display_sequence *= int(data['fps'])     #in stimulus frames
        
        stimulus_table = pd.DataFrame(sweep_frames, columns=('start', 'end'))
        stimulus_table['dif'] = stimulus_table['end']-stimulus_table['start']
        stimulus_table.start += display_sequence[0,0]                      
        for seg in range(len(display_sequence)-1):
            for index, row in stimulus_table.iterrows():
                if row.start >= display_sequence[seg,1]:
                    stimulus_table.start[index] = stimulus_table.start[index] - display_sequence[seg,1] + display_sequence[seg+1,0]
        stimulus_table.end = stimulus_table.start+stimulus_table.dif
        print len(stimulus_table)            
        stimulus_table = stimulus_table[stimulus_table.end <= display_sequence[-1,1]]
        stimulus_table = stimulus_table[stimulus_table.start <= display_sequence[-1,1]]            
        print len(stimulus_table)  
        sync_table = pd.DataFrame(np.column_stack((twop_frames[stimulus_table['start']], twop_frames[stimulus_table['end']])), columns=('Start', 'End'))


        
        if stim_name == 'drifting_gratings':
            sync_table['sweep_number'] = sweep_order[:len(stimulus_table)]  #should be able to remove the len(stimulus_table) when stimulus is fixed appropriately
            sweep_table = data['stimuli'][a]['sweep_table']
            dimnames = data['stimuli'][a]['dimnames']
            sweeptable = pd.DataFrame(sweep_table, columns=dimnames)
            sync_table['SF'] = np.NaN            
            sync_table['TF'] = np.NaN
            sync_table['Ori'] = np.NaN
            for index, row in sync_table.iterrows():
                if row['sweep_number'] >= 0:
                    sync_table['TF'][index] = sweeptable['TF'][int(row['sweep_number'])]
                    sync_table['Ori'][index] = sweeptable['Ori'][int(row['sweep_number'])]
                    sync_table['SF'][index] = sweeptable['SF'][int(row['sweep_number'])]
                else:
                    sync_table['TF'][index] = np.NaN
                    sync_table['Ori'][index] = np.NaN
                    sync_table['SF'][index] = np.NaN
        else:    
            sync_table['Frame'] = sweep_order[:len(stimulus_table)]
        sync_dict[str(a)] = sync_table
           
    return sync_dict

#exptpath = r'/Volumes/braintv/workgroups/nc-ophys/ImageData/Saskia/20170531_307744/NaturalScenesUP'            
#logpath, syncpath = get_files(exptpath)
logpath = r'/Volumes/programs/braintv/workgroups/nc-ophys/ImageData/Saskia/20171010_335139/171010135807-DriftingGratingsSFTF.pkl'
syncpath = r'/Volumes/programs/braintv/workgroups/nc-ophys/ImageData/Saskia/20171010_335139/171010_335139_DGSFTF171010125601.h5'
twop_frames, acquisition_rate = get_sync(syncpath)
sync_dict = get_sync_table(logpath, twop_frames)            
            
#        sync_table['Stimulus']=stim_name
#        sync_table_full = pd.concat([sync_table_full, sync_table], ignore_index=True)            
#    sync_table_full.sort(columns='Start', inplace=True, na_position='first')    #need to sort on Start time to find the spontaneous activity epochs
#    sync_table_full.reset_index(drop=True, inplace=True)
#    #get spontaneous activity epochs
#    sp_index = []
#    for index, row in sync_table_full.iterrows():
#        if (index>0) & (row.Start>0):
#            if not sync_table_full.End[index-1]>row.Start-2000:
#                sp_index = np.append(sp_index, index)
#    print "Spontaneous Activity Epochs: ", str(len(sp_index))
#    for spt in sp_index:        
#        temp = pd.DataFrame([['spontaneous',sync_table_full.End[spt-1]+150, sync_table_full.Start[spt]-1]], columns=('Stimulus','Start','End'))
#        sync_table_full = pd.concat([sync_table_full, temp], ignore_index=True)
#    #order columns, using only relevant columns
#    names = ['Stimulus','Start','End','Image','Ori','SF','TF','Phase','Frame']
#    names_final = []            
#    for stn in names:
#        if stn in list(sync_table_full.columns.values):
#            names_final = np.append(names_final, stn)
#    sync_table_full = sync_table_full[names_final]
#    sync_table_full.sort(columns='Start', inplace=True)#, na_position='first')    #want final dataframe to be consistent - eg. sorted by time
#    sync_table_full.reset_index(drop=True, inplace=True)
#try:
#    return sync_table_full, sweeptable
#except:
#    print "No sweeptable"
#    sweeptable = []
#    return sync_table_full, sweeptable