"""dataset_gen.py: Functions for loading segmented data from the cough counting dataset."""
__author__ = "Lara Orlandic"
__email__ = "lara.orlandic@epfl.ch"

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import sys
import IPython.display as ipd
from enum import Enum
from helpers import *
import json

def get_cough_windows(data_folder, fn, window_len, aug_factor=1):
    """Get the cough segments in a given recording by shifting them within the window
    Inputs:
    - data_folder: location of the recording
    - fn: file name of the ground_truth.json file listing cough locations
    - window_len: desired length of signal window in seconds
    - aug_factor: number of times to shift the cough within the window (i.e. data augmentation)
    Outputs:
    - audio_data: NxMx2 data matrix where 
        - N = number of coughs * augmentation factor
        - M = int(window_len * 16000)
        - first index = outer microphone, second index = body-facing microphone
    - imu_data: NxLx6 data matrix where
        - L = int(window_len * 100)
    - num_coughs: number of coughs in the recording
    """
    # Load cough segment annotations and signals
    with open(fn, 'rb') as f:
        loaded_dict = json.load(f)
    starts = np.array(loaded_dict["start_times"])
    ends = np.array(loaded_dict["end_times"])
    subj_id = fn.split('/')[-6]
    trial = fn.split('/')[-5].split('_')[1]
    mov = fn.split('/')[-4].split('_')[1]
    noise = fn.split('/')[-3].split('_')[2]
    if noise == "someone":
        noise = "someone_else_cough"
    sound = fn.split('/')[-2]
    air, skin = load_audio(data_folder, subj_id, trial, mov, noise, sound)
    imu = load_imu(data_folder, subj_id, trial, mov, noise, sound)
    
    # Set up arrays for storing data
    num_coughs = len(starts)
    window_len_audio = int(window_len*FS_AUDIO)
    window_len_imu = int(window_len*FS_IMU)
    audio_data = np.zeros((num_coughs*aug_factor,window_len_audio,2))
    imu_data = np.zeros((num_coughs*aug_factor,window_len_imu,6))
    
    for a in range(aug_factor):
        # Compute random offsets based on window length and cough lengths
        cough_lengths = ends-starts
        diffs = window_len - cough_lengths
        rand_uni = np.random.uniform(0,diffs)
        window_starts = starts - rand_uni
        end_of_signal = np.min((len(air)/FS_AUDIO,len(imu.x)/FS_IMU))
        #Check if the window exceeds the end of the signal. If so, shift from the end
        exceeds_end = window_starts > ( end_of_signal - window_len)
        if sum(exceeds_end) > 0:
            end_slack = np.max((end_of_signal - ends,np.zeros(ends.shape)), axis=0)
            window_starts[exceeds_end] = np.min((ends[exceeds_end], np.tile(end_of_signal, sum(exceeds_end))),axis=0) - window_len + np.random.uniform(0,np.min((diffs[exceeds_end],end_slack[exceeds_end]))-0.02)


        # Segment audio signals
        window_starts_audio = (window_starts*FS_AUDIO).astype(int)
        window_ends_audio = window_starts_audio + window_len_audio 
        windows_audio_ndx = np.round(np.linspace(window_starts_audio, window_ends_audio, window_len_audio)).astype(int)
        windows_audio_ndx = windows_audio_ndx.T
        windows_audio = np.stack((air[windows_audio_ndx],skin[windows_audio_ndx]),axis=2)
        audio_data[a*num_coughs:((a+1)*num_coughs),:,:] = windows_audio
        
        #Segment IMU signals
        window_starts_imu = (window_starts*FS_IMU).astype(int)
        window_ends_imu = window_starts_imu + window_len_imu 
        windows_imu_ndx = np.round(np.linspace(window_starts_imu, window_ends_imu, window_len_imu)).astype(int)
        windows_imu_ndx = windows_imu_ndx.T
        windows_imu = np.stack((imu.x[windows_imu_ndx],imu.y[windows_imu_ndx],imu.z[windows_imu_ndx],imu.Y[windows_imu_ndx],imu.P[windows_imu_ndx],imu.R[windows_imu_ndx]),axis=2)
        imu_data[a*num_coughs:((a+1)*num_coughs),:,:] = windows_imu
        
    return audio_data, imu_data, num_coughs

def get_non_cough_windows(data_folder,subj_id, trial,mov,noise,sound,n_samp, window_len):
    """Select n_samp audio samples from random locations in the signal with length window_len"""
    #Load data

    air, skin = load_audio(data_folder, subj_id, trial, mov, noise, sound)
    imu = load_imu(data_folder, subj_id, trial, mov, noise, sound)
    window_len_audio = int(window_len*FS_AUDIO)
    window_len_imu = int(window_len*FS_IMU)
    
    #Select random segments
    end_of_signal = np.min((len(air)/FS_AUDIO,len(imu.x)/FS_IMU))
    window_starts = rand_uni = np.random.uniform(0,end_of_signal-window_len,n_samp)
    
    # Segment audio signals
    window_starts_audio = (window_starts*FS_AUDIO).astype(int)
    window_ends_audio = window_starts_audio + window_len_audio 
    windows_audio_ndx = np.round(np.linspace(window_starts_audio, window_ends_audio, window_len_audio)).astype(int)
    windows_audio_ndx = windows_audio_ndx.T
    audio_data = np.stack((air[windows_audio_ndx],skin[windows_audio_ndx]),axis=2)
    
    #Segment IMU signals
    window_starts_imu = (window_starts*FS_IMU).astype(int)
    window_ends_imu = window_starts_imu + window_len_imu 
    windows_imu_ndx = np.round(np.linspace(window_starts_imu, window_ends_imu, window_len_imu)).astype(int)
    windows_imu_ndx = windows_imu_ndx.T
    imu_data = np.stack((imu.x[windows_imu_ndx],imu.y[windows_imu_ndx],imu.z[windows_imu_ndx],imu.Y[windows_imu_ndx],imu.P[windows_imu_ndx],imu.R[windows_imu_ndx]),axis=2)
    
    return audio_data, imu_data

def get_samples_for_subject(data_folder, subj_id, window_len, aug_factor):
    """
    For each subject, extract windows of all of the cough sounds for each movement (sit, walk) and noise condition (none, music, traffic, cough).
    Extract an equal number of non-cough windows for each non-cough sound (laugh, throat, breathe) for the corresponding conditons.
    Inputs: 
    - subj_id: ID number of the subject to process
    - window_len: desired data window length (in seconds)
    - aug_factor: augmentation factor; how many times to randomly shift the signal within the window
    Outputs:
    - audio_data: NxMx2 data matrix where 
        - N = (number of coughs x augmentation factor x 4)
        - M = int(window_len * 16000)
        - first index = outer microphone, second index = body-facing microphone
    - imu_data: NxLx6 data matrix where
        - L = int(window_len * 100)
        - third dimension specifies IMU signal (accel x,y,z, IMU y,p,r)
    - labels: Nx1 vector of labels
        - 1 = cough
        - 0 = non-cough
    - total_coughs: number of un-augmented cough signals for the subject
    """
    # Set up result vectors
    window_len_audio = int(window_len*FS_AUDIO)
    window_len_imu = int(window_len*FS_IMU)
    audio_data = np.zeros((1,window_len_audio,2))
    imu_data = np.zeros((1,window_len_imu,6))
    labels = np.zeros(1)
    total_coughs = 0
    
    # Extract signal windows for each noise condition
    for trial in Trial:
        for mov in Movement:
            for noise in Noise:
                
                # Extract cough windows
                sound = Sound.COUGH
                path = data_folder + subj_id + '/trial_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound
                if os.path.isdir(path) & os.path.isfile(path + '/ground_truth.json'):
                    fn = path + '/ground_truth.json'
                    audio, imu, num_coughs = get_cough_windows(data_folder,fn, window_len, aug_factor)
                    gt = np.ones(audio.shape[0])
                    audio_data = np.concatenate((audio_data,audio), axis=0)
                    imu_data = np.concatenate((imu_data,imu),axis=0)
                    labels = np.concatenate((labels,gt))
                    total_coughs += num_coughs
                    
                    # Extract non-cough windows
                    for sound in Sound:
                        path = data_folder + subj_id + '/trial_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound
                        if (sound != sound.COUGH) & (len(os.listdir(path)) > 0):
                            audio, imu = get_non_cough_windows(data_folder,subj_id, trial,mov,noise,sound,num_coughs*aug_factor, window_len)
                            gt = np.zeros(audio.shape[0])
                            audio_data = np.concatenate((audio_data,audio), axis=0)
                            imu_data = np.concatenate((imu_data,imu),axis=0)
                            labels = np.concatenate((labels,gt))
    
    audio_data = np.delete(audio_data,0,axis=0)
    imu_data = np.delete(imu_data,0,axis=0)
    labels = np.delete(labels,0)
    return audio_data, imu_data, labels, total_coughs