"""helpers.py: Functions for quickly looping through the cough counting dataset, loading biosignal files, and setting up constants."""
__author__ = "Lara Orlandic"
__email__ = "lara.orlandic@epfl.ch"

from enum import Enum
import os
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
import numpy as np

##### Sampling frequencies of the sensors #####
FS_AUDIO = 16000
FS_IMU = 100

##### Enums for easily accessing files #####
class Trial(str, Enum):
    # Trial number (1-3) of the experiment on a given subject
    ONE = '1'
    TWO = '2'
    THREE = '3'
    
class Movement(str, Enum):
    # Kinematic noise scenarios
    SIT = 'sit'
    WALK = 'walk'

class Noise(str, Enum):
    # Audio noise scenarios
    MUSIC = 'music'
    NONE = 'nothing'
    COUGH = 'someone_else_cough'
    TRAFFIC = 'traffic'
    
class Sound(str, Enum):
    # Sound that the subject performs
    COUGH = 'cough'
    LAUGH = 'laugh'
    BREATH = 'deep_breathing'
    THROAT = 'throat_clearing'

class IMU_Signal(str, Enum):
    x = "Accel X"
    y = "Accel Y"
    z = "Accel Z"
    Y = "Gyro Y"
    P = "Gyro P"
    R = "Gyro R"

class IMU_Short(str, Enum):
    x = "x"
    y = "y"
    z = "z"
    Y = "Y"
    P = "P"
    R = "R"
    
##### Data loading functions #####
def load_audio(folder, subject_id, trial, mov, noise, sound, normalize_1=False):
    """
    Load the audio signals (Both body-facing and outward-facing) of a given recording
        Inputs:
            - folder: string, folder where the database is stored
            - subject_id: string, numerical ID of the subject 
            - trial: Trial Enum, which trial the recording was part of
            - mov: Movement Enum, specifies kinematic noise condition of the recording
            - noise: Noise Enum, audio noise condition of the recording
            - sound: Sound Enum, which noise was being performed (ex. cough, laugh, etc.)
            - normalize_1: Whether to normalize recording s.t. it has a mean of zero and maximum absolute value of 1
        Outputs:
            - audio_air: outward-facing microphone signal
            - audio_skin: body-facing micriphone signal
    """
    
    fn = subject_id + '/trial_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound + '/'
    
    try:        
        fs_aa, audio_air = wavfile.read(folder + fn + "outward_facing_mic.wav")
    except FileNotFoundError as err:
        print("ERROR: Air mic file not found")

    try:        
        fs_as, audio_skin = wavfile.read(folder + fn + "body_facing_mic.wav")
    except FileNotFoundError as err:
        print("ERROR: Skin mic file not found")
    
    if (fs_aa != fs_as):
        print("ERROR: Mismatching sampling rates")
   
    
    if normalize_1:
        #Normalize recordings to [-1, +1] range
        audio_air = audio_air - np.mean(audio_air)
        audio_air = audio_air/(np.max(np.abs(audio_air))+1e-17)
        audio_skin = audio_skin - np.mean(audio_skin)
        audio_air = audio_skin/(np.max(np.abs(audio_skin))+1e-17)
    else:
        # Normalize recordings based on maximum value
        max_val = 1<<29
        audio_air = audio_air/max_val
        audio_skin = audio_skin/max_val
    
    return audio_air, audio_skin

def get_audio_time(audio_sig):
    """Return the time of a given audio recording"""
    return np.arange(0,len(audio_sig)/FS_AUDIO,1/FS_AUDIO)

class IMU:
    fs = 100
    def __init__(self, Y,P,R,x,y,z):
        self.x=x
        self.y=y
        self.z=z
        self.Y=Y
        self.P=P
        self.R=R
    def normalize(self):
        self.x = self.x - np.mean(self.x)
        self.x = self.x/np.max(np.abs(self.x))
        self.y = self.y - np.mean(self.y)
        self.y = self.y/np.max(np.abs(self.y))
        self.z = self.z - np.mean(self.z)
        self.z = self.z/np.max(np.abs(self.z))
        self.Y = self.Y - np.mean(self.Y)
        self.Y = self.Y/np.max(np.abs(self.Y))
        self.P = self.P - np.mean(self.P)
        self.P = self.P/np.max(np.abs(self.P))
        self.R = self.R - np.mean(self.R)
        self.R = self.R/np.max(np.abs(self.R))
    def standardize(self):
        self.x = self.x - np.mean(self.x)
        self.x = self.x/np.std(self.x)
        self.y = self.y - np.mean(self.y)
        self.y = self.y/np.std(self.y)
        self.z = self.z - np.mean(self.z)
        self.z = self.z/np.std(self.z)
        self.Y = self.Y - np.mean(self.Y)
        self.Y = self.Y/np.std(self.Y)
        self.P = self.P - np.mean(self.P)
        self.P = self.P/np.std(self.P)
        self.R = self.R - np.mean(self.R)
        self.R = self.R/np.std(self.R)
    def get_time(self):
        if self.x is not None:
            time = np.arange(0,len(self.x)/self.fs,1/self.fs)
            if len(time) > len(self.x):
                return time[:-1]
            return time
    def plot(self):
        fig, axs = plt.subplots(6,1, figsize=(10,21))
        time = self.get_time()
        axs[0].plot(time,self.x, label='Accel X')
        axs[0].set_title("Accel X")
        axs[1].plot(time,self.y, label='Accel Y')
        axs[1].set_title("Accel Y")
        axs[2].plot(time,self.z, label='Accel Z')
        axs[2].set_title("Accel Z")
        axs[3].plot(time,self.Y, label='Gyro Y')
        axs[3].set_title("Gyro Y")
        axs[4].plot(time,self.P, label='Gyro P')
        axs[4].set_title("Gyro P")
        axs[5].plot(time,self.R, label='Gyro R')
        axs[5].set_title("Gyro R")
        axs[5].set_xlabel("Time (s)")
    def set_fs(self,fs_new):
        fs=fs_new
    def make_segment_df(self):
        df_cough = pd.DataFrame({})
        df_cough['Accel x'] = self.x
        df_cough['Accel y'] = self.y
        df_cough['Accel z'] = self.z
        df_cough['Gyro Y'] = self.Y
        df_cough['Gyro P'] = self.P
        df_cough['Gyro R'] = self.R
        return df_cough

    
def delineate_imu(imu_z):
    """Return the peaks, valleys, and second derivative of the IMU Z signal"""
    deriv_imu = np.gradient(imu_z)
    fs_downsample = 1000
    b, a = butter(4, fs_downsample/FS_AUDIO, btype='lowpass') # 4th order butter lowpass filter
    deriv_imu_filt = filtfilt(b, a, deriv_imu)
    deriv_imu_filt = deriv_imu_filt/np.max(np.abs(deriv_imu_filt))
    deriv_imu = deriv_imu/np.max(np.abs(deriv_imu))
    second_deriv_imu = np.gradient(deriv_imu_filt)
    second_deriv_imu = second_deriv_imu/np.max(np.abs(second_deriv_imu))
    imu_valleys, _ = find_peaks(second_deriv_imu)
    imu_pks, _ = find_peaks(-second_deriv_imu)
    return imu_pks, imu_valleys, second_deriv_imu


def segment_cough(x,fs, cough_padding=0.2,min_cough_len=0.2, th_l_multiplier = 0.1, th_h_multiplier = 2):
    """Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power.
    Adapted from the COUGHVID repository: https://c4science.ch/diffusion/10770/
    
    Inputs:
    *x (np.array): cough signal
    *fs (float): sampling frequency in Hz
    *cough_padding (float): number of seconds added to the beginning and end of each detected cough to make sure coughs are not cut short
    *min_cough_length (float): length of the minimum possible segment that can be considered a cough
    *th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
    *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator
    
    Outputs:
    *coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    *cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress
    *starts (np.array): start indices of the coughs
    *ends (np.array): end indies of the coughs 
    *peaks (np.array): peak amplitude values of the coughs 
    *peak_locs (np.array): indices of the peaks
    """
                
    cough_mask = np.array([False]*len(x))
    

    #Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h =  th_h_multiplier*rms

    #Segment coughs
    coughSegments = []
    padding = round(fs*cough_padding)
    min_cough_samples = round(fs*min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01*fs)
    below_th_counter = 0
    
    segment_indices = []
    
    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample<seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding>min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end+1])
                        segment_indices.append((cough_start,cough_end))
                        cough_mask[cough_start:cough_end+1] = True
            elif i == (len(x)-1):
                cough_end=i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding>min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end+1])
                    segment_indices.append((cough_start,cough_end))
            else:
                below_th_counter = 0
        else:
            if sample>seg_th_h:
                cough_start = i-padding if (i-padding >=0) else 0
                cough_in_progress = True
    
    starts = np.zeros(len(segment_indices))
    for i, ndx in enumerate(segment_indices):
        starts[i] = ndx[0]
    ends = np.zeros(len(segment_indices))
    for i, ndx in enumerate(segment_indices):
        ends[i] = ndx[1]
    peaks = []
    peak_locs = []
    
    #Find the peak (maximum amplitude) of each cough signal
    for s, e in zip(starts,ends):
        sig = x[round(s):round(e)]
        pk = np.max(sig)
        loc = np.argmax(sig)
        peaks.append(pk)
        peak_locs.append(round(s)+loc)
    
    return coughSegments, cough_mask, starts, ends, peaks, peak_locs

def find_nth_closest_point(peak, f_points, pos='before',n=0):
    """Find the nth closest point in array f_points before or after a given peak"""
    if pos == 'before':
        distances = peak - f_points
    elif pos == 'after':
        distances = f_points - peak
    distances[distances<0] = 10
    return f_points[np.argsort(distances)[n]]
    
def load_imu(folder, subject_id, trial, mov, noise, sound):
    """Load the IMU signal from file into an IMU object"""
    fn = subject_id + '/trial_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound + '/imu.csv'

    try:        
        df = pd.read_csv(folder + fn)
    except FileNotFoundError as err:
        print("ERROR: IMU file not found")
        return 0
    
    
    Y = df['Gyro Y'].to_numpy()
    P = df['Gyro P'].to_numpy()
    R = df['Gyro R'].to_numpy()
    x = df['Accel x'].to_numpy()
    y = df['Accel y'].to_numpy()
    z = df['Accel z'].to_numpy()
    
    imu = IMU(Y,P,R,x,y,z) 
    
    return imu
