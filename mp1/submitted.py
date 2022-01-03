'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
import math
import h5py
from numpy import random
from numpy.lib.type_check import real

def make_frames(signal, hop_length, win_length):
    '''
    frames = make_frames(signal, hop_length, win_length)

    signal (num_samps) - the speech signal
    hop_length (scalar) - the hop length, in samples
    win_length (scalar) - the window length, in samples
    frames (num_frames, win_length) - array with one frame per row

    num_frames should be enough so that each sample from the signal occurs in at least one frame.
    The last frame may be zero-padded.
    '''
    num_frames = np.ceil(len(signal)/hop_length)
    #temp = len(signal)/num_frames
    frames = np.zeros((int(num_frames), int(win_length)))
    #frames[0,:] = signal[0:win_length]
    for i in range(int(num_frames)):
        start = i*hop_length
        end = start + int(win_length)
        if (start < len(signal)) and (end < len(signal)):
            frames[i] = signal[start:end]
        else:
            frames[i, :len(signal)-start] = signal[start:]
        
    return frames
    


def correlate(frames):
    '''
    autocor = correlate(frames)

    frames (num_frames, win_length) - array with one frame per row
    autocor (num_frames, 2*win_length-1) - each row is the autocorrelation of one frame
    '''
    num_frames, win_length= frames.shape
    autocor = np.zeros((num_frames, 2*win_length-1))
    for i in range(num_frames):
        x = frames[i]
        autocor[i] = np.convolve(x, x[::-1], 'full')
    return autocor

def make_matrices(autocor, p):
    '''
    R, gamma = make_matrices(autocor, p)

    autocor (num_frames, 2*win_length-1) - each row is symmetric autocorrelation of one frame
    p (scalar) - the desired size of the autocorrelation matrices
    R (num_frames, p, p) - p-by-p Toeplitz autocor matrix of each frame, with R[0] on main diagonal
    gamma (num_frames, p) - length-p autocor vector of each frame, R[1] through R[p]
    '''
    num_frames, temp = autocor.shape
    idx_mid = int((temp)/2)
    #print(temp)
    R = np.zeros((num_frames,p,p))
    gamma = np.zeros((num_frames,p))
    for i in range(num_frames):
        gamma[i] = autocor[i,idx_mid+1 : idx_mid+p+1]
        for j in range(p):
            R[i,j] = autocor[i,idx_mid-j : idx_mid-j+p]
    return R, gamma



def lpc(R, gamma):
    '''
    a = lpc(R, gamma)
    Calculate the LPC coefficients in each frame

    R (num_frames, p, p) - p-by-p Toeplitz autocor matrix of each frame, with R[0] on main diagonal
    gamma (num_frames, p) - length-p autocor vector of each frame, R[1] through R[p]
    a (num_frames,p) - LPC predictor coefficients in each frame
    '''
    num_frames, p = gamma.shape
    a = np.zeros((num_frames,p))
    #a_ = np.zeros(p)
    for i in range(num_frames):
        a[i] = np.dot((np.linalg.inv(R[i])),gamma[i])
    return a

def framepitch(autocor, Fs):
    '''
    framepitch = framepitch(autocor, samplerate)

    autocor (num_frames, 2*win_length-1) - autocorrelation of each frame
    Fs (scalar) - sampling frequency
    framepitch (num_frames) - estimated pitch period, in samples, for each frame, or 0 if unvoiced

    framepitch[t] = 0 if the t'th frame is unvoiced
    framepitch[t] = pitch period, in samples, if the t'th frame is voiced.
    Pitch period should maximize R[framepitch]/R[0], in the range 4ms <= framepitch < 13ms.
    Call the frame voiced if and only if R[framepitch]/R[0] >= 0.3, else unvoiced.
    '''
    num_frames, win_length_2 = autocor.shape
    framepitch = np.zeros(num_frames)
    idx_mid = int((win_length_2+1)/2 - 1)

    fp_4ms = idx_mid + int(0.004*Fs)
    fp_13ms = idx_mid + int(0.013*Fs)
    temp = 0
    for i in range(num_frames):
        temp = int(0.004*Fs + np.argmax(autocor[i,fp_4ms:fp_13ms]))
        if np.abs(autocor[i, idx_mid + temp] / autocor[i, idx_mid]) >= 0.3:
            framepitch[i] = temp
        else:
            framepitch[i] = 0
    return framepitch



    
            
def framelevel(frames):
    '''
    framelevel = framelevel(frames)

    frames (num_frames, win_length) - array with one frame per row
    framelevel (num_frames) - framelevel[t] = power (energy/duration) of the t'th frame, in decibels
    '''
    num_frames, win_length = frames.shape
    framelevel = np.zeros(num_frames)
    for i in range(num_frames):
        summation = 0
        for j in range(win_length):
            summation = summation + frames[i,j]**2
        framelevel[i] = 10*np.log10((1/win_length)*summation)
    return framelevel 
    
    


def interpolate(framelevel, framepitch, hop_length):
    '''
    samplelevel, samplepitch = interpolate(framelevel, framepitch, hop_length)

    framelevel (num_frames) - levels[t] = power (energy/duration) of the t'th frame, in decibels
    framepitch (num_frames) - estimated pitch period, in samples, for each frame, or 0 if unvoiced
    hop_length  (scalar) - number of samples between start of each frame
    samplelevel ((num_frames-1)*hop_length+1) - linear interpolation of framelevel
    samplepitch ((num_frames-1)*hop_length+1) - modified linear interpolation of framepitch

    samplelevel is exactly as given by numpy.interp.
    samplepitch is modified so that samplepitch[n]=0 if the current frame or next frame are unvoiced.
    '''
    num_frames = framelevel.shape[0]
    num_samples = (num_frames - 1)*hop_length+1
    frames = np.linspace(0, num_frames, num_frames)
    samples = np.linspace(0, num_frames, num_samples)

    samplelevel = np.interp(samples, frames, framelevel)
    samplepitch = np.interp(samples, frames, framepitch)

    for i in range(num_samples):
        curr_frame = int(np.floor(i/hop_length))
        if framepitch[curr_frame] == 0:
            samplepitch[i] = 0
        elif curr_frame+1 < num_frames:
            if framepitch[curr_frame+1] == 0:
                samplepitch[i] = 0

    return samplelevel, samplepitch

    
def excitation(samplelevel, samplepitch):
    '''
    phase, excitation = excitation(samplelevel, samplepitch)

    samplelevel ((num_frames-1)*hop_length+1) - effective level (in dB) of every output sample
    samplepitch ((num_frames-1)*hop_length+1) - effective pitch period for every output sample
    phase ((num_frames-1)*hop_length+1) - phase of the fundamental at every output sample,
      modulo 2pi, so that 0 <= phase[n] < 2*np.pi for every n.
    excitation ((num_frames-1)*hop_length+1) - LPC excitation signal
      if samplepitch[n]==0, then excitation[n] is zero-mean Gaussian
      if samplepitch[n]!=0, then excitation[n] is a delta function time-aligned to the phase
      In either case, excitation is scaled so its average power matches samplelevel[n].
    '''
    ######## WARNING: the following lines must remain, so that your random numbers will match the grader
    from numpy.random import Generator, PCG64
    rg = Generator(PCG64(1234))
    ## Your Gaussian random numbers must be generated using the command ***rg.normal***
    ## (See https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html).
    ## (1) You must generate them in order, from the beginning to the end of the waveform.
    ## (2) You must generate a random sample _only_ if the corresponding samplepitch[n] > 0.
    temp = samplepitch.shape[0]
    phase = np.zeros(temp)
    excitation = np.zeros(temp)

    for i in range(temp):
        if samplepitch[i] == 0:
            phase[i] = phase[i - 1]
            excitation[i] = rg.normal() * np.sqrt(np.power(10, samplelevel[i] / 10))
        else:
            phase[i] = phase[i - 1] + 2*np.pi / samplepitch[i]
        if phase[i] >= 2*np.pi:
            phase[i] %= 2*np.pi
            excitation[i] = np.sqrt(np.power(10, samplelevel[i] / 10) * samplepitch[i])
    return phase, excitation

    

def synthesize(excitation, a):
    '''
    y = synthesize(excitation, a)
    excitation ((num_frames-1)*hop_length+1) - LPC excitation signal
    a (num_frames,p) - LPC predictor coefficients in each frame
    y ((num_frames-1)*hop_length+1) - LPC synthesized  speech signal
    '''
    temp = excitation.shape[0]
    num_frames, p = a.shape
    hop_length = (temp - 1) / (num_frames-1)
    y = np.zeros(temp)
    for n in range(temp):
        summation = 0
        curr_frame = int(np.floor(n/hop_length))
        for k in range(p):
            if (n-k) < 0:
                break
            else:
                summation += a[curr_frame,k]*y[n-k-1]
        y[n] = excitation[n] + summation
    #print(y[])
    return y

