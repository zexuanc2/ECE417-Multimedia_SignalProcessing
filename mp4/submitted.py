import os, h5py
from scipy.stats import multivariate_normal
import numpy as  np
from math import inf

###############################################################################
# A possibly-useful utility function
def compute_features(waveforms, nceps=25):
    '''Compute two types of feature matrices, for every input waveform.

    Inputs:
    waveforms (dict of lists of (nsamps) arrays):
        waveforms[y][n] is the n'th waveform of class y
    nceps (scalar):
        Number of cepstra to retain, after windowing.

    Returns:
    cepstra (dict of lists of (nframes,nceps) arrays):
        cepstra[y][n][t,:] = windowed cepstrum of the t'th frame of the n'th waveform of the y'th class.
    spectra (dict of lists of (nframes,nceps) arrays):
        spectra[y][n][t,:] = liftered spectrum of the t'th frame of the n'th waveform of the y'th class. 

    Implementation Cautions:
        Computed with 200-sample frames with an 80-sample step.  This is reasonable if sample_rate=8000.
    '''
    cepstra = { y:[] for y in waveforms.keys() }
    spectra = { y:[] for y in waveforms.keys() }
    for y in waveforms.keys():
        for x in waveforms[y]:
            nframes = 1+int((len(x)-200)/80)
            frames = np.stack([ x[t*80:t*80+200] for t in range(nframes) ])
            spectrogram = np.log(np.maximum(0.1,np.absolute(np.fft.fft(frames)[:,1:100])))
            cepstra[y].append(np.fft.fft(spectrogram)[:,0:nceps])
            spectra[y].append(np.real(np.fft.ifft(cepstra[y][-1])))
            cepstra[y][-1] = np.real(cepstra[y][-1])
    return cepstra, spectra

###############################################################################
# TODO: here are the functions that you need to write
def initialize_hmm(X_list, nstates):
    '''Initialize hidden Markov models by uniformly segmenting input waveforms.

    Inputs:
    X_list (list of (nframes[n],nceps) arrays): 
        X_list[n][t,:] = feature vector, t'th frame of n'th waveform, for 0 <= t < nframes[n]
    nstates (scalar): 
        the number of states to initialize

    Returns:
    A (nstates,nstates):
        A[i,j] = p(state[t]=j | state[t-1]=i), estimates as
        (# times q[t]=j and q[t-1]=i)/(# times q[t-1]=i).
    Mu (nstates,nceps):
        Mu[i,:] = mean vector of the i'th state, estimated as
        average of the frames for which q[t]=i.
    Sigma (nstates,nceps,nceps):
        Sigma[i,:,:] = covariance matrix, i'th state, estimated as
        unbiased sample covariance of the frames for which q[t]=i.
    
    Function:
    Initialize the initial HMM by dividing each feature matrix uniformly into portions for each state:
    state i gets X_list[n][int(i*nframes[n]/nstates):int((i+1)*nframes[n]/nstates,:] for all n.
    Then, from this initial uniform alignment, estimate A, MU, and SIGMA.

    Implementation Cautions:
    - For the last state, (# times q[t-1]=i) is not the same as (# times q[t]=i).
    - "Unbiased" means that you divide by N-1, not N.  In np.cov, that means "bias=False".
    '''
    nwave = len(X_list)
    nceps = X_list[0].shape[1]
    A = np.zeros((nstates,nstates))
    Mu = np.empty((nstates, nceps))
    Sigma = np.empty((nstates,nceps,nceps))
    nframes = np.zeros(nwave)

    for i in range(nwave):
        nframes[i] = X_list[i].shape[0]
    
    state = []
    for i in range(nstates):
        for j in range(nwave):
            curr_wave = X_list[j]
            X_wave = curr_wave[int(i*nframes[j]/nstates):int((i+1)*nframes[j]/nstates),:]
            if j == 0:
                X_waveform = X_wave
            else:
                X_waveform = np.concatenate((X_waveform,X_wave),axis=0)
        state.append(X_waveform)

    for i in range(nstates):
        Mu[i] = np.average(state[i],axis=0)
        Sigma[i] = np.cov(state[i].T, bias=False)
        if i+1 <nstates:
            A[i,i+1] = nwave/state[i].shape[0]
            A[i,i] = 1 - A[i,i+1]
        else:
            A[i,i] = 1
    return A,Mu,Sigma
    
        

    

def observation_pdf(X, Mu, Sigma):
    '''Calculate the log observation PDFs for every frame, for every state.

    Inputs:
    X (nframes,nceps):
        X[t,:] = feature vector, t'th frame of n'th waveform, for 0 <= t < nframes[n]
    Mu (nstates,nceps):
        Mu[i,:] = mean vector of the i'th state
    Sigma (nstates,nceps,nceps):
        Sigma[i,:,:] = covariance matrix, i'th state

    Returns:
    B (nframes,nstates):
        B[t,i] = max(p(X[t,:] | Mu[i,:], Sigma[i,:,:]), 1e-100)

    Function:
    The observation pdf, here, should be a multivariate Gaussian.
    You can use scipy.stats.multivariate_normal.pdf.
    '''
    nframes = X.shape[0]
    nstates = Mu.shape[0]
    B = np.zeros((nframes,nstates))
    
    for t in range(nframes):
        for i in range(nstates):
            B[t,i] = max(multivariate_normal.pdf(X[t,:], Mu[i,:], Sigma[i,:,:]), 1e-100)
    return B



def scaled_forward(A, B):
    '''Perform the scaled forward algorithm.

    Inputs:
    A (nstates,nstates):
        A[i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])

    Returns:
    Alpha_Hat (nframes,nstates):
        Alpha_Hat[t,i] = p(q[t]=i | X[:t,:], A, Mu, Sigma)
        (that's its definition.  That is not the way you should compute it).
    G (nframes):
        G[t] = p(X[t,:] | X[:t,:], A, Mu, Sigma)
        (that's its definition.  That is not the way you should compute it).

    Function:
    Assume that the HMM starts in state q_1=0, with probability 1.
    With that assumption, implement the scaled forward algorithm.
    '''
    nframes, nstates = B.shape
    #pi = np.zeros(nstates)
    #pi[0] = 1
    Alpha_Hat = np.zeros((nframes,nstates))
    g = np.empty((nframes))

    Alpha_Hat[0,0] = B[0,0]
    g[0] = np.sum(Alpha_Hat[0])
    Alpha_Hat[0] /= g[0]

    for t in range(1,nframes):
        for j in range(nstates):
            for i in range(nstates):
                Alpha_Hat[t,j] += Alpha_Hat[t-1,i] * A[i,j] * B[t,j]
        g[t] = np.sum(Alpha_Hat[t])
        Alpha_Hat[t] /= g[t]

    return Alpha_Hat, g
    


def scaled_backward(A, B):
    '''Perform the scaled backward algorithm.

    Inputs:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])

    Returns:
    Beta_Hat (nframes,nstates):
        Beta_Hat[t,i] = p(X[t+1:,:]| q[t]=i, A, Mu, Sigma) / max_j p(X[t+1:,:]| q[t]=j, A, Mu, Sigma)
        (that's its definition.  That is not the way you should compute it).
    '''
    nframes, nstates = B.shape
    Beta_Hat = np.zeros((nframes,nstates))
    c = np.empty(nframes)

    Beta_Hat[nframes-1,:] = 1

    for t in reversed(range(nframes-1)):
        for i in range(nstates):
            for j in range(nstates):
                Beta_Hat[t,i] += A[i,j] * B[t+1,j] * Beta_Hat[t+1,j]
        c[t] = np.max(Beta_Hat[t])
        Beta_Hat[t] /= c[t]
    return Beta_Hat

def posteriors(A, B, Alpha_Hat, Beta_Hat):
    '''Calculate the state and segment posteriors for an HMM.

    Inputs:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])
    Alpha_Hat (nframes,nstates):
        Alpha_Hat[t,i] = p(q=i | X[:t,:], A, Mu, Sigma)
    Beta_Hat (nframes,nstates):
        Beta_Hat[t,i] = p(X[t+1:,:]| q[t]=i, A, Mu, Sigma) / prod(G[t+1:])

    Returns:
    Gamma (nframes,nstates):
        Gamma[t,i] = p(q[t]=i | X, A, Mu, Sigma)
                   = Alpha_Hat[t,i]*Beta_Hat[t,i] / sum_i numerator
    Xi (nframes-1,nstates,nstates):
        Xi[t,i,j] = p(q[t]=i, q[t+1]=j | X, A, Mu, Sigma)
                  = Alpha_Hat[t,i]*A{i,j]*B[t+1,j]*Beta_Hat[t+1,j] / sum_{i,j} numerator

    
    Implementation Warning:
    The denominators, in either Gamma or Xi, might sometimes become 0 because of roundoff error.
    YOU MUST CHECK FOR THIS!
    Only perform the division if the denominator is > 0.
    If the denominator is == 0, then don't perform the division.
    '''
    nframes, nstates = B.shape
    Gamma = np.zeros((nframes, nstates))
    Xi = np.zeros((nframes-1, nstates, nstates))

    for t in range(nframes):
        for i in range(nstates):
            sum_gamma = 0
            for j in range(nstates):
                sum_gamma += Alpha_Hat[t,j] * Beta_Hat[t,j]
            if sum_gamma != 0:
                Gamma[t,i] = Alpha_Hat[t,i] * Beta_Hat[t,i] / sum_gamma
    for t in range(nframes-1):
        for i in range(nstates):
            for j in range(nstates):
                sum_xi = 0
                for m in range(nstates):
                    for n in range(nstates):
                        sum_xi += Alpha_Hat[t,m] * A[m,n] * B[t+1,n] * Beta_Hat[t+1,n]
                if sum_xi != 0:
                    Xi[t,i,j] = Alpha_Hat[t,i] * A[i,j] * B[t+1,j] * Beta_Hat[t+1,j] / sum_xi

    return Gamma, Xi

def E_step(X, Gamma, Xi):
    '''Calculate the expectations for an HMM.

    Inputs:
    X (nframes,nceps):
        X[t,:] = feature vector, t'th frame of n'th waveform
    Gamma (nframes,nstates):
        Gamma[t,i] = p(q[t]=i | X, A, Mu, Sigma)
    Xi (nsegments,nstates,nstates):
        Xi_list[t,i,j] = p(q[t]=i, q[t+1]=j | X, A, Mu, Sigma)
        WARNING: rows of Xi may not be time-synchronized with the rows of Gamma.  

    Returns:
    A_num (nstates,nstates): 
        A_num[i,j] = E[# times q[t]=i,q[t+1]=j]
    A_den (nstates): 
        A_den[i] = E[# times q[t]=i]
    Mu_num (nstates,nceps): 
        Mu_num[i,:] = E[X[t,:]|q[t]=i] * E[# times q[t]=i]
    Mu_den (nstates): 
        Mu_den[i] = E[# times q[t]=i]
    Sigma_num (nstates,nceps,nceps): 
        Sigma_num[i,:,:] = E[(X[t,:]-Mu[i,:])@(X[t,:]-Mu[i,:]).T|q[t]=i] * E[# times q[t]=i]
    Sigma_den (nstates): 
        Sigma_den[i] = E[# times q[t]=i]
    '''
    nframes, nceps = X.shape
    nstates = Xi.shape[1]
    A_num = np.zeros((nstates,nstates))
    A_den = np.zeros((nstates))
    Mu_num = np.zeros((nstates,nceps))
    Mu_den = np.zeros((nstates))
    Sigma_num = np.zeros((nstates,nceps,nceps))
    Sigma_den = np.zeros((nstates))

    for i in range(nstates):
        for j in range(nstates):
            A_num[i,j] = np.sum(Xi[:,i,j])
            A_den[i] = np.sum(Xi[:,i,:])
        for t in range(nframes):
            Mu_num[i] += Gamma[t,i] * X[t]
        Mu_den[i] = np.sum(Gamma[:,i])
        for t in range(nframes):
            Sigma_num[i] += np.outer((X[t] - Mu_num[i] / Mu_den[i]),(X[t] - Mu_num[i] / Mu_den[i])) * Gamma[t,i]
        Sigma_den[i] = np.sum(Gamma[:,i])
    
    return A_num, A_den, Mu_num, Mu_den, Sigma_num, Sigma_den

def M_step(A_num, A_den, Mu_num, Mu_den, Sigma_num, Sigma_den, regularizer):
    '''Perform the M-step for an HMM.

    Inputs:
    A_num (nstates,nstates): 
        A_num[i,j] = E[# times q[t]=i,q[t+1]=j]
    A_den (nstates): 
        A_den[i] = E[# times q[t]=i]
    Mu_num (nstates,nceps): 
        Mu_num[i,:] = E[X[t,:]|q[t]=i] * E[# times q[t]=i]
    Mu_den (nstates): 
        Mu_den[i] = E[# times q[t]=i]
    Sigma_num (nstates,nceps,nceps): 
        Sigma_num[i,:,:] = E[(X[t,:]-Mu[i,:])@(X[t,:]-Mu[i,:]).T|q[t]=i] * E[# times q[t]=i]
    Sigma_den (nstates): 
        Sigma_den[i] = E[# times q[t]=i]
    regularizer (scalar):
        Coefficient used for Tikohonov regularization of each covariance matrix.

    Returns:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i), estimated as
        E[# times q[t]=j and q[t-1]=i]/E[# times q[t-1]=i)].
    Mu (nstates,nceps):
        Mu[i,:] = mean vector of the i'th state, estimated as
        E[average of the frames for which q[t]=i].
    Sigma (nstates,nceps,nceps):
        Sigma[i,:,:] = covariance matrix, i'th state, estimated as
        E[biased sample covariance of the frames for which q[t]=i] + regularizer*I
    '''
    nstates, nceps = Mu_num.shape
    A = np.empty((nstates,nstates))
    Mu = np.empty((nstates,nceps))
    Sigma = np.empty((nstates,nceps,nceps))

    for i in range(nstates):
        A[i] = A_num[i] / A_den[i]
        Mu[i] = Mu_num[i] / Mu_den[i]
        Sigma[i] = Sigma_num[i] / Sigma_den[i] + np.eye(nceps) * regularizer

    return A, Mu, Sigma


def recognize(X, Models):
    '''Perform isolated-word speech recognition using trained Gaussian HMMs.

    Inputs:
    X (list of (nframes[n],nceps) arrays):
        X[n][t,:] = feature vector, t'th frame of n'th waveform
    Models (dict of tuples):
        Models[y] = (A, Mu, Sigma) for class y
        A (nstates,nstates):
             A[i,j] = p(state[t]=j | state[t-1]=i, Y=y).
        Mu (nstates,nceps):
             Mu[i,:] = mean vector of the i'th state for class y
        Sigma (nstates,nceps,nceps):
             Sigma[i,:,:] = covariance matrix, i'th state for class y

    Returns:
    logprob (dict of numpy arrays):
       logprob[y][n] = log p(X[n] | Models[y] )
    Y_hat (list of strings):
       Y_hat[n] = argmax_y p(X[n] | Models[y] )

    Implementation Hint: 
    For each y, for each n,
    call observation_pdf, then scaled_forward, then np.log, then np.sum.
    '''
    classes = Models.keys()
    nwaveforms = len(X)
    logprob = {}
    Y_hat = np.empty((nwaveforms))
    
    # Calculate
    for y in classes:
        logprob[y] = np.empty(nwaveforms)
        for n in range(nwaveforms):
            B = observation_pdf(X[n], Models[y][1], Models[y][2])
            Alpha_hat, G = scaled_forward(Models[y][0], B)
            logprob[y][n] = np.sum(np.log(G))
    
    for n in range(nwaveforms):
        max_value = -inf
        max_y = None
        for y in classes:
            if logprob[y][n] > max_value:
                max_value = logprob[y][n]
                max_y = y
        Y_hat[n] = max_y

    return logprob, Y_hat
