'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
import math

def smooth_video(x, sigma, L):
    '''
    y = smooth_video(x, sigma, L)
    Smooth the video using a sampled-Gaussian smoothing kernel.

    x (TxRxC) - a video with T frames, R rows, C columns
    sigma (scalar) - standard deviation of the Gaussian smoothing kernel
    L (scalar) - length of the Gaussian smoothing kernel
    y (TxRxC) - the same video, smoothed in the row and column directions.
    '''
    T,R,C = x.shape
    h_r = np.zeros(L)
    h_c = np.zeros(L)
    z = np.zeros((T,R,C))
    y = np.zeros((T,R,C))
    #temp_r = np.zeros((T,2*R-1,C))
    #temp_c = np.zeros((T,R,2*C-1))
    m = int((L-1)/2)

    for i in range(L):
        h_r[i] = (1/((2*np.pi*sigma**2)**(1/2))) * np.exp(-0.5 * ((i-m)/sigma)**2)
        h_c[i] = (1/((2*np.pi*sigma**2)**(1/2))) * np.exp(-0.5 * ((i-m)/sigma)**2)
        
    #h_r = np.append(h_r, np.zeros(R-L))
    #h_c = np.append(h_c, np.zeros(C-L))

    for n in range(T):
        for i in range(R):
            z[n,i,:] = np.convolve(h_r, x[n,i,:])[m:-m]
        for j in range(C):
            #temp_c[n,:,i] = np.convolve(h_c, z[n,:,i])
            y[n,:,j] = np.convolve(h_c, z[n,:,j])[m:-m]

    return y





def gradients(x):
    '''
    gt, gr, gc = gradients(x)
    Compute gradients using a first-order central finite difference.

    x (TxRxC) - a video with T frames, R rows, C columns
    gt (TxRxC) - gradient in the time direction
    gr (TxRxC) - gradient in the vertical direction
    gc (TxRxC) - gradient in the horizontal direction
    '''
    T,R,C = x.shape
    gt = np.zeros((T,R,C))
    gr = np.zeros((T,R,C))
    gc = np.zeros((T,R,C))
    
    for i in range(1,T-1):
        gt[i,:,:] = 0.5*x[i+1,:,:] - 0.5*x[i-1,:,:]
    for i in range(1,R-1):
        gr[:,i,:] = 0.5*x[:,i+1,:] - 0.5*x[:,i-1,:]
    for i in range(1,C-1):
        gc[:,:,i] = 0.5*x[:,:,i+1] - 0.5*x[:,:,i-1]

    return gt,gr,gc

def lucas_kanade(gt, gr, gc, H, W):
    '''
    vr, vc = lucas_kanade(gt, gr, blocksize)

    gt (TxRxC) - gradient in the time direction
    gr (TxRxC) - gradient in the vertical direction
    gc (TxRxC) - gradient in the horizontal direction
    H (scalar) - height (in rows) of each optical flow block
    W (scalar) - width (in columns) of each optical flow block

    vr (Txint(R/H)xint(C/W)) - pixel velocity in vertical direction
    vc (Txint(R/H)xint(C/W)) - pixel velocity in horizontal direction
    '''
    T,R,C = gt.shape
    vr = np.zeros((T, int(R/H), int(C/W)))
    vc = np.zeros((T, int(R/H), int(C/W)))

    for i in range(T):
        for j in range(int(R/H)):
            for k in range(int(C/W)):
                b = np.array([[-1*np.sum(np.multiply(gr[i,j*H:(j+1)*H,k*W:(k+1)*W],gt[i,j*H:(j+1)*H,k*W:(k+1)*W]))],
                     [-1*np.sum(np.multiply(gc[i,j*H:(j+1)*H,k*W:(k+1)*W],gt[i,j*H:(j+1)*H,k*W:(k+1)*W]))]])
                A = np.array([[np.sum(gr[i,j*H:(j+1)*H,k*W:(k+1)*W]**2), np.sum(np.multiply(gr[i,j*H:(j+1)*H,k*W:(k+1)*W],gc[i,j*H:(j+1)*H,k*W:(k+1)*W]))],
                    [np.sum(np.multiply(gc[i,j*H:(j+1)*H,k*W:(k+1)*W],gr[i,j*H:(j+1)*H,k*W:(k+1)*W])), np.sum(gc[i,j*H:(j+1)*H,k*W:(k+1)*W]**2)]])
                v = np.linalg.inv(A) @ b
                vr[i,j,k] = v[0]
                vc[i,j,k] = v[1]
    return vr, vc
    


def medianfilt(x, H, W):
    '''
    y = medianfilt(x, H, W)
    Median-filter the video, x, in HxW blocks.

    x (TxRxC) - a video with T frames, R rows, C columns
    H (scalar) - the height of median-filtering blocks
    W (scalar) - the width of median-filtering blocks
    y (TxRxC) - y[t,r,c] is the median of the pixels x[t,rmin:rmax,cmin:cmax], where
      rmin = max(0,r-int((H-1)/2))
      rmax = min(R,r+int((H-1)/2)+1)
      cmin = max(0,c-int((W-1)/2))
      cmax = min(C,c+int((W-1)/2)+1)
    '''
    T,R,C = x.shape
    y = np.zeros((T,R,C))
    for t in range(T):
        for r in range(R):
            for c in range(C):
                rmin = max(0,r-int((H-1)/2))
                rmax = min(R,r+int((H-1)/2)+1)
                cmin = max(0,c-int((W-1)/2))
                cmax = min(C,c+int((W-1)/2)+1)
                y[t,r,c] = np.median(x[t, rmin:rmax, cmin:cmax])

    return y

            
def interpolate(x, U):
    '''
    y = interpolate(x, U)
    Upsample and interpolate an image using bilinear interpolation.

    x (TxRxC) - a video with T frames, R rows, C columns
    U (scalar) - upsampling factor
    y (Tx(U*R)x(U*C)) - interpolated image
    '''
    T,R,C = x.shape
    x_ = np.zeros((T,U*R,U*C))
    temp = np.zeros((T,U*R,U*C))
    y = np.zeros((T,U*R,U*C))
    for t in range(T):
        for r in range(R):
            for c in range(C):
                x_[t,r*U,c*U] = x[t,r,c]

    for t in range(T):
        for r in range(U*R):
            for c in range(0,U*C,U):
                fp = [ x_[t,r,c], x_[t,r,min(U*C-1,c+U)] ]
                temp[t,r,c:c+U] = np.interp(np.linspace(0,U,num=U,endpoint=False), np.array([0,U]), fp)
        for c in range(U*C):
            for r in range(0,U*R,U):
                fp = [ temp[t,r,c], temp[t,min(U*R-1,r+U),c] ]
                y [t,r:r+U,c] = np.interp(np.linspace(0,U,num=U,endpoint=False), np.array([0,U]), fp)
    return y

def scale_velocities(v, U): 
    '''
    delta = scale_velocities(v, U)
    Scale the velocities in v by a factor of U,
    then quantize them to the nearest integer.
    
    v (TxRxC) - T frames, each is an RxC velocity image
    U (scalar) - an upsampling factor
    delta (TxRxC) - integers closest to v*U
    '''
    return np.around(v*U)

def velocity_fill(x, vr, vc, keep):
    '''
    y = velocity_fill(x, vr, vc, keep)
    Fill in missing frames by copying samples with a shift given by the velocity vector.

    x (T,R,C) - a video signal in which most frames are zero
    vr (T,Ra,Cb) - the vertical velocity field, integer-valued
    vc (T,Ra,Cb) - the horizontal velocity field, integer-valued
        Notice that Ra and Cb might be less than R and C.  If they are, the remaining samples 
        of y should just be copied from y[t-1,r,c].
    keep (array) -  a list of frames that should be kept.  Every frame not in this list is
     replaced by samples copied from the preceding frame.

    y (T,R,C) - a copy of x, with the missing frames filled in.
    '''
    T,R,C = x.shape
    Ra = vr.shape[1]
    y = np.zeros((T,R,C))

    for t in range(T):
        for r in range(R):
            for c in range(C):
                if t in keep:
                    y[t,r,c] = x[t,r,c]
                elif r >= Ra:
                    y[t,r,c] = y[t-1,r,c]
                else:
                    if r-vr[t-1,r,c] >= 0 and c-vc[t-1,r,c] >= 0:
                        y[t,r,c] = y[t-1, r-vr[t-1,r,c], c-vc[t-1,r,c]]
                    elif r-vr[t-1,r,c] >= 0 and c-vc[t-1,r,c] < 0:
                        y[t,r,c] = y[t-1, r-vr[t-1,r,c], 0]
                    elif r-vr[t-1,r,c] < 0 and c-vc[t-1,r,c] >= 0:
                        y[t,r,c] = y[t-1, 0, c-vc[t-1,r,c]]
                    elif r-vr[t-1,r,c] < 0 and c-vc[t-1,r,c] < 0:
                        y[t,r,c] = y[t-1, 0, 0]
    return y


