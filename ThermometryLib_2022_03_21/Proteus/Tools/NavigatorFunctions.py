import numpy as np
from scipy import signal

def ConvertSlicePrePulseToNavigatorDisplacement(SlicePrePulse):
    '''
    Converts the data in the SlicePrePulse field of the metadata in the image header to Navigator Displacement
    This is only valid when using the "liver" patch in the scanner to hack this field
    Formula was given by Max Kohler from Philips
    '''
    return (SlicePrePulse/100.0-150.0)*1000.0 

def NavigatorFilterCoeffs(timevector,FrequencyCut):
    fRate=1.0/np.diff(timevector).mean()
    Nyquist=fRate/2.0
    NormCut=FrequencyCut/Nyquist
    b, a = signal.butter(8, NormCut)
    return b,a

def arburg(x,p):
   #   Ref: S. Kay, MODERN SPECTRAL ESTIMATION,
   #              Prentice-Hall, 1988, Chapter 7
   #        S. Orfanidis, OPTIMUM SIGNAL PROCESSING, 2nd Ed.
   #              Macmillan, 1988, Chapter 5
 
    if len(x.flatten()) < p+1:
        raise ValueError('InvalidDimension')
    N  = len(x.flatten())
    x=np.reshape(x,(len(x.flatten()),1))
    
    #% Initialization
    ef = x.copy()
    eb = x.copy()
    a = np.array([[1]])

    #% Initial error
    E=np.zeros(p+1)
    E[0] = x.T.dot(x)/N

    #% Preallocate 'k' for speed.
    k = np.zeros((p,1))
    
    for m in range(p):
       #% Calculate the next order reflection (parcor) coefficient
       efp = ef[1:,[0]]
       ebp = eb[0:-1,[0]]
       num = -2*ebp.T.dot(efp)
       den = efp.T.dot(efp)+ebp.T.dot(ebp)
       
       k[m,0] = num / den;
       
       #% Update the forward and backward prediction errors
       ef = efp + k[m]*ebp
       eb = ebp + k[m]*efp
       
       #% Update the AR coeff.
       tt=np.flipud(a)
       tt=np.vstack((0,np.conj(tt)))
       a=np.vstack((a,0)) + k[m]*(tt)
       
       #% Update the prediction error
       E[m+1] = (1 - k[m]*k[m])*E[m]
    
    return a.flatten(),E[-1],k.flatten()
