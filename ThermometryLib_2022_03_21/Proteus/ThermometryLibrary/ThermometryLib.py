'''
BSD License 2.0
Copyright (c) 2018, Samuel Pichardo
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of the University of Calgary nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL University of Calgary, Samuel Pichardo or an of the contributors BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
from __future__ import print_function

import math
import sys
from math import pi as pi

import numpy
import numpy as np

if sys.version_info > (3, 2):
    pass
    # MS (2021.03.21): Testing just loading the old library for Python 3.6
    # from Proteus.ThirdParty.mkl_fft import mkl_fft
else:
    pass

from operator import itemgetter
from scipy import signal
import matplotlib.path as PT
from Proteus.pyMRI.RescaleImages import RescaleMagnitude,RescalePhase
from Proteus.ThermometryLibrary.DriftAdaptative2D import DriftAdaptative2D
from Proteus.Tools.SortedCollection import SortedCollection
from Proteus.Tools.parseROIString import parseROIString
import scipy.interpolate as inter
from scipy.interpolate import UnivariateSpline

try:
    from skimage.restoration import unwrap_phase
except:
    print(" {} : not imported. Continue... ".format("unwrap_phase"))
import cProfile

LOGGER_NAME = 'MRgHIFU.by.Proteus'

import logging

logger = logging.getLogger(LOGGER_NAME)


# Calculate the phase change to temperature change conversion constant
def calculate_dP_to_dT_constant( echo_time, alpha=0.0094e-6, beta=3, gamma=42.58e6 ):
  K = 1.0/(-2.0*pi * alpha * beta * gamma * echo_time)
  return K

#Sep 26, why we are using this?

#def Oldnormalize_phase_image( P ):
#    IMAGE_MAX = 4095.0
#    P_normal = pi*(2*P.astype(np.float32)/IMAGE_MAX-1) # put on [-pi, pi]
#    return P_normal

def Normalize_phase_image_internal_dynamic( P ):
    #we use the fact that we know that the data is supposed to fit [-pi,pi]
    PMax=P.max()
    PMin=P.min()
    P_normal = -pi+(P.astype(np.float32)-PMin)/(PMax-PMin)*2*pi # put on [-pi, pi]
    return P_normal

def Oldnormalize_magnitude_image( M ):
   IMAGE_MAX = 4095.0
   M_normal = np.float32(M)/IMAGE_MAX  #put on [0, 1]
   return M_normal

# Get a noise mask from magnitude image and reference
def get_noise_mask( M, M_reference, K, T_acceptable_deviation ):
   M_reference = M - M_reference
   M_threshold = (abs(K) *M_reference.flatten(1).std(axis=0))/ (T_acceptable_deviation * math.sqrt(2)) #an acceptable standard deviation in the phase
   noise_mask = M > M_threshold
   return noise_mask

def get_noise_mask2(mag1, mag0, limit = 3, multiplier = 1.0, PhaseToTemperatureConversionFactor = -0.07355958):
    ''' Creates an SNR based mask similarly as in HIFU application. Mask gives True for the accepted voxels.'''
    Id = np.mean(mag1)/np.mean(mag0)*mag0-mag1
    n = np.std(Id)/math.sqrt(2)
    C = -1/PhaseToTemperatureConversionFactor
    inv = mag1.copy()
    #inv[mag1 == 0.0] = np.NAN
    inv[mag1 == 0.0] = 1
    u = C*n/inv
    mask = u < limit*multiplier
    mask[mag1==0.0]=False
    mask[mask == np.NAN] = False
    return mask

def stdDevDiffImage(  img_cur, img_prev, VoxelSize, incl_roiSq_size_mm, InvMaskCircle, InvMaskRect, InvUserMask,SelKey):
#~ %c_snrMask_stdDevDiffImage is part of the Magnitude mask computation
#~ %function set
#~ %
#~ %INPUTS:
#~ % - img_cur, img_prev are the matrix used for the STD Dev> calculation
#~ % - du and dv are the horiz. and vertical (mm) pixel spacings
#~ % - incl_roiSq_size_mm defines the size of the square defining the roi for the included
#~ % elements
#~ % - excl_roiCyl_radius_mm, and excl_roiCyl_h_mm define the dimension of the cylinder defining the roi for the excluded
#~ % elements
#~ % - sliceTp : the slice type in order to shape the excl. region (eg. 'Type = 0' = circle. for transversal slices or 'Type = 1' = rect)
    nx,ny = img_cur.shape[0],img_cur.shape[1]
    cx = (nx+1)/2.0
    cy = (ny+1)/2.0
    du, dv = VoxelSize[0]*1e3,VoxelSize[1]*1e3

    #%get the min and max values for the incl. ROI
    x_inclRoi_min = np.int(np.ceil(cx - incl_roiSq_size_mm/du/2.0))
    x_inclRoi_max = np.int(np.floor(cx + incl_roiSq_size_mm/du/2.0))
    y_inclRoi_min = np.int(np.ceil(cy - incl_roiSq_size_mm/dv/2.0))
    y_inclRoi_max = np.int(np.floor(cy + incl_roiSq_size_mm/dv/2.0))
    #%extract the sub-Image for the inclusion ROI
    img_cur_inclRoi = img_cur[x_inclRoi_min:x_inclRoi_max, y_inclRoi_min:y_inclRoi_max ]
    img_prev_inclRoi = img_prev[x_inclRoi_min:x_inclRoi_max, y_inclRoi_min:y_inclRoi_max ]

    nxroi,nyroi= img_cur_inclRoi.shape[0],img_cur_inclRoi.shape[1],

    if(SelKey == 'Coronal'):
        #%get the mask for the excl. ROI circle
        C=InvMaskCircle[x_inclRoi_min:x_inclRoi_max, y_inclRoi_min:y_inclRoi_max ]
    elif(SelKey == 'Sagittal'):
        #%get the mask for the excl. ROI rectangle
        C=InvMaskRect[x_inclRoi_min:x_inclRoi_max, y_inclRoi_min:y_inclRoi_max ]
    elif InvUserMask is not None:
        C=InvUserMask[x_inclRoi_min:x_inclRoi_max, y_inclRoi_min:y_inclRoi_max ]
    else:
        C = np.ones((nxroi,nyroi),np.bool)

    vect_img_cur = img_cur_inclRoi[C]
    vect_img_prev = img_prev_inclRoi[C]
    correctionFactor = img_cur.mean()/img_prev.mean()

    img_diff = vect_img_prev*correctionFactor - vect_img_cur


    noiseEstimate = np.std(img_diff)/np.sqrt(2.0)

    return noiseEstimate

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError ("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

# A much faster unwrap between two matrices P and P_prev
def unwrap_fast(P_prev, P, threshold=pi):
    #initialisation
    dP = P - P_prev
    P_unwrapped = np.zeros(np.shape(P),np.float32)
    P_unwrapped[:]=P
    while np.any(np.abs(dP[:])>threshold)==True: #threshold=period
          inds = (dP > threshold)
          P_unwrapped[inds] = P_unwrapped[inds] - 2*threshold
          inds = (dP < -threshold)
          P_unwrapped[inds] = P_unwrapped[inds] + 2*threshold
          dP = P_unwrapped - P_prev
    return P_unwrapped

# Calculate the phase difference between two phase images
def calculate_phase_change(P,P_prev):
    P_unwrapped = unwrap_fast(P_prev,P) #unwraped the phase images
    dP = P_unwrapped - P_prev
    return dP

# Adds temperature change to temperature image (degC)
def add_temperature_change( T, dP, K, noise_mask=None ):
   T2=np.zeros(np.shape(T),np.float32) #initialisation of T2 necessary for having T at each step
   if noise_mask is None:
      T2 = T + dP * K
   else:

      T2[noise_mask]=T[noise_mask]+dP[noise_mask]*K
   return T2

def bdc_pre_map(phases, n, ref,order):
    '''
    Returns the phase correction (normalized) for n-1th and nth images in phases,
    using a line fitting for values over roi (region of interest) at each
    point in time given by the phase{...}.info.DynamicAcquisitionTime
    '''
    x = np.zeros((n-ref+1).np.float32)
    TotalSize=phases[ref]['data'].shape[0]*phases[ref]['data'].shape[1]
    y = np.zeros((n-ref+1,TotalSize),np.float32)

    for subsel in range(n-ref+1):
        phase_i=subsel+ref
        x[subsel] = phases[phase_i]['info']['DynamicAcquisitionTime']
        if subsel == 0:
            y[subsel,:] = RescalePhase(phases[phase_i]['data'].flatten(),phases[phase_i]['info'])
        else:
            y[subsel,:] = unwrap_fast(y[subsel-1,:], RescalePhase(phases[phase_i]['data'].flatten(),phases[phase_i]['info']))

    # Fit line to drift in region of interest
    p = np.polyfit(x, y, order)


    AccumDerivate=np.zeros(TotalSize,np.float32)
    for m in range(len(x)):
        for n in range(p.shape[0]-1):
            PowerLevel=p.shape[0]-n-1
            PowerValue=x[m]**(PowerLevel-1) #derivate
            Derivate=PowerValue*PowerLevel*p[n,:]
            AccumDerivate=AccumDerivate+Derivate
    corrections = -np.reshape(AccumDerivate/len(x),(phases[ref]['data'].shape[0],phases[ref]['data'].shape[1]))
    return corrections

def  bdc_linear_correction_on_line( phases, n, ref, roi_mask,order ):
    '''
    Returns the phase correction (normalized) for n-1th and nth images in phases,
    using a line fitting for values over roi (region of interest) at each
    point in time given by the phase{...}.info.DynamicAcquisitionTime
    '''
    if roi_mask is None or roi_mask ==[]:
        return 0.0
    if (n-ref)>50: #we keep only the last 50 observations for the fitting,that will keep the calculations not so heavy and precise enough
        ref=n-50
    # Build 'polyfit'able matrix from phase images and ROI
    x = np.zeros((n-ref+1),np.float32)
    TotalSize=roi_mask.sum()
    y = np.zeros((n-ref+1,TotalSize),np.float32)

    for subsel in range(n-ref+1):
        phase_i=subsel+ref
        x[subsel] = phases[phase_i]['info']['DynamicAcquisitionTime']
        if subsel == 0:
            y[subsel,:] = RescalePhase(phases[phase_i]['data'][roi_mask].flatten(),phases[phase_i]['info'])
        else:
            y[subsel,:] = unwrap_fast(y[subsel-1,:], RescalePhase(phases[phase_i]['data'][roi_mask].flatten(),phases[phase_i]['info']))

    # Fit line to drift in region of interest

    p = np.polyfit(x, y, order)
    # Compute the drift for each image given by the slope p
    #we send back average slope
    if order==1:
        corrections = -p[0,:].mean()
    else:
        AccumDerivate=np.zeros(TotalSize,np.float32)
        for m in range(len(x)):
            for n in range(p.shape[0]-1):
                PowerLevel=p.shape[0]-n-1
                PowerValue=x[m]**(PowerLevel-1) #derivate
                Derivate=PowerValue*PowerLevel*p[n,:]
                AccumDerivate=AccumDerivate+Derivate
        corrections=-(AccumDerivate/len(x)).mean()
    return corrections


class LUTMotion():
    '''
    This class controls the lookup table used to compensate the motion
    From the navigator data, whcih has been previously filtered and classified
    in phases of exhalation, inhalation and motion less, the position of each slide will be placed
    in a "normalized timeline".
    Each slide occurs at a given fraction of the respiration cycle, we assume that breathing is more or
    less regular, after a given time of learning we should collect enough points that can be used later
    to produce a synthetic reference phase:
    phi(x)=a*x+b
    (Hey et al, Mag Reson in Med 61:1494-1499, 2009)
    '''
    def __init__(self,TemporaryData,SelKey,nSlice,LengthTime=100.0):
        self.TemporaryData=TemporaryData
        self.TemporarySliceData=TemporaryData[SelKey][nSlice]
        self.LengthTime=LengthTime
        self.SelKey=SelKey

        self.LUT=SortedCollection(key=itemgetter(0))
        self.TempLUT=SortedCollection(key=itemgetter(0))
        self.LastEntryNavDisplacement=0
        self.BackwardLength = 20
        self.GlobalCorr={-1:2.0,0:0,1:1}

        self.ThermalLUT={}
        #self.ThermalLUT['MotionLess']=SortedCollection(key=itemgetter(0))
        #~ self.ThermalLUT['Inhalation']=SortedCollection(key=itemgetter(0))
        #~ self.ThermalLUT['Exhalation']=SortedCollection(key=itemgetter(0))
        self.ThermalLUT['All']=SortedCollection(key=itemgetter(0))
        self.count = 0
        self.RelocatePrevEntry=None
        self.SelectedBefore=None
        self.SelectedAfter=None

    def GenerateInterpolatedPopulation(self,TBaseLine,NumberInterpolationSteps):
        #entry=[NormalizedTime,ClassData,LookUpPosition,TimeDisplacement,FilteredDisplacement,RealMagnitude,RealPhase,ImageDisplacement,ImageTime]
        #SubEntry=FilteredDisplacement,RealPhase,ImageDisplacement,RealMagnitude
        Temp={}
        #Temp['MotionLess']=SortedCollection(key=itemgetter(0))
        #Temp['Inhalation']=SortedCollection(key=itemgetter(0))
        #Temp['Exhalation']=SortedCollection(key=itemgetter(0))

        Temp['All']=SortedCollection(key=itemgetter(0))

        Fitting={}

        #logger.info( ' CalculateTemperature : Generating interpolated population' )

        for Entry in self.LUT:
            #SubEntry=[Entry[4],Entry[6],Entry[8],Entry[5]] #FilteredDisplacement,RealPhase,ImageTime,RealMagnitude
            SubEntry=[Entry[0],Entry[6],Entry[8],Entry[5]] #FilteredDisplacement,RealPhase,ImageTime,RealMagnitude
            #~ if Entry[1]==0:
                #~ #Temp['MotionLess'].insert([Entry[4],Entry[6]])
                #~ if Entry[0]<0.5:
                    #~ Temp['Exhalation'].insert(SubEntry)
                #~ else:
                    #~ Temp['Inhalation'].insert(SubEntry)
            #~ el
            assert Entry[1] !=0.0
            #~ if Entry[1]==1:
                #~ if SubEntry[0]>=2.0 or SubEntry[0]<1.0:
                    #~ logger.error('Image at %f was classified as inhalation but it has as normalized time of %f' %(Entry[-1],SubEntry[0]))
                #~ Temp['Inhalation'].insert(SubEntry)
            #~ else Entry[1]==-1:
                #~ Temp['Exhalation'].insert(SubEntry)

            Temp['All'].insert(SubEntry)


        for k in Temp:
            PhaseLearn=np.zeros((len(Temp[k]),Temp[k][0][1].shape[0]*Temp[k][0][1].shape[1]),np.float32)
            MagnitudeLearn=np.zeros((len(Temp[k]),Temp[k][0][3].shape[0]*Temp[k][0][3].shape[1]),np.float32)
            XDisplacement=np.zeros(len(Temp[k]))

            for n in range(len(Temp[k])):
                XDisplacement[n]=Temp[k][n][0]
                #PhaseLearn[n,:]=(Temp[k][n][1]*self.SNRMask).flatten()
                PhaseLearn[n,:]=Temp[k][n][1].flatten()
                MagnitudeLearn[n,:]=Temp[k][n][3].flatten()
            UnwrappedPhaseLearn=np.unwrap(PhaseLearn,axis=0)
            InterpFunctionPhase = inter.interp1d(XDisplacement,UnwrappedPhaseLearn,axis=0,kind='linear')
            InterpFunctionMagnitude = inter.interp1d(XDisplacement,MagnitudeLearn,axis=0,kind='linear')

            Fit,residuals, rank, singular_values, rcond =np.polyfit(XDisplacement,UnwrappedPhaseLearn, 1,full=True)
            LinearFit=[Fit,residuals, rank, singular_values, rcond]
            Fit,residuals, rank, singular_values, rcond =np.polyfit(XDisplacement,UnwrappedPhaseLearn, 2,full=True)
            QuadraticFit=[Fit,residuals, rank, singular_values, rcond]

            Fitting[k]={}
            Fitting[k]['LinearFit']=LinearFit
            Fitting[k]['QuadraticFit']=QuadraticFit


            #NewDisplacement=np.linspace(XDisplacement.min(),XDisplacement.max(),20)
            #NewDisplacement=np.linspace(XDisplacement.min(),XDisplacement.max(),len(Temp[k]))
            NewDisplacement=np.linspace(XDisplacement.min(),XDisplacement.max(),NumberInterpolationSteps)
            logger.info('Interpolating for ' + k + ' from %f to %f ' %(XDisplacement.min(),XDisplacement.max()))
            logger.info(str(XDisplacement))

            NewHighResPhase=InterpFunctionPhase(NewDisplacement)
            NewHighResMagnitude=InterpFunctionMagnitude(NewDisplacement)

            #ReWrappedHighResPhase=unwrap_diff(NewHighResPhase)

            for n in range(len(NewDisplacement)):
                lte=Temp[k].find_le(NewDisplacement[n])
                gte=Temp[k].find_ge(NewDisplacement[n])

                if np.abs(NewDisplacement[n]-lte[0]) < np.abs(NewDisplacement[n]-gte[0]):
                    selTime=lte[2]
                else:
                    selTime=gte[2]

                #FilteredDisplacement,NewHighResPhase,Temperture,ImageDisplacement,selTime,NewHighResMagnitude
                NewEntry=[float(NewDisplacement[n]),
                          NewHighResPhase[n,:].reshape(Temp[k][0][1].shape),
                          np.ones(Temp[k][0][1].shape,np.float32)*TBaseLine,
                          selTime,
                          NewHighResMagnitude[n,:].reshape(Temp[k][0][3].shape),
                          np.ones(Temp[k][0][1].shape,np.float32)]

                self.ThermalLUT[k].insert(NewEntry)


    def RemoveOldEntries(self,TimeWindowtoKeepInLUT):

        self.count+=1
        self.LUT.key= itemgetter(8)
        n=0
        while(1):
            if self.LUT[-1][8]-self.LUT[0][8]>=TimeWindowtoKeepInLUT:
                logger.info( ' CalculateTemperature : Removing old image before creating LUT for thermometry' )
                item=self.LUT[0]
                self.LUT.remove(item)
                n+=1
            else:
                break
        

        self.LUT.key= itemgetter(0)

    def GenerateNonInterpolatedPopulationReunwrapped(self,TBaseLine):
        #take average of displacement of last 2 cycles, this will be


        Temp={}
        #Temp['MotionLess']=SortedCollection(key=itemgetter(0))
        Temp['Inhalation']=SortedCollection(key=itemgetter(0))
        Temp['Exhalation']=SortedCollection(key=itemgetter(0))


        for Entry in self.LUT:
            #remove images older than TimeWindowtoKeepInLUT
            SubEntry=[Entry[4],Entry[6],Entry[8],Entry[5]]

            if Entry[1]==0:
                #Temp['MotionLess'].insert([Entry[4],Entry[6]])
                if Entry[0]<0.5:
                    Temp['Exhalation'].insert(SubEntry)
                else:
                    Temp['Inhalation'].insert(SubEntry)
            elif Entry[1]==1:
                Temp['Inhalation'].insert(SubEntry)
            elif Entry[1]==-1:
                Temp['Exhalation'].insert(SubEntry)
            else:
                raise ValueError('wth?')

        for k in Temp:
            PhaseLearn=np.zeros((len(Temp[k]),Temp[k][0][1].shape[0]*Temp[k][0][1].shape[1]),np.float32)
            MagnitudeLearn=np.zeros((len(Temp[k]),Temp[k][0][3].shape[0]*Temp[k][0][3].shape[1]),np.float32)
            XDisplacement=np.zeros(len(Temp[k]))

            for n in range(len(Temp[k])):
                XDisplacement[n]=Temp[k][n][0]
                #PhaseLearn[n,:]=(Temp[k][n][1]*self.SNRMask).flatten()
                PhaseLearn[n,:]=Temp[k][n][1].flatten()
                MagnitudeLearn[n,:]=Temp[k][n][3].flatten()

            NewDisplacement=XDisplacement

            NewHighResPhase=PhaseLearn
            NewHighResMagnitude=MagnitudeLearn


            for n in range(len(NewDisplacement)):
                NewEntry=[NewDisplacement[n],NewHighResPhase[n,:].reshape(Temp[k][0][1].shape),np.ones(Temp[k][0][1].shape,np.float32)*TBaseLine,Temp[k][n][2],NewHighResMagnitude[n,:].reshape(Temp[k][0][3].shape)]
                self.ThermalLUT[k].insert(NewEntry)



    def GenerateNonInterpolatedPopulation(self,TBaseLine):
        #take average of displacement of last 2 cycles, this will be

        for Entry in self.LUT:
            SubEntry=[Entry[0],Entry[6],np.ones(Entry[6].shape,np.float32)*TBaseLine,Entry[8],Entry[5],np.ones(Entry[6].shape,np.float32) ]

            k='All'
            self.ThermalLUT[k].insert(SubEntry)

    def Learn(self,RealMagnitude,RealPhase,nDynamic,StartReference,TBaseLine,TimeWindowtoKeepInLUT,NumberInterpolationSteps):
        '''
        For coronal acquisitions, the motion makes the  acquistition to be off-plane,
        we are going to fit the phase depending on a linearization of LUT planes acquired
        during previously to the sonication

        '''
        #first we update the window of nav data we keep for the observation
        #~
        #~ while self.LastEntryNavDisplacement < len(self.TemporaryData['NavigatorDisplacement-Time'])-1 and\
            #~ self.TemporaryData['NavigatorDisplacement-Time'][self.LastEntryNavDisplacement]<self.LengthTime:
                #~ self.LastEntryNavDisplacement+=1
                #~
        #try:
        self.LastEntryNavDisplacement=len(self.TemporaryData['NavigatorDisplacement-Time'])-1

        ImageNavigator=self.TemporarySliceData['NavigatorDisplacement']
        if nDynamic!=len(ImageNavigator)-1:
            raise ValueError ('nDynamic is not matching the size of the entries')

        #if nDynamic==0:
        #    self.SNRMask=np.ones(RealPhase.shape,np.bool)



        if nDynamic<2:
            return #we discard first image...

        ImageTime=ImageNavigator[nDynamic][0]
        ImageDisplacement=ImageNavigator[nDynamic][1]
        #we found the index in nav data that is closest in time to image timestamp

        LookUpPosition=self.LastEntryNavDisplacement
        while self.TemporaryData['NavigatorDisplacement-Time'][LookUpPosition]>ImageTime and\
            LookUpPosition>0:
            LookUpPosition-=1
        #~ if self.TemporaryData['NavigatorDisplacement-Time'][LookUpPosition]>=ImageTime and LookUpPosition>0:
            #~ #we'll get as a the closer marker the entry the one from  "from the left"
            #~ LookUpPosition-=1

        LookUpPosition-=1


        logger.debug("Final position %i,chosed amplitudes (filtered and image raw) = %f, %f" % (LookUpPosition,self.TemporaryData['NavigatorDisplacement-SmoothData'][LookUpPosition],ImageDisplacement))


        FilteredDisplacement = self.TemporaryData['NavigatorDisplacement-SmoothData'][LookUpPosition]
        RawDisplacement = self.TemporaryData['NavigatorDisplacement-Data'][LookUpPosition]
        TimeDisplacement=self.TemporaryData['NavigatorDisplacement-Time'][LookUpPosition]
        ClassData=self.TemporaryData['NavigatorDisplacement-ClassData'][LookUpPosition]


        NormalizedTime,Certitude=self.FindNormalizedLocation(LookUpPosition)

        entry=[NormalizedTime,ClassData,LookUpPosition,TimeDisplacement,RawDisplacement,RealMagnitude,RealPhase,ImageDisplacement,ImageTime]


        if ClassData==0 and NormalizedTime<0.5:
           ReClass='Exhalation'
        elif ClassData==0 and NormalizedTime>=0.5:
           ReClass='Inhalation'
        elif ClassData==1:
           ReClass='Inhalation'
        elif ClassData==-1:
           ReClass='Exhalation'
        else:
            raise ValueError('wth?')

        logger.debug("Original image time %f, lookup time %f, classified as %s " % (ImageTime,TimeDisplacement,ReClass))

        if Certitude == False:
            logger.warning("Image Displacement POORLY classified")

        self.LastEntry = entry
        if nDynamic<=StartReference:
    
            self.TempLUT.insert(entry)
        elif nDynamic==StartReference+1:
            self.UpdateTemporayLocations()
            self.RemoveOldEntries(TimeWindowtoKeepInLUT)
            if NumberInterpolationSteps>0:
                self.GenerateInterpolatedPopulation(TBaseLine,NumberInterpolationSteps)
            else:
                self.GenerateNonInterpolatedPopulation(TBaseLine)
            self.LocateClosestThermalEntry(TimeWindowtoKeepInLUT)
        else:
            if self.RelocatePrevEntry is not None:
                logger.info('re-classyifing previous image')
                CorrectedNormalizedTime,CorrectedCertitude=self.FindNormalizedLocation(self.RelocatePrevEntry[0])
                if CorrectedCertitude!=True:
                    logger.error('We should be able to classify last temporary entry')
                else:
                    logger.info('Previous image at %f was classified with %f, now it is %f' % (self.RelocatePrevEntry[3],self.RelocatePrevEntry[2],CorrectedNormalizedTime))
                self.ToUpdateTemp[0]=CorrectedNormalizedTime
                k=self.RelocatePrevEntry[1]
                IfElem=self.ThermalLUT[k].find(self.ToUpdateTemp[0])
                if IfElem is not None:
                    self.ThermalLUT[k].remove(IfElem)
                self.ThermalLUT[k].insert(self.ToUpdateTemp)

            k=self.LocateClosestThermalEntry(TimeWindowtoKeepInLUT)
            if Certitude:
                IfElem=self.ThermalLUT[k].find(self.ToUpdateTemp[0])
                if IfElem is not None:
                    self.ThermalLUT[k].remove(IfElem)
                self.ThermalLUT[k].insert(self.ToUpdateTemp)
                self.RelocatePrevEntry=None
            else:
                self.RelocatePrevEntry=[LookUpPosition,k,NormalizedTime,ImageTime]





    def UpdateTemporayLocations(self):
        if self.TemporaryData['FirstClassificationNavDataDone']==False:
            return #we can't do any classication of the image displacement until the first clasification if the navigator displacement is done.....
        for entryTemp in self.TempLUT:
            TempNormalizedTime,ClassData,LookUpPosition,TimeDisplacement,FilteredDisplacement,RealMagnitude,RealPhase,ImageDisplacement,TimeImage=entryTemp[:]
            NormalizedTime,Certitude=self.FindNormalizedLocation(LookUpPosition)
            if Certitude!=True:
                logger.warning('We should be able to classify this temporary entry')
            #We update classification
            ClassData=self.TemporaryData['NavigatorDisplacement-ClassData'][LookUpPosition]
            entry=[NormalizedTime,ClassData,LookUpPosition,TimeDisplacement,FilteredDisplacement,RealMagnitude,RealPhase,ImageDisplacement,TimeImage]

            ExistAlready=self.LUT.find(NormalizedTime)
            if ExistAlready is not None:
                self.LUT.remove(ExistAlready)
            self.LUT.insert(entry)

        self.TempLUT.release()


    def CalculateSyntheticReferencePhase(self):
        '''
        here we go, once we got a LUT more or less filled up, now we can produce
        a linearized version of the phase at different location,
        the synthetic phase will be the interpolation between the two closest locations
        '''

        if self.EntryAfter[0]==self.LastEntry[0]:
            return self.EntryAfter[6]

        if self.EntryBefore[0]==self.LastEntry[0]:
            return self.EntryBefore[6]

        assert self.EntryBefore[0]!=self.EntryAfter[0]

        Top=self.EntryAfter[0]
        Base =self.EntryBefore[0]
        if Base>self.LastEntry[0]:
            Base-=3.0
        if Top<self.LastEntry[0]:
            Top+=3.0
        DT=Top-Base
        dt=(self.LastEntry[0]-Base)/DT #this is the linear coeff, it will range from 0 to 1
        assert dt>=0.0 and dt<=1.0

        SyntheticPhase=self.EntryBefore[6]+(self.EntryAfter[6]-self.EntryBefore[6])*dt
        return SyntheticPhase


    def CalculateDeltaPhi(self,PhaseData):
        SyntheticPhase=self.CalculateSyntheticReferencePhase()
        DeltaPhi=np.zeros(SyntheticPhase.shape,np.float32)
        DeltaPhi=unwrap_diff(PhaseData-SyntheticPhase)

        #need to test how it changed compared to previous, there may be over accumulated wrapping... this may happen after long time

        return DeltaPhi


    def FindNormalizedLocation(self,LookUpPosition):
        #returns the relative position during the breathing phase (between 0 and 1)
        #it also returns that the estimation is known with certitude (True)
        if self.TemporaryData['FirstClassificationNavDataDone']==False:
            return 0.0, False #we can't do any classication of the image displacement until the first clasification if the navigator displacement is done....

        ClassData=self.TemporaryData['NavigatorDisplacement-ClassData'][LookUpPosition]
        IndexLastChangeOfClass=self.TemporaryData['NavigatorChangeClassIndexes'][-1][0]
        if IndexLastChangeOfClass==0:
            #we do not have yet enough data,
            return 0.0, False
        if IndexLastChangeOfClass>LookUpPosition:
            #this means that LookUpPosition is already in a whole segment
            #lets verify that we are in the good segment
            ElemBefore=  self.TemporaryData['NavigatorChangeClassIndexes'].find_le(LookUpPosition)
            ElemAfter =  self.TemporaryData['NavigatorChangeClassIndexes'].find_gt(LookUpPosition)

            assert ElemBefore is not None and ElemAfter  is not None
            ElemBefore=ElemBefore[0]
            ElemAfter=ElemAfter[0]
            assert self.TemporaryData['NavigatorDisplacement-ClassData'][ElemBefore]==ClassData
            nTemporalSteps=ElemAfter-ElemBefore;
            NormLocation=np.linspace(0,1.0,nTemporalSteps+1)
            Pos = LookUpPosition-ElemBefore
            return float(NormLocation[Pos]+self.GlobalCorr[ClassData]), True #this is the only case where we are sure of the relative position.

        #this mean we need to estimate roughly in the previous phase
        #we verify first that the class of the last jump is the same as the one we are evaluating
        assert ClassData == self.TemporaryData['NavigatorDisplacement-ClassData'][IndexLastChangeOfClass]
        if len(self.TemporaryData['NavigatorChangeClassIndexes'])<4:
            #we do not have yet enough data,
            return 0.0, False
        else:
            PreviousPhase= self.TemporaryData['NavigatorChangeClassIndexes'][-3][0]
            NextJump=self.TemporaryData['NavigatorChangeClassIndexes'][-2][0]
            nTemporalSteps=NextJump-PreviousPhase

            Pos = LookUpPosition-IndexLastChangeOfClass

            if Pos >nTemporalSteps: #this means that in the previous phase there were less time steps than in the current, so for the moment we will assume it is at the edge (1.0), later this will fixed
                logger.info('Last temporal window was %i, and we are %i' %(nTemporalSteps,Pos))
                return float(0.999+self.GlobalCorr[ClassData]), False
            NormLocation=np.linspace(0,1.0,nTemporalSteps+1)
            return float(NormLocation[Pos]+self.GlobalCorr[ClassData]), False

    def LocateClosestThermalEntry(self,TimeWindowtoKeepInLUT):
        #
        NormalizedTime,ClassData,LookUpPosition,TimeDisplacement,FilteredDisplacement,RealMagnitude,RealPhase,ImageDisplacement,ImageTime=self.LastEntry[:]
 
        k='All'
        SelectedDisplacement=NormalizedTime

        ExactEntry= self.ThermalLUT[k].find(SelectedDisplacement)
        if ExactEntry  is not None:
            self.ClosestThermalLocation=ExactEntry
            self.SelectedBefore=ExactEntry
            self.SelectedAfter=ExactEntry
            #SelGroup=k
        else:
            #GroupBefore=k
            #GroupAfter=k
            EntryBefore=self.ThermalLUT[k].find_lt(SelectedDisplacement)
            if EntryBefore is None:
                #EntryBefore=self.ThermalLUT[Comp_k][-1]
                EntryBefore=self.ThermalLUT[k][0]
                #GroupBefore=Comp_k
            EntryAfter=self.ThermalLUT[k].find_gt(SelectedDisplacement)
            if EntryAfter  is None:
                #EntryAfter=self.ThermalLUT[Comp_k][0]
                EntryAfter=self.ThermalLUT[k][-1]
               # GroupAfter=Comp_k
            assert EntryBefore is not None and EntryAfter is not None

            self.SelectedBefore=EntryBefore
            self.SelectedAfter=EntryAfter

            if np.abs(EntryBefore[0]-SelectedDisplacement)<np.abs(EntryAfter[0]-SelectedDisplacement):
                self.ClosestThermalLocation=EntryBefore
            else:
                self.ClosestThermalLocation=EntryAfter
        

        pTime=ImageTime
        while(1):
            DelElem=None
            for xx in self.ThermalLUT[k]:
                if pTime-xx[3]>=TimeWindowtoKeepInLUT:
                    DelElem=xx
                    break
            if DelElem is not None:
                self.ThermalLUT[k].remove(DelElem)
            else:
                break

        self.ToUpdateTemp=[float(SelectedDisplacement),RealPhase,None,ImageTime,RealMagnitude,None]

        return k


    def ComputePhaseDiffAndLastTMap(self):
        NewPhase=self.LastEntry[6]
        DeltaPhi=unwrap_diff(NewPhase-self.ClosestThermalLocation[1])
        RefMagnitude=self.ClosestThermalLocation[4]
        SNRMask=self.ClosestThermalLocation[5]
        return DeltaPhi,self.ClosestThermalLocation[2].copy(),RefMagnitude,SNRMask

    def ComputePhaseDiffAndBeforeAndAfter(self):
        NewPhase=self.LastEntry[6]
        T={'Before':[unwrap_diff(NewPhase-self.SelectedBefore[1]),self.SelectedBefore[2].copy(),self.SelectedBefore[4],self.SelectedBefore[5]],
           'After': [unwrap_diff(NewPhase-self.SelectedAfter[1]), self.SelectedAfter[2].copy(), self.SelectedAfter[4], self.SelectedAfter[5]]}
        return T

    def UpdateTMapThermalLut(self,TMap,SNRMask):
        self.ToUpdateTemp[2]=TMap
        self.ToUpdateTemp[5]=SNRMask


class TemperatureWithMotionCorrection():
    '''
    Main class dealing with MRI thermometry
    '''
    def __init__(self,IMAGES,GlobalTemporaryData,SelKey,nSlice,LengthTime=100.0):
        self.GlobalTemporaryData=GlobalTemporaryData
        self.nSlice=nSlice
        self.SelKey=SelKey
        self.IMAGES=IMAGES
        self.TemporaryData=GlobalTemporaryData[SelKey][nSlice]

        self.LUT = LUTMotion(GlobalTemporaryData,SelKey,nSlice,LengthTime)
        
        self.DoProfile=False
        if self.DoProfile:
            self.pr = cProfile.Profile()

        __LocalDriftObjs={}

    def CalculateTemperature(self,nDynamic,NumberSlicesStack,OBJ_FOR_DATA,OldDenormalize=True):
        Alpha=OBJ_FOR_DATA.Alpha
        Beta=OBJ_FOR_DATA.Beta
        Gamma=OBJ_FOR_DATA.Gamma
        T_tolerance=OBJ_FOR_DATA.T_tolerance
        CorrectionOrder=OBJ_FOR_DATA.CorrectionOrder
        NumberOfAverageForReference=OBJ_FOR_DATA.NumberOfAverageForReference
        StartReference=OBJ_FOR_DATA.StartReference
        TBaseLine=OBJ_FOR_DATA.TBaseLine
        CalculateLargeHistory=OBJ_FOR_DATA.CalculateLargeHistory
        TimeWindowtoKeepInLUT=OBJ_FOR_DATA.TimeWindowtoKeepInLUT
        NumberPointsInterpolateInitialLUT=OBJ_FOR_DATA.NumberPointsInterpolateInitialLUT
        UseMotionCompensation=OBJ_FOR_DATA.UseMotionCompensation
        
        UseUserDriftMask = OBJ_FOR_DATA.UseUserDriftMask
        
        ROIs=parseROIString(OBJ_FOR_DATA.ROIs)
        UserDriftROIs=parseROIString(OBJ_FOR_DATA.UserDriftROIs)

        CircleSizeFORSNRCoronal=OBJ_FOR_DATA.CircleSizeFORSNRCoronal
        RectSizeFORSNRTransverse=OBJ_FOR_DATA.RectSizeFORSNRTransverse
        MaxSizeSNRRegion=OBJ_FOR_DATA.MaxSizeSNRRegion

        UseTCoupleForDrift=OBJ_FOR_DATA.UseTCoupleForDrift


        RequestDose=False

        if self.DoProfile:
            self.pr.enable()

        IMAGES=self.IMAGES
        TemporaryData=self.TemporaryData
        SelKey=self.SelKey
        if nDynamic==-1:
            nDynamic=len(IMAGES['Phase'])-1

        if not( nDynamic==len(IMAGES['Temperature']) or  nDynamic==len(IMAGES['Temperature'])-1):
            raise ValueError("Somehow we got screwed, the current dynamic should be either equal to [length of temperature dynamic] or [length of temperature dynamic]-1")

        if nDynamic+1>len(IMAGES['Temperature']):
            nSlice=0
            TemperatureStack=[]
            IMAGES['Temperature'].append(TemperatureStack)
            DoseStack=[]
            IMAGES['Dose'].append(DoseStack)
        else:
            TemperatureStack=IMAGES['Temperature'][nDynamic]
            nSlice=len(TemperatureStack) #we assume that the slice to process is 1+ the last entry in the temperature satck
            if nSlice >=len(IMAGES['Magnitude'][nDynamic]):
                raise ValueError("how did we get this slice # higher than the maximum possible value")
            DoseStack=IMAGES['Dose'][nDynamic]

        if(nSlice != self.nSlice):
            logger.error(' CalculateTemperature : ' + str(SelKey) +' ' + str(nDynamic) +\
                    ' '+ str(nSlice) + 'nSlice does not match the expected value of self.nSlice (%i)' %(self.nSlice))
            raise ValueError(' CalculateTemperature : ' + str(SelKey) +' ' + str(nDynamic) +\
                    ' '+ str(nSlice) + 'nSlice does not match the expected value of self.nSlice (%i)' %(self.nSlice))

        Magnitude=IMAGES['Magnitude'][nDynamic][nSlice]
        Phase=IMAGES['Phase'][nDynamic][nSlice]

        logger.info( ' CalculateTemperature : ' + str(SelKey) +' ' + str(nDynamic) +\
                ' '+ str(nSlice) + " dynamic levels (magnitude,phase) = " +\
                str(Magnitude['info']['DynamicLevel']) + ' ' +str(Phase['info']['DynamicLevel']))

        BodyTemperature=TBaseLine
        CorrectionTCouple=0.0

        if UseTCoupleForDrift>0 and nDynamic>0:
            if len(self.GlobalTemporaryData['Thermocouple'])>0:
                ThermocoupleData = np.array(self.GlobalTemporaryData['Thermocouple'])
                StVector=-(ThermocoupleData[:,0]-Magnitude['TimeStamp'])
                IndcurTmap=np.argmin(np.abs(StVector))

                while IndcurTmap>0 and StVector[IndcurTmap]<0:
                    IndcurTmap-=1
                if IndcurTmap>0:
                    prevTimeStamp=IMAGES['Magnitude'][nDynamic-1][nSlice]['TimeStamp']
                    StVector=-(ThermocoupleData[:,0]-prevTimeStamp)
                    IndPrevTmap=np.argmin(np.abs(StVector))
                    while IndPrevTmap>0 and StVector[IndPrevTmap]<0:
                        IndPrevTmap-=1
                    if IndPrevTmap!=IndcurTmap:

                        MinIndex=IndPrevTmap-10
                        if MinIndex<0:
                            MinIndex=0
                        subT=ThermocoupleData[MinIndex:IndcurTmap+1,UseTCoupleForDrift]
                        subTime=ThermocoupleData[MinIndex:IndcurTmap+1,0]
                        f2 = UnivariateSpline(subTime, subT,s=1)
                       
                        backImtime=subTime[-1]-(Magnitude['TimeStamp']-prevTimeStamp)

                        CorrectionTCouple=f2(backImtime)

                        if nDynamic>1:
                            BodyTemperature=TBaseLine+(CorrectionTCouple-TemporaryData['TCoupleCorrection'][1])
                            logger.info(' CalculateTemperature : '+ str(SelKey) +'  Updated body temperature = %3.2f' %(BodyTemperature))

                    else:
                        logger.warning('CalculateTemperature: We found the same thermocouple entry for two thermal maps ([%3.2f,%3.2f,%i],[%3.2f,%3.2f,%i])' %\
                                (Magnitude['TimeStamp'],ThermocoupleData[IndcurTmap,0],IndcurTmap,prevTimeStamp,ThermocoupleData[IndPrevTmap,0],IndPrevTmap))




        EchoTime=Magnitude['info']['EchoTime']
        K = calculate_dP_to_dT_constant(EchoTime, Alpha, Beta, Gamma)
        phaseToTemperatureConversionFactor = K

        MagnitudeData= RescaleMagnitude(Magnitude['data'],Magnitude['info'])
        #~ PhaseData= RescalePhase    (Phase['data'],Phase['info'])

        if OldDenormalize:
            PhaseDataOld = Normalize_phase_image_internal_dynamic(Phase['data'])
        else:
            PhaseDataOld = RescalePhase(Phase['data'],Phase['info'])
            #~ logger.warning("RescalePhase:Phase data not scaled between pi and pi!!")
            
        if not np.alltrue(np.abs(PhaseDataOld)<=np.pi):
            logger.warning("Phase data not scaled between pi and pi!!")
        #~ if not np.alltrue(np.abs(PhaseDataInDym)<=np.pi):
            #~ logger.warning("Normalize_phase_image_internal_dynamic:Phase data not scaled between pi and pi!!")

        PhaseData=PhaseDataOld
        #if nDynamic >= NumberDynamicsForPCA:#Use this to superimpose hotspot onto phase images
        #    PhaseData = PhaseData+(simHeat[:,:,nDynamic]-37.0)/phaseToTemperatureConversionFactor

        timediff=  Magnitude['info']['DynamicAcquisitionTime']- IMAGES['Magnitude'][0][0]['info']['DynamicAcquisitionTime']

        TemporaryData['CurPhase']=PhaseData

        cum_dose=np.zeros(MagnitudeData.shape,np.float32)
        snr=np.zeros(MagnitudeData.shape,np.float32)
        integration_time=0;

        if UseMotionCompensation:
            self.LUT.Learn(MagnitudeData,PhaseData,nDynamic,StartReference,TBaseLine,TimeWindowtoKeepInLUT,NumberPointsInterpolateInitialLUT)

        
        if nDynamic ==0:

            MaskAverage=MakeMaskForTemperatureAverage(SelKey,IMAGES,nDynamic,nSlice,ROIs) #we use the 16mm ROI at center
            user_mask=MaskAverage
            InvUserMask= user_mask==False

            while len(IMAGES['TemperatureROIMask'])<(nSlice+1):
                IMAGES['TemperatureROIMask'].append(None)
            IMAGES['TemperatureROIMask'][nSlice]=MaskAverage

            TemporaryData['noise_mask_reference'] =np.ones(MagnitudeData.shape,np.bool)
            TemporaryData['averagedInitialPha']=TemporaryData['CurPhase']
            TemporaryData['noise_mask']=np.ones(MagnitudeData.shape,np.bool)
            TemporaryData['cold_mask']=np.ones(MagnitudeData.shape,np.bool)
            TemporaryData['cumul_drift_phase']=np.zeros(MagnitudeData.shape,np.float32)
            UserDriftMask=None
            if UseUserDriftMask:
                UserDriftMask=MakeMaskForTemperatureAverage(SelKey,IMAGES,nDynamic,nSlice,UserDriftROIs)
            TemporaryData['UserDriftMask']=UserDriftMask

            self.__HistoricDriftObjs=DriftAdaptative2D(SelKey,
                                                NumberOfAverageForReference,
                                                TBaseLine,Magnitude,
                                                CircleSizeFORSNRCoronal,RectSizeFORSNRTransverse,CorrectionOrder,False)
            self.__InstantaneousDriftObjs=DriftAdaptative2D(SelKey,
                                                1,
                                                TBaseLine,Magnitude,
                                                CircleSizeFORSNRCoronal,RectSizeFORSNRTransverse,CorrectionOrder,False)

        else:
            user_mask=IMAGES['TemperatureROIMask'][nSlice]
            InvUserMask=user_mask==False
            MaskAverage=user_mask
            UserDriftMask=TemporaryData['UserDriftMask']

        #THIS IS THE MOST COMMON CASE for thermometry
        if nDynamic !=0 and UseMotionCompensation==False :
            noiseEstimateSelected=stdDevDiffImage(MagnitudeData,TemporaryData['MagnitudeReference'],
                                                            Magnitude['info']['VoxelSize'],
                                                            MaxSizeSNRRegion,
                                                            self.__InstantaneousDriftObjs.InvMaskCircle,
                                                            self.__InstantaneousDriftObjs.InvMaskRect,
                                                            InvUserMask,
                                                            SelKey)

            range_zero_mag = (MagnitudeData <= 0)
            snr=np.zeros(MagnitudeData.shape,)
            snr[range_zero_mag==False]=(noiseEstimateSelected)*abs(phaseToTemperatureConversionFactor)/(MagnitudeData[range_zero_mag==False])
            snrMask = (snr) < T_tolerance
            snrMask[range_zero_mag]=False
            if np.all(snrMask==False):
                logger.warning(' CalculateTemperature :  Noise mask is null!')
            if nDynamic<=StartReference:
                TemporaryData['noise_mask_reference'][snrMask==0] = 0
                phaPreUnwrappedTerm = UnwrapImgToReference(PhaseData,TemporaryData['averagedInitialPha'])
                TemporaryData['averagedInitialPha']=unwrap_diff((TemporaryData['averagedInitialPha']*(nDynamic) + phaPreUnwrappedTerm)/(nDynamic+1))
            else:
                snrMask[TemporaryData['noise_mask_reference']==0]=0
                TemporaryData['noise_mask'] = snrMask



            if nDynamic <= 2+StartReference:
                TemporaryData['prev2_pha']=TemporaryData['averagedInitialPha'].copy()
            else:
                TemporaryData['prev2_pha']=TemporaryData['prev_pha'].copy()

            if nDynamic <= 1+StartReference:
                TemporaryData['prev_pha']=TemporaryData['averagedInitialPha'].copy()
            else:
                TemporaryData['prev_pha']=TemporaryData['PreviousPhase'].copy()

        DriftCorrection=0
        MeanTemperatureNonHeated=TBaseLine
        AvgTemp=TBaseLine
        PeakTemp=TBaseLine
        AvgDose=0.0
        PeakDose=0.0
        StdDevTemperatureNonHeated=0
        MeanDriftTemperature=0


        if nDynamic <=StartReference:
            #before hitting the reference image and the number of averages we send back constant map,
            TempSlice=np.ones(MagnitudeData.shape,np.float32)*TBaseLine
            DisplaceSlice=np.zeros(MagnitudeData.shape,np.float32)
            dphi=np.zeros(MagnitudeData.shape,np.float32)


        else:

            if UseMotionCompensation:
                SelBeforeAfter=self.LUT.ComputePhaseDiffAndBeforeAndAfter()

                CompetetiveAnalysis={}

                for kk in SelBeforeAfter:
                    dphi,TempSlice, RefMagnitude,PrevSNRMAsk=SelBeforeAfter[kk][:]

                    noiseEstimateSelected=stdDevDiffImage(MagnitudeData,RefMagnitude,
                                                                    Magnitude['info']['VoxelSize'],
                                                                    200.0,
                                                                    self.__InstantaneousDriftObjs.InvMaskCircle,
                                                                    self.__InstantaneousDriftObjs.InvMaskRect,
                                                                    InvUserMask,
                                                                    SelKey)
                    range_zero_mag = (MagnitudeData <= 0)
                    snr=np.zeros(MagnitudeData.shape,np.float32)
                    snr[range_zero_mag==False]=(noiseEstimateSelected)*abs(phaseToTemperatureConversionFactor)/(MagnitudeData[range_zero_mag==False])
                    snrMask = (snr) < T_tolerance
                    snrMask[range_zero_mag]=False
                    DisabledPixels=np.logical_xor(PrevSNRMAsk,snrMask)
                    snrMask[DisabledPixels]=False

                    CompetetiveAnalysis[kk]=np.std(dphi[snrMask])

                #we select between the two images showing the smallest variation in phase (in principle the varition would me only by drifting)
                if  CompetetiveAnalysis['Before']<CompetetiveAnalysis['After']:
                    kk='Before'
                else:
                    kk='After'

                dphi,TempSlice, RefMagnitude,PrevSNRMAsk=SelBeforeAfter[kk][:]

                noiseEstimateSelected=stdDevDiffImage(MagnitudeData,RefMagnitude,
                                                                    Magnitude['info']['VoxelSize'],
                                                                    200.0,
                                                                    self.__InstantaneousDriftObjs.InvMaskCircle,
                                                                    self.__InstantaneousDriftObjs.InvMaskRect,
                                                                    InvUserMask,
                                                                    SelKey)
                range_zero_mag = (MagnitudeData <= 0)
                snr=np.zeros(MagnitudeData.shape,np.float32)
                snr[range_zero_mag==False]=(noiseEstimateSelected)*abs(phaseToTemperatureConversionFactor)/(MagnitudeData[range_zero_mag==False])
                snrMask = (snr) < T_tolerance
                snrMask[range_zero_mag]=False

                DisabledPixels=np.logical_xor(PrevSNRMAsk,snrMask)
                PrevTempValues=TempSlice[DisabledPixels]

                TemporaryData['noise_mask'] = snrMask.copy()

                TemperatureChange=dphi[TemporaryData['noise_mask']]*phaseToTemperatureConversionFactor
                TemperatureChange[np.abs(TemperatureChange)>20.0]=0 #noisy pixels at air interfaces make a mess....,
                TempSlice[TemporaryData['noise_mask']]+=TemperatureChange
                TempSlice[DisabledPixels]=TBaseLine

                self.__InstantaneousDriftObjs.update(timediff,TempSlice,TemporaryData['noise_mask'],user_mask,BodyTemperature,UserDriftMask=UserDriftMask)
                TempSlice[TemporaryData['noise_mask']]+=self.__InstantaneousDriftObjs.Inst_Correction_Array[TemporaryData['noise_mask']]
                MaskForMean=self.__InstantaneousDriftObjs.maskSNRcold
                MeanDriftTemperature=self.__InstantaneousDriftObjs.Inst_Correction_Array[TemporaryData['noise_mask']].mean()


                self.LUT.UpdateTMapThermalLut(TempSlice,snrMask)
            else:
                # Standard thermometry
                snrMask_ROI=(TemporaryData['prev_noise_mask'] == False)
                TemporaryData['prev_pha'][snrMask_ROI] = TemporaryData['prev2_pha'][snrMask_ROI]

                Prev_snrMask_ROI=(TemporaryData['noise_mask'] == False)
                TemporaryData['CurPhase'][Prev_snrMask_ROI]=TemporaryData['prev2_pha'][Prev_snrMask_ROI]

                dphi_w =  TemporaryData['CurPhase'] -TemporaryData['prev_pha']
                dphi = unwrap_diff( dphi_w );
                TempSlice=IMAGES['Temperature'][nDynamic-1][nSlice]['data'].copy()

                TemperatureChange=dphi*phaseToTemperatureConversionFactor
                TemperatureChange[np.abs(TemperatureChange)>20.0]=0 #noisy pixels at air interfaces make a mess....,
                TempSlice+=TemperatureChange
                self.__HistoricDriftObjs.update(timediff,TempSlice,TemporaryData['noise_mask'],user_mask,BodyTemperature,UserDriftMask=UserDriftMask)
                TempSlice[TemporaryData['noise_mask']]+=self.__HistoricDriftObjs.Inst_Correction_Array[TemporaryData['noise_mask']]

                MaskForMean=self.__HistoricDriftObjs.maskSNRcold
                MeanDriftTemperature=self.__HistoricDriftObjs.Inst_Correction_Array[TemporaryData['noise_mask']].mean()

            MeanTemperatureNonHeated=TempSlice[MaskForMean].mean()
            StdDevTemperatureNonHeated=TempSlice[MaskForMean].std()
            TemporaryData['cold_mask']=MaskForMean


            #############################
            #DOSE calculation
            if RequestDose:
                cur_dose=np.zeros(MagnitudeData.shape,np.float32)
                integration_time = Magnitude['info']['DynamicAcquisitionTime']-IMAGES['Magnitude'][nDynamic-1][nSlice]['info']['DynamicAcquisitionTime'];
                #high temperature limit: prevent overflow
                T_curr=TempSlice-43
                highT_curr = T_curr > 25
                T_curr[highT_curr] = 25
                T_prev=IMAGES['Temperature'][nDynamic-1][nSlice]['data']-43
                highT_prev = T_prev > 25
                T_prev[highT_prev] = 25

                #case 1
                full_dose_pixels = np.logical_and(T_curr > 0 , T_prev > 0)
                full_dose_pixels  =np.logical_and(full_dose_pixels,T_curr != T_prev)
                cur_dose[full_dose_pixels] = ( 2.0**T_curr[full_dose_pixels] - 2.0**(T_prev[full_dose_pixels]) )* integration_time /( (T_curr[full_dose_pixels] - T_prev[full_dose_pixels])*np.log(2) )
                #case 2
                full_dose_equal_pixels = np.logical_and(T_curr > 0 ,T_prev > 0)
                full_dose_equal_pixels = np.logical_and(full_dose_equal_pixels, T_curr == T_prev)
                cur_dose[full_dose_equal_pixels] = ( 2.0**T_curr[full_dose_equal_pixels] ) * integration_time
                #case 3
                    #>43 part
                curr_dose_pixels = np.logical_and(T_curr > 0 , T_prev < 0)
                true_integration_time_sup43 = T_curr[curr_dose_pixels]/(T_curr[curr_dose_pixels]-T_prev[curr_dose_pixels])*integration_time
                cur_dose[curr_dose_pixels] = ( 2.0**T_curr[curr_dose_pixels] - 1 )*true_integration_time_sup43 / ( T_curr[curr_dose_pixels]*np.log(2) )
                    #<43 part
                true_integration_time_inf43 = (1 - T_curr[curr_dose_pixels]/(T_curr[curr_dose_pixels]-T_prev[curr_dose_pixels]))*integration_time
                cur_dose[curr_dose_pixels] = cur_dose[curr_dose_pixels] + ( 1 - 4.0**T_prev[curr_dose_pixels])*true_integration_time_inf43 / ( -T_prev[curr_dose_pixels]*np.log(4) )
                #case 4
                    #>43 part
                prev_dose_pixels = np.logical_and(T_prev > 0 , T_curr < 0)
                true_integration_time_sup43 = T_prev[prev_dose_pixels]/(T_prev[prev_dose_pixels]-T_curr[prev_dose_pixels])*integration_time
                cur_dose[prev_dose_pixels] =  ( 2.0**T_prev[prev_dose_pixels] - 1 )*true_integration_time_sup43 / ( T_prev[prev_dose_pixels]*np.log(2) )
                    #<43 part
                true_integration_time_inf43 = (1 - T_prev[prev_dose_pixels]/(T_prev[prev_dose_pixels]-T_curr[prev_dose_pixels]))*integration_time
                cur_dose[prev_dose_pixels] =  cur_dose[prev_dose_pixels] + ( 1 - 4.0**T_curr[prev_dose_pixels])*true_integration_time_inf43 / ( -T_curr[prev_dose_pixels]*np.log(4) )

                # case 5 #<43
                full_dose_pixels_inf43 = np.logical_and(T_curr < 0 , T_prev < 0)
                full_dose_pixels_inf43 = np.logical_and(full_dose_pixels_inf43, T_curr != T_prev)
                cur_dose[full_dose_pixels_inf43] = ( 4.0**T_curr[full_dose_pixels_inf43] - 4.0**(T_prev[full_dose_pixels_inf43]) )* \
                                                    integration_time /( (T_curr[full_dose_pixels_inf43] - T_prev[full_dose_pixels_inf43])*np.log(4) )

                # case 6 #<43
                full_dose_equal_pixels_inf43 = np.logical_and(T_curr < 0 , T_prev < 0)
                full_dose_equal_pixels_inf43 = np.logical_and( full_dose_equal_pixels_inf43, T_curr == T_prev)
                cur_dose[full_dose_equal_pixels_inf43] = ( 4.0**T_curr[full_dose_equal_pixels_inf43] ) * integration_time
                #trigger mask
                cur_dose=cur_dose/60

                floatOverflowVal = 1e25
                floatNonOverflowAndSNRMask = IMAGES['Dose'][nDynamic-1][nSlice]['data']<(floatOverflowVal-cur_dose)

                cum_dose = IMAGES['Dose'][nDynamic-1][nSlice]['data'];
                cum_dose[floatNonOverflowAndSNRMask] = cum_dose[floatNonOverflowAndSNRMask] + cur_dose[floatNonOverflowAndSNRMask]

        logger.info( ' CalculateTemperature : ' + str(SelKey) +' ' + str(nDynamic) +\
                    ' '+ str(nSlice) + " Mean Temperature Non Heated Zone " +str(MeanTemperatureNonHeated) +', and drift temp/time =' +str(MeanDriftTemperature))

        MaskAverage=np.logical_and(MaskAverage,TemporaryData['noise_mask']) #need to eliminate voxels with not enough SNR. This is mostly common in catheter devices or using "negative" masks

        if np.any(MaskAverage):
            SelTemp=TempSlice[MaskAverage]   
            AvgTemp=SelTemp.mean()
            PeakTemp=SelTemp.max()
            StdTemp=np.std(SelTemp)
            T90=np.percentile(SelTemp,10,interpolation='nearest');
            T10=np.percentile(SelTemp,90,interpolation='nearest');

        flag_print_stats_for_ROI = 1   # this part was added from MKassinopoulos
        if flag_print_stats_for_ROI==1:
            print('--------------  Dynamic: %d' %nDynamic)
            print(SelTemp)
            print('AvgTemp: %3.1f'  %AvgTemp)
            print('PeakTemp: %3.1f'  %PeakTemp)
            print('StdTemp: %3.1f'  %StdTemp)
            print('T90: %3.1f'  %T90)
            print('T10: %3.1f'  %T10)
            print('---------------------------------------')


            AvgDose=cum_dose[MaskAverage].mean()
            PeakDose=cum_dose[MaskAverage].max()
        else:
            SelTemp=np.nan
            AvgTemp=np.nan
            PeakTemp=np.nan
            StdTemp=np.nan
            T90=np.nan
            T10=np.nan


        if not('AvgTemperature'  in TemporaryData):
            TemporaryData['AvgTemperature']=[]
            TemporaryData['TimeTemperature']=[]
            TemporaryData['PeakTemperature']=[]
            TemporaryData['StdTemperature']=[]
            TemporaryData['MeanTemperatureNonHeated']=[]
            TemporaryData['T90']=[]
            TemporaryData['T10']=[]
            TemporaryData['DriftCorrection']=[]
            TemporaryData['TCoupleCorrection']=[]

        if nDynamic<len(TemporaryData['AvgTemperature']):
            TemporaryData['AvgTemperature'][nDynamic]=AvgTemp
            TemporaryData['PeakTemperature'][nDynamic]=PeakTemp
            TemporaryData['StdTemperature'][nDynamic]=StdTemp
            TemporaryData['TimeTemperature'][nDynamic]=timediff
            TemporaryData['MeanTemperatureNonHeated'][nDynamic]=MeanTemperatureNonHeated
            TemporaryData['T90'][nDynamic]=T90
            TemporaryData['T10'][nDynamic]=T10
            TemporaryData['DriftCorrection'][nDynamic]=MeanDriftTemperature
            TemporaryData['TCoupleCorrection'][nDynamic]=CorrectionTCouple
        elif nDynamic ==len(TemporaryData['AvgTemperature']):
            TemporaryData['AvgTemperature'].append(AvgTemp)
            TemporaryData['PeakTemperature'].append(PeakTemp)
            TemporaryData['StdTemperature'].append(StdTemp)
            TemporaryData['TimeTemperature'].append(timediff)
            TemporaryData['MeanTemperatureNonHeated'].append(MeanTemperatureNonHeated)
            TemporaryData['T90'].append(T90)
            TemporaryData['T10'].append(T10)
            TemporaryData['DriftCorrection'].append(MeanDriftTemperature)
            TemporaryData['TCoupleCorrection'].append(CorrectionTCouple)

        else:
            raise ValueError( SelKey + "How nDynamic is neither == or less than TemporaryData['AvgTemperature']")

        TemporaryData['MaskAverage']=MaskAverage


        TempItem ={'MeanTemperatureNonHeated':MeanTemperatureNonHeated,
                    'StdDevTemperatureNonHeated':StdDevTemperatureNonHeated,
                    'DriftCorrection':DriftCorrection,
                    'TimeTemperature':timediff,
                    'AvgTemperature':AvgTemp,
                    'PeakTemperature':PeakTemp,
                    'StdTemperature':StdTemp,
                    'T90':T90,
                    'T10':T10,
                    'data':TempSlice}
   
        if CalculateLargeHistory:
            TempItem['SNR_Mask']=TemporaryData['noise_mask'].copy()
            TempItem['SNR_ColdMask']=TemporaryData['cold_mask'].copy()
            TempItem['SNR']=snr
            TempItem['dphi']=dphi

        TemperatureStack.append(TempItem)

        if RequestDose:
            DoseItem={ 'AvgDose':AvgDose,
                        'PeakDose':PeakDose,
                        'data':cum_dose}

            DoseStack.append(DoseItem)

        logger.info( ' CalculateTemperature : ' + str(SelKey) +' ' + str(nDynamic) +\
                    ' '+ str(nSlice) + " Avg Dose " +str(AvgDose) + ' with time step= ' + str(integration_time))

        if len(TemperatureStack) > NumberSlicesStack:
            raise ValueError('How did we get more slices in the temperature stack?')

        TemporaryData['MagnitudeReference'] = MagnitudeData
        TemporaryData['PreviousPhase'] = PhaseData.copy()
        TemporaryData['prev_noise_mask']=TemporaryData['noise_mask'].copy()

    

__LocalDriftObjs={}

def smooth(x,window_len=11,window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def NoiseInDrift(TimeVector,TimeWindowLength,DriftOverTime):
    if TimeWindowLength>=len(DriftOverTime):
        return np.zeros(len(DriftOverTime),np.float32), DriftOverTime
    dt=np.diff(TimeVector).mean()

    NDt=np.floor(TimeWindowLength/dt);

    SmoothDrift=smooth(DriftOverTime,NDt)

    SmoothDrift=SmoothDrift[:len(DriftOverTime)]

    return DriftOverTime-SmoothDrift,SmoothDrift

def unwrap_diff( phi ):
	output = phi.copy();

	sup2pi = phi>pi
	output[sup2pi]-= np.ceil( (phi[sup2pi] - np.pi)/(2.0*np.pi) )*2.0*np.pi

	inf2pi = phi<-pi
	output[inf2pi]-= np.floor( (phi[inf2pi] + np.pi)/(2.0*np.pi) )*2.0*np.pi
	return output



def UnwrapImgToReference( phi, refPhi ):

	diffPhi = phi-refPhi
	newPhi = phi.copy()
	sup2pi = diffPhi>np.pi
	newPhi[sup2pi] = phi[sup2pi] - np.ceil ( (diffPhi[sup2pi] - np.pi)/(2*np.pi) )*2*np.pi
	inf2pi = diffPhi<-np.pi
	newPhi[inf2pi] = phi[inf2pi] - np.floor( (diffPhi[inf2pi] + np.pi)/(2*np.pi) )*2*np.pi
	return newPhi


def ConvolveTemperature(InputTemp):
    ConvolveVector=np.ones(4)
    ConvolveVector=ConvolveVector/len(ConvolveVector)
    if len(InputTemp)>len(ConvolveVector):
        SmoothTemperature=np.convolve(InputTemp,ConvolveVector,'same')
        SmoothTemperature[0]=InputTemp[0] #we eliminate the border effect by copying the first and last values of the input source
        SmoothTemperature[-1]=InputTemp[-1]
    else:
        SmoothTemperature=InputTemp
    return SmoothTemperature

def GetPathROI(SelKey,IMAGES,nDynamic,nSlice,ROIs):
    Info=IMAGES['Magnitude'][nDynamic][nSlice]['info']
    SelSliceForROI=Info['SliceNumber']
    Rotation=0.0
    if SelSliceForROI>=len(ROIs):
        return np.array([[np.nan],[np.nan]])
    if ROIs[SelSliceForROI]['Type'] in ['CIRCLE','ELLIPSE','SQUARE','RECTANGLE']:
        CenterX=ROIs[SelSliceForROI]['CenterX']
        CenterY=ROIs[SelSliceForROI]['CenterY']
    if ROIs[SelSliceForROI]['Type'] in ['ELLIPSE','SQUARE','RECTANGLE']:
        Rotation=ROIs[SelSliceForROI]['Rotation']

    if ROIs[SelSliceForROI]['Type']=='CIRCLE':
        PathXY=GetPathForTemperatureAverage(Radius1=ROIs[SelSliceForROI]['Radius'],
            Radius2=ROIs[SelSliceForROI]['Radius'],
            CenterX=CenterX,
            CenterY=CenterY)
    elif ROIs[SelSliceForROI]['Type']=='ELLIPSE':
        PathXY=GetPathForTemperatureAverage(Radius1=ROIs[SelSliceForROI]['RadiusX'],
            Radius2=ROIs[SelSliceForROI]['RadiusY'],
            CenterX=CenterX,
            CenterY=CenterY,
            Rotation=Rotation)
    elif ROIs[SelSliceForROI]['Type']=='SQUARE':
        PathXY=GetRectanglePathForTemperatureAverage(LengthX=ROIs[SelSliceForROI]['Length'],
            LengthY=ROIs[SelSliceForROI]['Length'],
            CenterX=CenterX,
            CenterY=CenterY,
            Rotation=Rotation)

    elif ROIs[SelSliceForROI]['Type']=='RECTANGLE':
        PathXY=GetRectanglePathForTemperatureAverage(LengthX=ROIs[SelSliceForROI]['LengthX'],
            LengthY=ROIs[SelSliceForROI]['LengthY'],
            CenterX=CenterX,
            CenterY=CenterY,
            Rotation=Rotation)
    elif ROIs[SelSliceForROI]['Type']=='USER':
        PathXY=GetUserDefinedPathForTemperatureAverage(PointsX=ROIs[SelSliceForROI]['PointsX'],
            PointsY=ROIs[SelSliceForROI]['PointsY'])
    else:
        raise ValueError('Invalid ROIs '+str(ROIs[SelSliceForROI]))

    return PathXY

def MakeMaskForTemperatureAverage(SelKey,IMAGES,nDynamic,nSlice,ROIs):
    PathXY=GetPathROI(SelKey,IMAGES,nDynamic,nSlice,ROIs)
    Info=IMAGES['Magnitude'][nDynamic][nSlice]['info']
    voxelsize=Info['VoxelSize']
    imgsize=IMAGES['Magnitude'][nDynamic][nSlice]['data'].shape
    xx= (np.linspace(-np.floor(imgsize[0]/2.0+1),np.floor(imgsize[0]/2.0),imgsize[0])*voxelsize[0] )*1e3
    yy= (np.linspace(-np.floor(imgsize[1]/2.0+1),np.floor(imgsize[1]/2.0),imgsize[1])*voxelsize[1] )*1e3
    y,x = np.meshgrid(yy,xx)
    x, y = x.flatten(), y.flatten()
    pointsForMaskDetection = np.vstack((x,y)).T
    selPath = PT.Path(PathXY,closed=True)
    TempMask=selPath.contains_points(pointsForMaskDetection)
    TempMask=TempMask.reshape((imgsize[0],imgsize[1])).T

    SelSliceForROI=Info['SliceNumber']
    if ROIs[SelSliceForROI]['NegativeMask']:
        TempMask=np.logical_not(TempMask)

    return TempMask

def GetPathForTemperatureAverage(Radius1=8.0,Radius2=8.0,CenterX=0.,CenterY=0.,Rotation=0.):
    EllipsePath=np.linspace(0,2*np.pi,50)
    PathXY=np.zeros((len(EllipsePath),2),np.float32)
    PathXY[:,0],PathXY[:,1]=np.cos(EllipsePath)*Radius2,np.sin(EllipsePath)*Radius1
    if Rotation !=0.0 :
        alpha=np.radians(Rotation)
        PathXY=np.dot(np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]]),PathXY.T).T
    PathXY[:,0]+=CenterX
    PathXY[:,1]+=CenterY

    return PathXY

def GetRectanglePathForTemperatureAverage(LengthX=10.,LengthY=10.0,CenterX=0.,CenterY=0.,Rotation=0.):
    PathXY=np.zeros((5,2),np.float32)
    PathXY[0,0],PathXY[0,1]=-LengthX/2.,-LengthY/2.
    PathXY[1,0],PathXY[1,1]=-LengthX/2.,+LengthY/2.
    PathXY[2,0],PathXY[2,1]=+LengthX/2.,+LengthY/2.
    PathXY[3,0],PathXY[3,1]=+LengthX/2.,-LengthY/2.
    PathXY[4,0],PathXY[4,1]=-LengthX/2.,-LengthY/2.
    if Rotation !=0.0 :
        alpha=np.radians(Rotation)
        PathXY=np.dot(np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]]),PathXY.T).T
    PathXY[:,0]+=CenterX
    PathXY[:,1]+=CenterY
    return PathXY

def GetUserDefinedPathForTemperatureAverage(PointsX=[0.],PointsY=[0.]):
    PathXY=np.zeros((len(PointsX),2),np.float32)
    PathXY[:,0]+=PointsX
    PathXY[:,1]+=PointsY
    return PathXY

##################
class EntryProcessing:
    '''
    This class organizes the thermometry library processors following the structure of MRI data followed in Proteus
    '''
    def __init__(self,IMAGES,TemporaryData,
                NavigatorData,
                ImagesKeyOrder,
                IncreaseCounterImageProcFunc,
                MaxSlicesPerDynamicFunc,
                GetStackFromSliceNumberFunc,
                NumberSlicesStackFunc,
                ReleaseOnlyNavigatorFunc):
        self.TotalImages = 0
        self.ImagingFields= None
        self.TProcessor={}
        self.IMAGES = IMAGES
        self.TemporaryData=TemporaryData
        self.NavigatorData=NavigatorData
        self.ImagesKeyOrder=ImagesKeyOrder
        self.IncreaseCounterImageProcFunc=IncreaseCounterImageProcFunc
        self.MaxSlicesPerDynamicFunc=MaxSlicesPerDynamicFunc
        self.GetStackFromSliceNumberFunc=GetStackFromSliceNumberFunc
        self.NumberSlicesStackFunc=NumberSlicesStackFunc
        self.ReleaseOnlyNavigatorFunc=ReleaseOnlyNavigatorFunc

    def ResetNavigatorData(self):
        self.POOL_SIZE=10000
        self.POOL_TIME_NAV=np.zeros(self.POOL_SIZE)
        self.POOL_DATA_NAV=np.zeros(self.POOL_SIZE)
        self.POOL_FILT_DATA_NAV=np.zeros(self.POOL_SIZE)
        self.POOL_MOTIONLESS=np.ones(self.POOL_SIZE)*np.nan
        self.POOL_INHALATION=np.ones(self.POOL_SIZE)*np.nan
        self.POOL_EXHALATION=np.ones(self.POOL_SIZE)*np.nan
        self.POOL_FILT_DATA_CLASS=np.zeros(self.POOL_SIZE)
        self.POOL_DATA_INDEX=0
        self.BottomIndexForFiltering=0
        #self.BackPointsToRefresh=200
        self.BackPointsToRefresh=0

    def ProcessImage(self,NewEntry,FlipImage=True,ImagingFields = None):
        if not (ImagingFields  is None):
            self.ImagingFields = ImagingFields
        Info=NewEntry['info']
        SelSlice=Info['SliceNumber']
        
        DoThermalMap=False

        MaxSlicesPerDynamic = self.MaxSlicesPerDynamicFunc()

        if SelSlice <MaxSlicesPerDynamic:
            IndexKey,SubSlice = self.GetStackFromSliceNumberFunc(SelSlice)

            SelKey=self.ImagesKeyOrder[IndexKey]
            #we timestamp when the image arrive

            nDynamic= Info['DynamicLevel']
            if Info['IsPhaseImage']==True:
                ImageCollection=self.IMAGES[SelKey]['Phase']
            else:
                ImageCollection=self.IMAGES[SelKey]['Magnitude']


            NumberSlicesStack = self.NumberSlicesStackFunc(SelKey)


            if self.TemporaryData[SelKey]==[]:
                for n in range(NumberSlicesStack):
                    self.TemporaryData[SelKey].append({})
                    self.TemporaryData[SelKey][-1]['NavigatorDisplacement']=[]
            if ImageCollection==[]:
                LastStack=[NewEntry]
                ImageCollection.append(LastStack)
            else:
                LastStack=ImageCollection[-1]
                LastImage=LastStack[-1]
                if LastImage['info']['DynamicLevel']==nDynamic:
                    #quite weird, sometimes we received a duplicate???
                    if LastImage['info']['SliceNumber']==Info['SliceNumber']:
                        logger.warning(self.__class__.__name__ + ': quite weird, we received a duplicated for '+ SelKey + str(Info['DynamicLevel']) + str(Info['IsPhaseImage']))
                        logger.warning(self.__class__.__name__ + ': we skip any further processing....')
                        print (LastImage['info'])
                        print (Info)
                        return False,SelKey
                    else:
                        LastStack.append(NewEntry)
                else:
                    LastStack=[NewEntry]
                    ImageCollection.append(LastStack)

            if Info['IsPhaseImage']==False:
                logger.info(self.__class__.__name__ + ': Updating navigator for subslice ' +str(SubSlice) )
                MinTime=self.GetReferenceTime()
                #we append the Navigator array
                self.TemporaryData[SelKey][SubSlice]['NavigatorDisplacement'].append([ NewEntry['TimeStamp']-MinTime,
                                                                                                    self.ConvertSlicePrePulseToNavigatorDisplacement(NewEntry['info']['SlicePrepulseDelay'])])
                alldata = np.array(self.TemporaryData[SelKey][SubSlice]['NavigatorDisplacement'])
                timevector=alldata[:,0].flatten()
                NavData=alldata[:,1].flatten()
                self.TemporaryData[SelKey][SubSlice]['NavigatorDisplacement-Time']=timevector
                self.TemporaryData[SelKey][SubSlice]['NavigatorDisplacement-Data']=NavData

            DoThermalMap=False

            if Info['IsPhaseImage']==True:
                ComplementaryImageType='Magnitude'
            else:
                ComplementaryImageType='Phase'

            ImageCollectionComplementary=self.IMAGES[SelKey][ComplementaryImageType]
            if ImageCollectionComplementary==[]:
                LastStackComplementary=[]
            else:
                LastStackComplementary=ImageCollectionComplementary[-1]


            if len(self.IMAGES[SelKey][ComplementaryImageType])>0:
                if self.IMAGES[SelKey][ComplementaryImageType][-1][-1]['info']['DynamicLevel']==nDynamic and \
                    self.IMAGES[SelKey][ComplementaryImageType][-1][-1]['info']['SliceNumber']==SelSlice:
                    DoThermalMap=True
                    #we handle an special case where it may be we received uncompleted images (either Phase or Image a beginning of acquisition that may be caused by the
                    # pausing of the scan, at leas when using the simulator, it may not happen with the real scanner

                    if len(self.IMAGES[SelKey]['Phase'])+1==len(self.IMAGES[SelKey]['Magnitude']):
                        logger.warning(self.__class__.__name__ + ': deleting orphan Magnitude image in '+ SelKey)
                        Orphan=self.IMAGES[SelKey]['Magnitude'][-2]
                        self.IMAGES[SelKey]['Magnitude'].remove(Orphan)
                    elif len(self.IMAGES[SelKey]['Phase'])==len(self.IMAGES[SelKey]['Magnitude'])+1:
                        logger.warning(self.__class__.__name__ + ': deleting orphan Phase image in '+ SelKey)
                        Orphan=self.IMAGES[SelKey]['Phase'][-2]
                        self.IMAGES[SelKey]['Phase'].remove(Orphan)

                    if len(self.IMAGES[SelKey]['Phase'])!=len(self.IMAGES[SelKey]['Magnitude']):
                        raise ValueError('We have serious issues, the lists are not anymore consistent')

            if DoThermalMap:
                self.CalculateTemperature(SelKey,-1,len(LastStack)-1)
    
        self.IncreaseCounterImageProcFunc()

        return DoThermalMap,SelKey

    def ProcessNavigator(self, NewEntry,ImagingFields = None):
        if not (ImagingFields  is None):
            self.ImagingFields = ImagingFields
        if len(self.NavigatorData)==0:
            self.TemporaryData['NavigatorChangeClassIndexes']=SortedCollection(key=itemgetter(0))
            self.TemporaryData['NavigatorChangeClassIndexes'].insert([0]) #the first entry is considered a change of class
            self.TemporaryData['FirstClassificationNavDataDone'] = False
        self.NavigatorData.append(NewEntry)
        self.TemporaryData['NavigatorDisplacement'].append([NewEntry['TimeStamp'],NewEntry['data']['NavigatorDisplacement']])
        MinTime = self.GetReferenceTime()
        self.POOL_DATA_INDEX=len(self.TemporaryData['NavigatorDisplacement'])-1

        if self.POOL_DATA_INDEX>=len(self.POOL_TIME_NAV):
            self.POOL_TIME_NAV=np.append(self.POOL_TIME_NAV,np.zeros(self.POOL_SIZE))
            self.POOL_DATA_NAV=np.append(self.POOL_DATA_NAV,np.zeros(self.POOL_SIZE))
            self.POOL_FILT_DATA_NAV=np.append(self.POOL_FILT_DATA_NAV,np.zeros(self.POOL_SIZE))
            self.POOL_FILT_DATA_CLASS=np.append(self.POOL_FILT_DATA_CLASS,np.zeros(self.POOL_SIZE))
            self.POOL_MOTIONLESS=np.append(self.POOL_MOTIONLESS,np.ones(self.POOL_SIZE)*np.nan)
            self.POOL_INHALATION=np.append(self.POOL_INHALATION,np.ones(self.POOL_SIZE)*np.nan)
            self.POOL_EXHALATION=np.append(self.POOL_EXHALATION,np.ones(self.POOL_SIZE)*np.nan)

        self.POOL_TIME_NAV[self.POOL_DATA_INDEX]=NewEntry['TimeStamp']-MinTime
        self.POOL_DATA_NAV[self.POOL_DATA_INDEX]=NewEntry['data']['NavigatorDisplacement']

        #These matrices will be used later for the motion compensation
        self.TemporaryData['NavigatorDisplacement-Time']=self.POOL_TIME_NAV[0:self.POOL_DATA_INDEX+1]
        self.TemporaryData['NavigatorDisplacement-TimeFilt']=self.POOL_TIME_NAV[0:self.POOL_DATA_INDEX+1]

        #we update the bottom index for the classificator
        if (self.POOL_TIME_NAV[self.POOL_DATA_INDEX ]-self.POOL_TIME_NAV[self.BottomIndexForFiltering]) >= self.ImagingFields.TimeWindowForFiltering:
            self.BottomIndexForFiltering+=1

        SizeDataForFilter =self.POOL_DATA_INDEX-self.BottomIndexForFiltering+1


        self.TemporaryData['NavigatorDisplacement-Data']=self.POOL_DATA_NAV[0:self.POOL_DATA_INDEX+1]
        #we'll filter the signal once we have collected a few seconds,
        #there is an inherent error in the acq rate since, the timestamp used to estimate the rate is based on
        #the computer clock

        if self.TemporaryData['FilterForNavigator'] != []:
            #filter already calculated
            #we only use the last 1000 values
            # we can later tune this, but so far we can process more than 3000 nav entries per second
            if self.POOL_DATA_INDEX+1<SizeDataForFilter:
                nP=-(self.POOL_DATA_INDEX+1)
            else:
                nP=-SizeDataForFilter

            #filtdata=signal.filtfilt(self.TemporaryData['FilterForNavigator'][0],
            #                         self.TemporaryData['FilterForNavigator'][1],self.TemporaryData['NavigatorDisplacement-Data'][nP:])
            ##we estimate the last points using the arburg method, like this we reduce the effect of the noise on the last points
            #subdata=self.__PredictValue(filtdata)
            #self.POOL_FILT_DATA_NAV[self.POOL_DATA_INDEX]=subdata[-1]
            #Feb 3, 2014, this little change may smooth a little more, we'll see how affects classification

            cv=np.ones((15))/15.0

            subdata=self.TemporaryData['NavigatorDisplacement-Data'][nP:]

            filtdata=np.convolve(subdata,cv,mode='valid')

            self.POOL_FILT_DATA_NAV[self.POOL_DATA_INDEX-self.BackPointsToRefresh:self.POOL_DATA_INDEX+1]=subdata[-self.BackPointsToRefresh-1:]

            dataderiv=np.diff(filtdata[-25:]) #we just need the last derivate

            if self.POOL_FILT_DATA_CLASS[self.TemporaryData['NavigatorChangeClassIndexes'][-1][0]]==-1:
                IndChange= -2
            else:
                IndChange= -1

            while self.POOL_FILT_DATA_CLASS[self.TemporaryData['NavigatorChangeClassIndexes'][IndChange][0]]!=-1: #and abs(IndChange)<len(self.TemporaryData['NavigatorChangeClassIndexes']):
                IndChange-=1
            BegIndex=(self.POOL_DATA_INDEX-(self.TemporaryData['NavigatorChangeClassIndexes'][IndChange][0]-1))


            #RelVal=(subdata[-1]-subdata.min())/(subdata.max()-subdata.min())
            RelVal=(subdata[-1]-subdata[-BegIndex:].min())/(subdata[-BegIndex:].max()-subdata[-BegIndex:].min())

            #~ if RelVal <=self.ImagingFields.AmplitudeCriteriaForRestMotion/100.0: #motionless
                #~ self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX]=0
            #~ el
            if dataderiv[-1]>0 :
                self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX]=1 #innhalation
            elif dataderiv[-1]<0 :
                self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX]=-1 #exhalation
            else: # dataderiv==0: #I'd be surprised, but just in case we use the last classification
                self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX]=self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX-1]

            ################ Feb 3, 2014

            if self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX]!=self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX-1] and\
                    (self.POOL_DATA_INDEX-self.TemporaryData['NavigatorChangeClassIndexes'][-1][0])>15:
                self.TemporaryData['NavigatorChangeClassIndexes'].insert([self.POOL_DATA_INDEX])
            else:
                self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX]=self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX-1]

            SubClass=self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX-self.BackPointsToRefresh:self.POOL_DATA_INDEX+1]
            LastSubData=subdata[-self.BackPointsToRefresh-1:]
            assert len(LastSubData)==len(SubClass)

            self.POOL_MOTIONLESS[self.POOL_DATA_INDEX-self.BackPointsToRefresh:self.POOL_DATA_INDEX+1][SubClass==0]=LastSubData[SubClass==0]
            self.POOL_INHALATION[self.POOL_DATA_INDEX-self.BackPointsToRefresh:self.POOL_DATA_INDEX+1][SubClass==1]=LastSubData[SubClass==1]
            self.POOL_EXHALATION[self.POOL_DATA_INDEX-self.BackPointsToRefresh:self.POOL_DATA_INDEX+1][SubClass==-1]=LastSubData[SubClass==-1]

            assert self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX]!=0
            #~ for n in range(-10,0):
                #~ nin=self.TemporaryData['NavigatorChangeClassIndexes'][n][0]
                #~ if self.POOL_FILT_DATA_CLASS[nin-1]==0:
                    #~ self.POOL_MOTIONLESS[nin]=self.POOL_FILT_DATA_NAV[nin]
                #~ elif self.POOL_FILT_DATA_CLASS[nin-1]==+1:
                    #~ self.POOL_INHALATION[nin]=self.POOL_FILT_DATA_NAV[nin]
                #~ else:
                    #~ self.POOL_EXHALATION[nin]=self.POOL_FILT_DATA_NAV[nin]
        elif  self.TemporaryData['NavigatorDisplacement-Time'][-1]>=self.ImagingFields.TimeBeforeFilterNavigator:
            #if we reach the condition, then we calculate the filter coefficients
            #b,a=self.NavigatorFilterCoeffs(self.TemporaryData['NavigatorDisplacement-Time'],self.ImagingFields.FrequencyCut)
            #self.TemporaryData['FilterForNavigator']=[b,a]

            self.TemporaryData['FilterForNavigator']=[5,3]

            #~ subdata=signal.filtfilt(self.TemporaryData['FilterForNavigator'][0],
                                        #~ self.TemporaryData['FilterForNavigator'][1],self.TemporaryData['NavigatorDisplacement-Data'])
                                        #~
            #~ ##we estimate the last points using the arburg method, like this we reduce the effect of the noise on the last points
            #~ subdata=self.__PredictValue(subdata)

            subdata=self.TemporaryData['NavigatorDisplacement-Data']


            cv=np.ones((15))/15.0
            subdataR=np.convolve(subdata,cv,mode='same')

            self.POOL_FILT_DATA_NAV[:self.POOL_DATA_INDEX+1]=subdataR.copy()

            subdataR=subdataR-subdataR.min()
            subdataR=subdataR/subdataR.max() #we normalize between 0 and 1
            dataderiv=np.zeros(subdataR.shape)
            dataderiv[1:]=np.diff(subdataR)
            dataderiv[0]=dataderiv[1]#we copy the derivate of the second term
            dataderiv[-7:]=dataderiv[-8:-1]

            classdata=np.zeros(subdata.shape)
            #~ classdata[np.logical_and(subdataR>self.ImagingFields.AmplitudeCriteriaForRestMotion/100.0,dataderiv>0)]=1 #innhalation
            #~ classdata[np.logical_and(subdataR>self.ImagingFields.AmplitudeCriteriaForRestMotion/100.0,dataderiv<0)]=-1 #exhalation
            for tt in range(len(dataderiv)):
                if dataderiv[tt]==0:
                    dataderiv[tt]=dataderiv[tt-1]
            classdata[dataderiv>0]=1 #innhalation
            classdata[dataderiv<0]=-1 #exhalation

            self.POOL_FILT_DATA_CLASS[self.POOL_DATA_INDEX+1-len(classdata):self.POOL_DATA_INDEX+1]=classdata
            assert np.all(classdata!=0)
            self.POOL_MOTIONLESS[self.POOL_DATA_INDEX+1-len(classdata):self.POOL_DATA_INDEX+1][classdata==0] =subdata[classdata==0]
            self.POOL_EXHALATION[self.POOL_DATA_INDEX+1-len(classdata):self.POOL_DATA_INDEX+1][classdata==-1]=subdata[classdata==-1]
            self.POOL_INHALATION[self.POOL_DATA_INDEX+1-len(classdata):self.POOL_DATA_INDEX+1][classdata==1] =subdata[classdata==1]


            #we do a quick scan to detect the changes of classes
            for n in range(1,self.POOL_DATA_INDEX+1):
                if self.POOL_FILT_DATA_CLASS[n]!=self.POOL_FILT_DATA_CLASS[n-1]:
                    self.TemporaryData['NavigatorChangeClassIndexes'].insert([n])
                if self.POOL_FILT_DATA_CLASS[n-1]==0:
                    self.POOL_MOTIONLESS[n]=subdata[n]
                elif self.POOL_FILT_DATA_CLASS[n-1]==+1:
                    self.POOL_INHALATION[n]=subdata[n]
                else:
                    self.POOL_EXHALATION[n]=subdata[n]
            self.TemporaryData['FirstClassificationNavDataDone']=True
            logger.info('FirstClassificationNavDataDone DONE')
        else:
            #before reaching the condition we just copy the unfiltered data
            self.POOL_FILT_DATA_NAV[self.POOL_DATA_INDEX]=self.POOL_DATA_NAV[self.POOL_DATA_INDEX]

        self.TemporaryData['NavigatorDisplacement-SmoothData']=self.POOL_FILT_DATA_NAV[:self.POOL_DATA_INDEX+1]
        self.TemporaryData['NavigatorDisplacement-ClassData']=self.POOL_FILT_DATA_CLASS[:self.POOL_DATA_INDEX+1]
        self.TemporaryData['NavigatorDisplacement-MotionLess']=self.POOL_MOTIONLESS[:self.POOL_DATA_INDEX+1]
        self.TemporaryData['NavigatorDisplacement-Exhalation']=self.POOL_EXHALATION[:self.POOL_DATA_INDEX+1]
        self.TemporaryData['NavigatorDisplacement-Inhalation']=self.POOL_INHALATION[:self.POOL_DATA_INDEX+1]

        self.IncreaseCounterImageProcFunc()




    def GetReferenceTime(self):
        '''
        We scan the first timestamp event in the data collection to be used as the
        baseline for the displaying, in most of cases, either the navigator or the temperature
        data will arrive first than anything else, so the first event will be used to adjust the display of
        timeline everywhere
        '''
        MinTime= np.inf
        for k in self.IMAGES:
            for k2 in ['Magnitude','Phase']:
                if len(self.IMAGES[k][k2])>0:
                    FirstStack=self.IMAGES[k][k2][0]
                    FirstImage=FirstStack[0]
                    if MinTime>FirstImage['TimeStamp']:
                        MinTime=FirstImage['TimeStamp']
        if len(self.NavigatorData)>0:
            if MinTime>self.NavigatorData[0]['TimeStamp']:
                MinTime=self.NavigatorData[0]['TimeStamp']
        return MinTime


    def __PredictValue(self,FiltData):
        N=self.ImagingFields.OrderForPredictor
        #the last points in the filtered signal are still too close to the
        #original noisy signal, we'll predict those points
        #and the last point will be the effective filtered point used to be
        #stored
        P=len(FiltData)
        M=P-self.ImagingFields.DiscardedPointsInPredictor
        subdata=FiltData[:M]
        Aa,d1,d2=self.arburg(subdata,N)
        inP1=np.concatenate(([0],Aa[1:]))
        zi=signal.lfiltic(-inP1, 1, subdata)
        dum,zf = signal.lfilter(-inP1, 1, subdata,zi=abs(zi));
        res= signal.lfilter([0, 0], -Aa, np.zeros(P-M), zi=zf);
        NewData=FiltData.copy()
        NewData[M:]=res[0]
        return NewData
    
    def arburg(self,x,p):
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
        

    def NavigatorFilterCoeffs(self,timevector,FrequencyCut):
        fRate=1.0/np.diff(timevector).mean()
        Nyquist=fRate/2.0
        NormCut=FrequencyCut/Nyquist
        b, a = signal.butter(8, NormCut)
        return b,a

    def ConvertSlicePrePulseToNavigatorDisplacement(self,SlicePrePulse):
        '''
        Converts the data in the SlicePrePulse field of the metadata in the image header to Navigator Displacement
        This is only valid when using the "liver" patch in the scanner to hack this field
        Formula was given by Max Kohler from Philips
        '''
        return (SlicePrePulse/100.0-150.0)*1000.0

    def CalculateTemperature(self,SelKey,nDynamic,nSlice):
        #we can now be sure that all navigator data is being processed as close as possible when we processed the next temperature map
        self.ReleaseOnlyNavigatorFunc()

        logger.info(self.__class__.__name__ + ':' + SelKey + ' ' + str(nDynamic) + ' ' + str(nSlice) );
        NumberSlicesStack = self.NumberSlicesStackFunc(SelKey)

        if len(self.IMAGES[SelKey]['Phase'])==1 and nSlice==0:
            self.TProcessor[SelKey]=[]
            for n in range (NumberSlicesStack):
                self.TProcessor[SelKey].append(TemperatureWithMotionCorrection(self.IMAGES[SelKey],
                                                                                                 self.TemporaryData,
                                                                                                 SelKey,n,100.0))

        self.TProcessor[SelKey][nSlice].CalculateTemperature(nDynamic,NumberSlicesStack, self.ImagingFields)
        
