#!/usr/bin/env python
# coding: utf-8

# # Demo of Proteus MR Thermometry
# ```
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# * Neither the name of the University of Calgary nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL University of Calgary, Samuel Pichardo or an of the contributors BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ```
# 
# This notebook illustrates the basic operation to use Proteus MR Thermometry library
# 
# Be aware some of the underlying structure for the processing is aligned how the main Proteus GUI application organizes the data.

# In[Section 1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np
from Proteus.ThermometryLibrary import ThermometryLib
from Proteus.File_IO.H5pySimple import ReadFromH5py, SaveToH5py
import tables
import logging 
from pprint import pprint
from Proteus.ThermometryLibrary.ThermometryLib import  LOGGER_NAME

from skimage import data, img_as_float
from skimage import exposure
import warnings
import cv2

logger = logging.getLogger(LOGGER_NAME)

stderr_log_handler = logging.StreamHandler()
logger.addHandler(stderr_log_handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger.setLevel(logging.ERROR) #use logging.INFO or logging.DEBUG for detailed step information for thermometry

stderr_log_handler.setFormatter(formatter)

logger.info(': INITIALIZING MORPHEUS application, powered by Proteus')


# ## Functions and classess required to prepare the MR processing

# In[Section 2]:


def CompareTwoOrderedLists(list1,list2,bPrintResults=False):
    '''
    Tool function to evaluate two MRI data collections are equivalent
    '''
    #please note NavigatorData will be empty, we kept for completeness purposes
    [IMAGES2,Navigator2]=list2
    [IMAGES,Navigator]=list1
    badimagecount = 0
    badimages = []
    badimageindex = []
    
    notallsame = False


    #Element wise comparison of two sets of results to ensure they match eachother within tolerance
    for Stack in IMAGES:
        for Map in IMAGES[Stack]:
            if Map in ['TimeArrival','SelPointsROI', 'MaskROI' ,'TemperatureROIMask']:
                continue
            for Number in range(len(IMAGES[Stack][Map])):
                for Slice in range(len(IMAGES[Stack][Map][Number])):

                    for Data in IMAGES[Stack][Map][Number][Slice]:
                        if type(IMAGES[Stack][Map][Number][Slice][Data]) is np.ndarray:
                            comparison=np.all(np.isclose(IMAGES[Stack][Map][Number][Slice][Data],
                                                  IMAGES2[Stack][Map][Number][Slice][Data]))
                            if comparison == False:
                                notallsame=True
                                if badimageindex.count(Number) == 0:
                                    badimagecount += 1
                                    badimageindex.append(Number)
                                badimages.append((badimagecount,Stack,Map,Number,Slice,Data))
                        elif type(IMAGES[Stack][Map][Number][Slice][Data]) is dict:
                            for k in IMAGES[Stack][Map][Number][Slice][Data]:
                                v1=IMAGES[Stack][Map][Number][Slice][Data][k]
                                v2=IMAGES2[Stack][Map][Number][Slice][Data][k]
                                if type(v1) is np.ndarray:
                                    comparison=np.all(np.isclose(v1,v2))
                                else:
                                    comparison=v1==v2
                                if comparison == False:
                                    notallsame=True
                                    if badimageindex.count(Number) == 0:
                                        badimagecount += 1
                                        badimageindex.append(Number)
                                    badimages.append((badimagecount,Stack,Map,Number,Slice,Data,k))
                        else:
                            comparison = (IMAGES[Stack][Map][Number][Slice][Data] == IMAGES2[Stack][Map][Number][Slice][Data])
                            if comparison == False:
                                notallsame=True
                                if badimageindex.count(Number) == 0:
                                    badimagecount += 1
                                    badimageindex.append(Number)
                                badimages.append((badimagecount,Stack,Map,Number,Slice,Data))

    

    if bPrintResults:
        if notallsame == True:
            if len(badimages)>0:
                print ('The following images did not match within tolerance')
                for e in badimages:
                    print(e)
        else:
            print('*'*40+'\nDatasets were equivalent')
    
    return notallsame==False        
    
def CreateSortedDataForProcessing(OBJ):
    '''
    The two main results to extract for processing are the images and navigator data dictionaries
    For thermometry processing , we only need to recover magnitude and phase data
    '''
    
    IMAGES=OBJ['IMAGES']
    NavigatorData=OBJ['ExtraData']['NavigatorData']
    IMAGES2 = {}
    ALL_ITEMS=[]
    for k in IMAGES:
        #this helps to initializes some empty data structures
        IMAGES2[k]={'MaskROI':[None],'SelPointsROI':[None]}
        for k2 in {'MaskROI':[None],'SelPointsROI':[None]}:
            IMAGES2[k][k2] = IMAGES[k][k2]
            
    #we reorder the data to mimic how it comes when collecting from a real MRI scanner
    for SelKey in IMAGES:
        for StackMag,StackPhase in zip(IMAGES[SelKey]['Magnitude'],IMAGES[SelKey]['Phase']):
            for ImagMag,ImagPhase in zip(StackMag,StackPhase):
                ALL_ITEMS.append(ImagMag)
                ALL_ITEMS.append(ImagPhase)
    ALL_ITEMS.extend(NavigatorData)
    #the data is organized by time of arrival, to emulate how it works during MRI data collection
    ORDERED_ITEMS = sorted(ALL_ITEMS, key=lambda k: k['TimeStamp'])
    return IMAGES2,ORDERED_ITEMS


class InspectMPSData(object):
    '''
    Minimal class to open MPS files for the re processing
    '''
    def __init__(self,fname):
        print('fname',fname)
        self.fname=fname
        self.ATables=tables.open_file(fname,'r')
        A=self.ATables

        NumberTreatments=A.root.Data.MRIONLINE._g_getnchildren()
        print("Number of treatments ",NumberTreatments)

        for treatment in A.root.Data.MRIONLINE._f_list_nodes():
             print('   '+treatment._v_name)

    def GetDataTreatment(self,iddata):
        node=self.ATables.get_node("/Data/MRIONLINE/" +iddata)
        print(node)
        return ReadOnlineMRIData(node)
    
class FieldsForImaging:
    '''
    Class containing attributes defining the parameters controlling the thermometry
    
    The values can be adjusted to test different parameter conditions
    '''
    def __init__(self):
        self.Alpha = 9.4e-09  #Thermometry temperature coefficient
        self.Beta = 3.0  # Beta Coefficient
        self.Gamma = 42580000.0  #Gyromagnetic ratio
        self.T_tolerance = 12.0  #SNR limit (*C)
        self.CorrectionOrder = 0 #Order of drift correction
        self.NumberOfAverageForReference = 4 #number of dynamics for averaging
        self.StartReference = 4  #dyn. index ref., thermometry is not calculated in dynamics prior to this #
        self.TBaseLine = 37  #Baseline temperature
        self.CalculateLargeHistory = True  #Calculate extra history
        self.UseUserDriftMask = True # use user-specified ROIs to select mask for drift corrector
        self.ROIs ='1 C 4' # string defining ROI mask for monitoring, take a look at \Proteus\Tools\parseROIString.py for details for use
        self.UserDriftROIs = '1 R 25 12 0 25' # string defining ROI mask for drift corrector, take a look at \Proteus\Tools\parseROIString.py for details for use
        #old mask settings for drift, better to UserDriftROIs instead
        self.CircleSizeFORSNRCoronal=45.0
        self.RectSizeFORSNRTransverse=110.0
        self.MaxSizeSNRRegion=200.0
        
        self.UseTCoupleForDrift = False #use this if have a setting using thermocouples to minimize excessive drift correction

        self.NumberSlicesCoronal = 1  #Number of slices in coronal stack
        self.T_mask = 37.0 #Lower limit for temperature mask
        
        #ECHO NAVIGATOR MOTION COMPENSATOR RELATED parameters
        # just kept for completeness as they are now rarely used as we do not have anymore the echonavigator patch
        self.UseMotionCompensation = False  #Use Motion Compensation, keep this FALSE unless you have a dataset with echo navigator
        self.TimeBeforeFilterNavigator = 10.0  #time before filtering (s)
        self.OrderForPredictor = 5  #Order of predictor
        self.DiscardedPointsInPredictor = 100  #Tail points to ignore
        self.AmplitudeCriteriaForRestMotion = 25.0 # ampl. limit for motion-less detection (%)
        self.TimeWindowForClassification = 11  #time window for class. (s)
        self.TimeWindowForFiltering = 100  #time window for filter. (s)
        self.NumberPointsInterpolateInitialLUT = 100  #Number of points for interpolation fir
        self.NumberNavMessagesToWait = 0 #Number of Navigator messages to wait for
        self.TimeWindowtoKeepInLUT = 175.0  #'Length of window (s) of entries to keep in LUT'
        self.FrequencyCut = 0.8  #Frequency cutoff for butterworth filter (Hz)
        

#Empty Main object to preserve the structure required by thermometrylib
class MainObject: pass

class UnitTest:
    def __init__(self):
        #setting up supporting structures required to perform thermometry
        self.ImagingFields=FieldsForImaging()

        self.MainObject = MainObject()
        self.MainObject.TemporaryData = {}
        self.MainObject.TemporaryData['NavigatorDisplacement']=[]
        self.MainObject.TemporaryData['FilterForNavigator']=[]
        self.MainObject.NavigatorData=[]
        self.MainObject.ImagesKeyOrder=['Coronal','Sagittal','User1','User2']
        self.MainObject.IMAGES={}
        for k in self.MainObject.ImagesKeyOrder:
            self.MainObject.IMAGES[k]={'Magnitude':[],'Phase':[],'Temperature':[],'Dose':[],'MaskROI':[None],'SelPointsROI':[None],
                                        'TemperatureROIMask':[None]}
            self.MainObject.TemporaryData[k]=[]
        self.POOL_SIZE=10000
        self.POOL_TIME_NAV=np.zeros(self.POOL_SIZE)
        self.POOL_DATA_NAV=np.zeros(self.POOL_SIZE)
        self.POOL_FILT_DATA_NAV=np.zeros(self.POOL_SIZE)
        self.POOL_MOTIONLESS=np.ones(self.POOL_SIZE)*np.nan
        self.POOL_INHALATION=np.ones(self.POOL_SIZE)*np.nan
        self.POOL_EXHALATION=np.ones(self.POOL_SIZE)*np.nan
        self.POOL_FILT_DATA_CLASS=np.zeros(self.POOL_SIZE)
        self.POOL_DATA_INDEX=0

        self.BackPointsToRefresh=200
        self.TotalImages = 0
        self.BottomIndexForFiltering=0
        self.TProcessor={}
        self.InBackground = False
        self.cback_UpdateTemperatureProfile = lambda x: None
        self.cback_UpdateNavigatorDisplacementProfile = lambda: None
        self.cback_UpdateMRIImage = lambda x,y,z: None
        self.cback_LockMutex = lambda x: None
        self.cback_LockList = lambda x: None
        self.IncreaseCounterImageProc = lambda: None
        self.MaxSlicesPerDynamicProc =  lambda: 1 #THIS IS ONLUY VALID FOR DATA COLLECTIONS WITH 1 slice per dyanmic
        self.GetStackFromSliceNumberFunc =  lambda x: (0,0)
        self.NumberSlicesStackFunc = lambda x: 1 #THIS IS ONLUY VALID FOR DATA COLLECTIONS WITH 1 slice per dyanmic
        self.ReleaseOnlyNavigatorProc = lambda: None
            
    
    def ReturnElementsToInitializeprocessor(self):
        '''
        This function prepares a minimal 
        '''
        MO=self.MainObject
        return [MO.IMAGES,
                MO.TemporaryData,
                MO.NavigatorData,
                MO.ImagesKeyOrder,
                self.IncreaseCounterImageProc,
                self.MaxSlicesPerDynamicProc,
                self.GetStackFromSliceNumberFunc,
                self.NumberSlicesStackFunc,
                self.ReleaseOnlyNavigatorProc]
            
    def BatchProccessor(self, inputdata):
        '''
        This function reprocess all the magnitude and phase data to recreate a new data collection including thermometry data
        '''
        #add input data to parent
        [IMAGES2,self.MainObject.ORDERED_ITEMS] = CreateSortedDataForProcessing(inputdata)
        self.MainObject.MinTime = self.ep.GetReferenceTime()
        #process entries one by one
        for NewEntry in self.MainObject.ORDERED_ITEMS:

            if 'info' in NewEntry:
                self.ep.ProcessImage(NewEntry)
                self.TotalImages+=1

            else:
                self.ep.ProcessNavigator(NewEntry)
                self.TotalImages+=1

        return [self.MainObject.IMAGES,self.MainObject.NavigatorData]
    
    def BatchProccessorFromList(self, ListInputdata):
        '''
        This function reprocess all the magnitude and phase data to recreate a new data collection including thermometry data
        '''
        #add input data to parent
        self.MainObject.ORDERED_ITEMS=ListInputdata
        self.MainObject.MinTime = 0.0
        #process entries one by one
        for NewEntry in self.MainObject.ORDERED_ITEMS:

            if 'info' in NewEntry:
                self.ep.ProcessImage(NewEntry)
                self.TotalImages+=1

            else:
                self.ep.ProcessNavigator(NewEntry)
                self.TotalImages+=1

 


# ## function to load DICOM

# In[Section 3]:


import sys
import glob
import os
import pydicom as dicom
def LoadDICOMGe(DCMDir='./',NumberSlicesCoronal =1,ManualTimeBetweenImages=5.0):
    '''
    Function to load an MR dataset from GE Scanner.

    '''

    AllFiles=glob.glob(DCMDir+os.sep+'*.dcm')
    AllFiles.sort()
    NumberFiles=len(AllFiles)

    #if must be mutliple of number of dynamics x 2 (real,imag)
    if NumberFiles%3 !=0:
        raise ValueError('The number of images must be a multiple of 3')

    #we'll scan and store all the images and verify how many stacks are in function of the different image orientation

    
    MatPosOrientation=np.zeros((4,4))
    MatPosOrientation[3,3]=1

    AllImag=[]
    am=None
    ar=None
    ai=None
    pPos = None
    nDynamic=1
    PreSort=[]
    for n in range(NumberFiles):
        fdcm = dicom.read_file(AllFiles[n])
        PreSort.append(fdcm)
        
    
    warnings.warn('The DICOMS are missing TriggerTime in their Metadata, \n so there is no automatic way to recover the timing ')
    #PreSort.sort(key=lambda fdcm: float(fdcm.TriggerTime))

    #print(PreSort)

    for fdcm in PreSort:
        if fdcm[0x0043, 0x102f].value==0:
            if am is not None:
                raise ValueError('There should not be preloaded magnitude')
            am = fdcm
        elif fdcm[0x0043, 0x102f].value==2:
            if ar is not None:
                raise ValueError('There should not be preloaded real')
            ar = fdcm
        elif fdcm[0x0043, 0x102f].value==3:
            if ai is not None:
                raise ValueError('There should not be preloaded imag')
            ai = fdcm
        else:
            raise ValueError('unhandled image type'  +str(fdcm))

        if am is None or ai is None or ar is None:
            continue
        im = am
        if pPos is not None:
            if pPos==im.ImagePositionPatient:
                nDynamic+=1
        for m in range(2):
            entry={}
            entry['TimeStamp']=nDynamic*ManualTimeBetweenImages
            #float(im.TriggerTime)/1000.0
            Sl={}
            if m==0:
                imdata=(im.pixel_array).astype(np.float32)
            else:
                cdata= (ar.pixel_array).astype(np.float32)+                        (ai.pixel_array).astype(np.float32) *1j
                imdata=-np.angle(cdata) # surface coil
                #imdata=np.angle(cdata) #Body coil

            Sl['VoxelSize']=np.zeros(3)
            Sl['VoxelSize'][0:2]=np.array(im.PixelSpacing)/1e3
            Sl['VoxelSize'][2]=float(im.SliceThickness)/1e3
            Sl['DynamicLevel']=nDynamic
            Sl['EchoTime']=float(im.EchoTime)/1e3
            Sl['DynamicAcquisitionTime']=entry['TimeStamp']
            Sl['ImageOrientationPatient']=np.array(im.ImageOrientationPatient)
            Sl['ImagePositionPatient']=np.array(im.ImagePositionPatient)

            ImagePositionPatient=np.array(im.ImagePositionPatient)
            ImageOrientationPatient=np.array(im.ImageOrientationPatient)
            VoxelSize=np.array(im.PixelSpacing)
            MatPosOrientation=np.zeros((4,4))
            MatPosOrientation[3,3]=1
            MatPosOrientation[0:3,0]=ImageOrientationPatient[0:3]*VoxelSize[0]
            MatPosOrientation[0:3,1]=ImageOrientationPatient[3:]*VoxelSize[1]
            MatPosOrientation[0:3,3]=ImagePositionPatient

            CenterRow=im.Rows/2
            CenterCol=im.Columns/2
            IndCol=np.zeros((4,1))
            IndCol[0,0]=CenterRow
            IndCol[1,0]=CenterCol
            IndCol[2,0]=0
            IndCol[3,0]=1

            CenterImagePosition=np.dot(MatPosOrientation,IndCol)

            Sl['OffcentreAnteriorPosterior']=CenterImagePosition[1,0]
            Sl['OffcentreFeetHead']=CenterImagePosition[2,0]
            Sl['OffcentreRightLeft']=CenterImagePosition[0,0]

            Sl['RescaleSlope']=1.0
            Sl['RescaleIntercept']=0.0
            Sl['ScaleSlope']=1.0
            Sl['ScaleIntercept']=0.0
            Sl['SlicePrepulseDelay']=0
            Sl['IsPhaseImage']=(m!=0)
            NewEntry={'TimeStamp':entry['TimeStamp'],'info':Sl,'data':imdata}
            AllImag.append(NewEntry)
        am=None
        ar=None
        ai=None
        if pPos is None:
            pPos=im.ImagePositionPatient


    #we recreated a pseudo-arrival by ordering the images by timestamp and by type (mag or phase)
    SortedImag=sorted(AllImag, key=lambda d: (d['TimeStamp'],d['info']['IsPhaseImage']))


    ListOfStacks=[]
    FinalList=[]
    for entry in SortedImag:
        Sl=entry['info']['ImageOrientationPatient'].tolist()+entry['info']['ImagePositionPatient'].tolist()
        if Sl not in ListOfStacks:
            ListOfStacks.append(Sl)
    for entry in SortedImag:
        Sl=entry['info']['ImageOrientationPatient'].tolist()+entry['info']['ImagePositionPatient'].tolist()
        entry['info']['SliceNumber']=ListOfStacks.index(Sl)
        FinalList.append(entry)
    return FinalList


# In[Section 4]:


ListDataExample=LoadDICOMGe('MR_Thermometry_Examples\ExampleC\All',ManualTimeBetweenImages=4.0)
print('Total number of images (both magnitude and phase) =',len(ListDataExample))
print('Basic Metatada')
pprint(ListDataExample[0]['info'])


# ----
# We use the UnitTest class for demonstration to reprocess MRI data

# In[Section 5]:


ut = UnitTest() #Instantiate a parent class
ut.ep = ThermometryLib.EntryProcessing(*ut.ReturnElementsToInitializeprocessor()) #Instantiate an entry processor member on the parent class
ut.ep.ImagingFields = ut.ImagingFields #Instantiate a class full of image processing parameters
ut.ep.ImagingFields.Beta=1.5
ut.ep.ImagingFields.ROIs='1 C 5 -8.2 -3.15'
ut.ep.ImagingFields.UserDriftROIs='-1 R 70 25 -8.2 -3.15'
ut.ep.ImagingFields.UseUserDriftMask = True
ut.ep.ImagingFields.T_tolerance =1.5
ut.ep.ImagingFields.StartReference =0
ut.ep.ImagingFields.NumberOfAverageForReference=3



# In[Section 6]:


for k in dir(ut.ep.ImagingFields):
    if '_' not in k:
        print(k,getattr(ut.ep.ImagingFields,k))
        


# We process the magntiude and phase data. We also use the CompareTwoOrderedLists to show that the repreocess thermometry is the same as in the original dataset

# In[Section 7]:


ut.BatchProccessorFromList(ListDataExample) #Parent class must posses a method directing the processing of entries



# In[Section 8A; testing mode; Plot two images overlayed]:

import copy

Main=ut.MainObject

xlim1 = 0
xlim2 = 255
ylim1 = 0
ylim2 = 255

plt.close('all')        

# fig,ax1 = plt.subplots(ncols=1, figsize=[5,5])
plt.figure(figsize=(5,4))


IMAGES = copy.deepcopy(Main.IMAGES)
nDynamic = 20

p2, p98 = np.percentile(IMAGES['Coronal']['Magnitude'][nDynamic][0]['data'], (2, 98))
img_rescale = exposure.rescale_intensity(IMAGES['Coronal']['Magnitude'][nDynamic][0]['data'], in_range=(p2, p98))

statmap = IMAGES['Coronal']['Temperature'][nDynamic][0]['data']

temp_thresh = 37.2
for x in range(1,xlim2):
    for y in range(1,ylim2):
        if statmap[x,y]<temp_thresh:
            statmap[x,y] = nan      
  
plt.close('all')   
imshow(img_rescale,cmap=plt.cm.gray,interpolation='none')
imshow(statmap,vmin=40,vmax=55,cmap=plt.cm.YlOrBr, alpha = 1)

plt.colorbar()
plt.title('Temp map (thresholded)')

    

# In[Section 8A; Plot images]:



def PlotImages(nDynamic, Main,gtitle):
    IMAGES=Main.IMAGES
    plt.figure(figsize=(16,8))
    plt.subplot(2,3,1)
    p2, p98 = np.percentile(IMAGES['Coronal']['Magnitude'][nDynamic][0]['data'], (2, 98))
    img_rescale = exposure.rescale_intensity(IMAGES['Coronal']['Magnitude'][nDynamic][0]['data'], in_range=(p2, p98))
    plt.imshow(img_rescale,cmap=plt.cm.gray)
    plt.title('Magnitude')
    plt.subplot(2,3,2)
    plt.imshow(IMAGES['Coronal']['Phase'][nDynamic][0]['data'],cmap=plt.cm.gray)
    plt.title('Phase')
    plt.subplot(2,3,3)
    plt.imshow(IMAGES['Coronal']['Temperature'][nDynamic][0]['data'],vmin=40,vmax=55,cmap=plt.cm.jet)
    #plt.imshow(IMAGES['Coronal']['Temperature'][nDynamic][0]['data'],vmin=40,vmax=55,cmap=plt.cm.jet)
    #plt.xlim(0,255)
    #plt.ylim(255,0)
    plt.xlim(xlim1,xlim2)
    plt.ylim(ylim2,ylim1)
    plt.colorbar()
    plt.title('Temperature map')
    
    plt.subplot(2,3,4)
    plt.imshow(IMAGES['Coronal']['Temperature'][nDynamic][0]['SNR_Mask'],cmap=plt.cm.gray)
    plt.title('SNR mask')
    plt.subplot(2,3,5)
    plt.imshow(IMAGES['Coronal']['Temperature'][nDynamic][0]['SNR_ColdMask'],cmap=plt.cm.gray)
    plt.title('"cold" SNR mask used for drift correction')
    
    
    
    plt.subplot(2,3,6)  
        
    statmap = copy.deepcopy(IMAGES['Coronal']['Temperature'][nDynamic][0]['data'])
    temp_thresh = 37.2
    for x in range(1,xlim2):
        for y in range(1,ylim2):
            if statmap[x,y]<temp_thresh:
                statmap[x,y] = nan      
            
    imshow(img_rescale,cmap=plt.cm.gray,interpolation='none')
    # imshow(statmap,vmin=40,vmax=55,cmap=plt.cm.YlOrBr, alpha = 1)
    imshow(statmap,vmin=40,vmax=55,cmap=plt.cm.YlOrBr, alpha = 1, animated = True)
    plt.colorbar()
    plt.title('Temp map (thresholded)')    
    
    plt.suptitle(gtitle)
    
plt.close('all')        
    
#PlotImages(0,Main,'Dynamic=0')#
# PlotImages(1,Main,'Dynamic=1')#
#PlotImages(2,Main,'Dynamic=2')#
#PlotImages(3,Main,'Dynamic=3')#
PlotImages(4,Main,'Dynamic=4')#
PlotImages(5,Main,'Dynamic=5')# 
# PlotImages(6,Main,'Dynamic=6')#  
#PlotImages(7,Main,'Dynamic=7')#
PlotImages(8,Main,'Dynamic=8')#
#PlotImages(9,Main,'Dynamic=9')#
PlotImages(10,Main,'Dynamic=10')#
#PlotImages(11,Main,'Dynamic=11')#
# PlotImages(12,Main,'Dynamic=12')#
#PlotImages(13,Main,'Dynamic=13')#
#PlotImages(14,Main,'Dynamic=14')#
# PlotImages(15,Main,'Dynamic=15')#
# PlotImages(16,Main,'Dynamic=16')#
#PlotImages(17,Main,'Dynamic=17')#
#PlotImages(18,Main,'Dynamic=18')#
#PlotImages(19,Main,'Dynamic=19')#
# PlotImages(20,Main,'Dynamic=20')#




# In[Section 8A; print figures through a loop; Plot two images overlayed]:


fig_tmp = plt.figure(figsize=(16,8))
fig_tmp = plt.figure(figsize=(6.38,9.66))

def PlotImages_temp(nDynamic, Main,gtitle):
    IMAGES=Main.IMAGES
    # fig_tmp = plt.figure(figsize=(16,8))
    
    # plt.figure(figsize=(16,8))
    plt.subplot(2,3,1)
    p2, p98 = np.percentile(IMAGES['Coronal']['Magnitude'][nDynamic][0]['data'], (2, 98))
    img_rescale = exposure.rescale_intensity(IMAGES['Coronal']['Magnitude'][nDynamic][0]['data'], in_range=(p2, p98))
    plt.imshow(img_rescale,cmap=plt.cm.gray)
    plt.title('Magnitude')
    plt.subplot(2,3,2)
    plt.imshow(IMAGES['Coronal']['Phase'][nDynamic][0]['data'],cmap=plt.cm.gray)
    plt.title('Phase')
    plt.subplot(2,3,3)
    plt.imshow(IMAGES['Coronal']['Temperature'][nDynamic][0]['data'],vmin=40,vmax=55,cmap=plt.cm.jet)
    #plt.imshow(IMAGES['Coronal']['Temperature'][nDynamic][0]['data'],vmin=40,vmax=55,cmap=plt.cm.jet)
    #plt.xlim(0,255)
    #plt.ylim(255,0)
    plt.xlim(xlim1,xlim2)
    plt.ylim(ylim2,ylim1)
    plt.colorbar()
    plt.title('Temperature map')
    
    plt.subplot(2,3,4)
    plt.imshow(IMAGES['Coronal']['Temperature'][nDynamic][0]['SNR_Mask'],cmap=plt.cm.gray)
    plt.title('SNR mask')
    plt.subplot(2,3,5)
    plt.imshow(IMAGES['Coronal']['Temperature'][nDynamic][0]['SNR_ColdMask'],cmap=plt.cm.gray)
    plt.title('"cold" SNR mask used for drift correction')
    
    
    
    plt.subplot(2,3,6)  
        
    statmap = copy.deepcopy(IMAGES['Coronal']['Temperature'][nDynamic][0]['data'])
    temp_thresh = 37.2
    for x in range(1,xlim2):
        for y in range(1,ylim2):
            if statmap[x,y]<temp_thresh:
                statmap[x,y] = nan      
            
    imshow(img_rescale,cmap=plt.cm.gray,interpolation='none')
    # imshow(statmap,vmin=40,vmax=55,cmap=plt.cm.YlOrBr, alpha = 1)
    imshow(statmap,vmin=40,vmax=55,cmap=plt.cm.YlOrBr, alpha = 1, animated = True)
    plt.colorbar()
    plt.title('Temp map (thresholded)')    
    
    plt.suptitle(gtitle)
    plt.show()



# from celluloid import Camera
import time

ind = [4,5,8,10]
nFrames = 100    # 389
ind = range(0,nFrames)

plt.close('all')  
fig_tmp = plt.figure(figsize=(16,8))
# camera = Camera(fig_tmp)
for i in ind:
    PlotImages_temp(i,Main,'Dynamic=' + str(i))
    print('Dynamic=' + str(i))
    plt.show()

    filename='Frames\image_'+str(i)+'.png'
    plt.savefig(filename, dpi=75)

    # time.sleep(5)
    # camera.snap()

# animation = camera.animate(interval = 1, repeat = False)
# animation.save('Video_temp.mp4', writer = 'FFwriter' )    #  FFMpegWriter , pillow,

# animation.save('basic_animation_1.mp4', fps=1)


import glob
    
img_array = []
ind = range(0,nFrames)
for i in ind: 
    filename='Frames\image_'+str(i)+'.png'    
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# In[Section 8B]:


def PlottempImages(Main):
    IMAGES=Main.IMAGES
    #subinum = 1
    #plt.figure(figsize=(35,35))
    # for i in range(len(IMAGES['Coronal']['Temperature'])):
    for i in range(0,10):
        subi = i
        itteration_num = i +1
        if ((subi %6) == 0 ):
            plt.figure(figsize=(18,10))
            subinum = 1
        plt.subplot(2,3,subinum)
        plt.imshow(IMAGES['Coronal']['Temperature'][i][0]['data'],vmin=40,vmax=60,cmap=plt.cm.jet)
        #plt.xlim(0,255)
        #plt.ylim(255,0)
        plt.xlim(xlim1,xlim2)
        plt.ylim(ylim2,ylim1)
        plt.colorbar()
        plt.title(itteration_num)
        subinum = subinum + 1
        
        
PlottempImages(Main)


# `Main.TemporaryData` has also the temperature profile over time resulting from the thermometry in the user ROI

# In[Section 9]:


def PlotTemporalData(Main):
    timeD=np.array(Main.TemporaryData['Coronal'][0]['TimeTemperature'])
    AvgTemp=np.array(Main.TemporaryData['Coronal'][0]['AvgTemperature'])
    T10=np.array(Main.TemporaryData['Coronal'][0]['T10'])
    T90=np.array(Main.TemporaryData['Coronal'][0]['T90'])
    plt.figure(figsize=(12,6))
    plt.plot(timeD,AvgTemp)
    plt.plot(timeD,T10)
    plt.plot(timeD,T90)
    plt.legend(['Avg. Temperature','T10','T90'])
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (43$^{\circ}$))')

PlotTemporalData(Main)


# In[ test plot function ]: 


def PlotTemporalData(Main):
    timeD=np.array(Main.TemporaryData['Coronal'][0]['TimeTemperature'])
    AvgTemp=np.array(Main.TemporaryData['Coronal'][0]['AvgTemperature'])
    T10=np.array(Main.TemporaryData['Coronal'][0]['T10'])
    T90=np.array(Main.TemporaryData['Coronal'][0]['T90'])
    
    plt.figure(figsize=(12,6))
    plt.plot(timeD,AvgTemp)
    plt.plot(timeD,T10)
    plt.plot(timeD,T90)
    plt.legend(['Avg. Temperature','T10test','T90'])
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (43$^{\circ}$)')

PlotTemporalData(Main)


# In[ ]:

Main_data = Main.TemporaryData['Coronal'][0]
    

timeD=np.array(Main_data['TimeTemperature'])

plt.figure()

plt.plot(timeD)
















