# -*- coding: utf-8 -*-
"""
Unittest and template to perform offline thermometry processing for debugging and new development
Samuel Pichardo
May 15, 2021
"""
import numpy as np
from Proteus.ThermometryLibrary import ThermometryLib
from Proteus.File_IO.CLIO.BackEndForThermometry import ReadOnlineMRIData
import tables
import logging 

from Proteus.api import logger


def CompareTwoOrderedLists(list1,list2,bPrintResults=False):
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
        
    
class FieldsForImaging:
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
        self.ROIs ='1 C 4'
        self.UserDriftROIs = '1 R 25 12 0 25'
        #old mask settings for drift, better to UserDriftROIs instead
        self.CircleSizeFORSNRCoronal=45.0
        self.RectSizeFORSNRTransverse=110.0
        self.MaxSizeSNRRegion=200.0
        
        self.UseTCoupleForDrift = False #use this if have a setting using thermocouples to minimize excessive drift correction

        #PCA-PDF related compensator
        self.UsePCAPDF = False
        self.UsePCAPolyFit= False
        self.NumberDynamicsForPCA=16
        self.NumberEigenImagesForPCA=16
        self.MaxPDFIter=30
        self.PDFIterTol=0.1
        self.PDFMaskTol=3.0
        self.PolyFitDriftAccuracy=0.01
        self.PolyFitMaxIteration=30
        
        
        #ARFI RELATED parameters
        self.DoARFI = False
        self.GradientRampUp = 0.155
        self.GradientDuration = 1.0
        self.GradientStrength= 30.0
        self.DispUnwrapOffset = 60.0
        self.GradientAlternate = False
        

        self.NumberSlicesCoronal = 1  #Number of slices in coronal stack
        self.T_mask = 38.0 #Lower limit for temperature mask
        
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
            self.MaxSlicesPerDynamicProc =  lambda: 1
            self.GetStackFromSliceNumberFunc =  lambda x: (0,0)
            self.NumberSlicesStackFunc = lambda x: 1
            self.ReleaseOnlyNavigatorProc = lambda: None
            
    
    def ReturnElementsToInitializeprocessor(self):
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
 

def CreateSortedDataForProcessing(OBJ):
    #The two main results to extract for processing are the images and navigator data dictionaries
    IMAGES=OBJ['IMAGES']
    NavigatorData=OBJ['ExtraData']['NavigatorData']
    IMAGES2 = {}
    ALL_ITEMS=[]
    for k in IMAGES:
        IMAGES2[k]={'MaskROI':[None],'SelPointsROI':[None]}
        for k2 in {'MaskROI':[None],'SelPointsROI':[None]}:
            IMAGES2[k][k2] = IMAGES[k][k2]
    for SelKey in IMAGES:
        for StackMag,StackPhase in zip(IMAGES[SelKey]['Magnitude'],IMAGES[SelKey]['Phase']):
            for ImagMag,ImagPhase in zip(StackMag,StackPhase):
                ALL_ITEMS.append(ImagMag)
                ALL_ITEMS.append(ImagPhase)
    ALL_ITEMS.extend(NavigatorData)
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

if __name__ == "__main__":
    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)

    # nice output format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-8s - %(message)s')
    stderr_log_handler.setFormatter(formatter)
    stderr_log_handler.setLevel(logging.INFO)
    logger.info(' INITIALIZING ThermalLibrary')

    ut = UnitTest() #Instantiate a parent class
    #Be sure of adjusting the path to the file
    Experiment=InspectMPSData('C:\\Users\\samme\\Downloads\\DemoDataThermometry.mps')  
    OriginalImagingData=Experiment.GetDataTreatment('treatmentCount_00001')
    ut.ep = ThermometryLib.EntryProcessing(*ut.ReturnElementsToInitializeprocessor()) #Instantiate an entry processor member on the parent class
    ut.ep.ImagingFields = ut.ImagingFields #Instantiate a class full of image processing parameters

    #we assign the exact ImagingFields for thermometry as in the original file
    print('*'*40+'\nCopying Original Imaging Fields...')
    for k in dir(ut.ep.ImagingFields):
        if '_' not in k:
            print('  ',k,OriginalImagingData['ExtraData']['ImagingFields'][k])
            setattr(ut.ep.ImagingFields,k,OriginalImagingData['ExtraData']['ImagingFields'][k])


    newdata = ut.BatchProccessor(OriginalImagingData) #Parent class must posses a method directing the processing of entries
    Res=CompareTwoOrderedLists(newdata,(OriginalImagingData['IMAGES'],OriginalImagingData['ExtraData']['NavigatorData'])) #Compare the results to ensure that
    assert(Res==True) #this should pass

    #Now, we modify baseline parameter
    ut = UnitTest() #Instantiate a parent class
    ut.ep = ThermometryLib.EntryProcessing(*ut.ReturnElementsToInitializeprocessor()) #Instantiate an entry processor member on the parent class
    ut.ep.ImagingFields = ut.ImagingFields #Instantiate a class full of image processing parameters
    for k in dir(ut.ep.ImagingFields):
        if '_' not in k:
            print('  ',k,OriginalImagingData['ExtraData']['ImagingFields'][k])
            setattr(ut.ep.ImagingFields,k,OriginalImagingData['ExtraData']['ImagingFields'][k])
    
    #we change the baseline, as this will be enough to produce different thermal maps
    ut.ImagingFields.TBaseLine=40.
    ut.ep = ThermometryLib.EntryProcessing(*ut.ReturnElementsToInitializeprocessor()) #Instantiate an entry processor member on the parent class
    ut.ep.ImagingFields = ut.ImagingFields #Instantiate a class full of image processing parameters
    newdata = ut.BatchProccessor(OriginalImagingData) #Parent class must posses a method directing the processing of entries
    Res=CompareTwoOrderedLists(newdata,(OriginalImagingData['IMAGES'],OriginalImagingData['ExtraData']['NavigatorData'])) #Compare the results to ensure that
    assert(Res==False) #this should pass


