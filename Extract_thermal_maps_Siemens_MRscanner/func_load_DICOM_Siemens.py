# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:04:32 2022

@author: Michalis
"""

# In[Section 3]:
# ## function to load DICOM


import glob
import os
import pydicom as dicom

import numpy as np
import warnings

def LoadDICOMSiemens(magnitude_dir='./',phase_dir='./',NumberSlicesCoronal =1,ManualTimeBetweenImages=5.0):

    magnitude_list = glob.glob(magnitude_dir +os.sep+'*')
    magnitude_list.sort()
        
    phase_list = glob.glob(phase_dir +os.sep+'*')
    phase_list.sort()
            
    AllFiles=[]
    
    for i in range(len(magnitude_list)):
        AllFiles.append(magnitude_list[i])
        AllFiles.append(phase_list[i])
    
    NumberFiles=len(AllFiles)
    
    MatPosOrientation=np.zeros((4,4))
    MatPosOrientation[3,3]=1
        
    AllImag=[]
    
    pPos = None
    nDynamic=1
    PreSort=[]
    timeSc = []    
    for n in range(NumberFiles):
        fdcm = dicom.read_file(AllFiles[n])
        PreSort.append(fdcm)
        if (n % 2) == 1:
            time_acq = fdcm.AcquisitionTime
            time_acq_secs = float(MRtime_to_seconds(time_acq))           
            timeSc.append(time_acq_secs)

    timeSc = np.array(timeSc)
    timeSc = timeSc-timeSc[0]  
    x1 = timeSc[:-1]
    x2 = timeSc[1:]
    dx = x2-x1  
    print('Time intervals between images: ' + str(dx))
    Ts = np.mean(dx)


    warnings.warn('The DICOMS are missing TriggerTime in their Metadata, \n so there is no automatic way to recover the timing ')
    
    for icounter in range(len(PreSort)):
        im = PreSort[icounter]
    
        if (icounter>0 and (icounter%2)==0):
            nDynamic+=1
            
        print(nDynamic)
        
        imdata=(im.pixel_array).astype(np.float32)
       
        entry={}
        # entry['TimeStamp']=nDynamic*Ts
        entry['TimeStamp'] = timeSc[nDynamic-1]
        #float(im.TriggerTime)/1000.0
        Sl={}
    
    
        Sl['VoxelSize']=np.zeros(3)
        Sl['VoxelSize'][0:2]=np.array(im.PixelSpacing)/1e3
        Sl['VoxelSize'][2]=float(im.SliceThickness)/1e3
        Sl['DynamicLevel']=nDynamic
        Sl['EchoTime']=float(im.EchoTime)/1e3
        Sl['DynamicAcquisitionTime']=entry['TimeStamp']
        # Sl['DynamicAcquisitionTime'] = Ts
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
        Sl['IsPhaseImage']=((icounter%2)==1)
    
    
        NewEntry={'TimeStamp':entry['TimeStamp'],'info':Sl,'data':imdata}
        AllImag.append(NewEntry)
    
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
    
# magnitude_dir ="C:/Users/andre/Documents/ULTRASOUND/Medical Images/Germany Oncology Center/MRI/04_05_2022/00000001/00000022"
# phase_dir ="C:/Users/andre/Documents/ULTRASOUND/Medical Images/Germany Oncology Center/MRI/04_05_2022/00000001/00000023"

def MRtime_to_seconds(MRtime):     
    MRtime_secs = int(MRtime[:2])*(3600) + int(MRtime[2:4])*60 +  int(MRtime[4:6]) + 0.0001* int(MRtime[7:11])
    return MRtime_secs

