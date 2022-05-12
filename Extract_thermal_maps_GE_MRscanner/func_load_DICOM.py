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

def LoadDICOMGe(DCMDir='./',NumberSlicesCoronal =1,ManualTimeBetweenImages=5.0,coil_polarity_status=0,
    number_of_reference = 3,Lower_Window_Index=0,Upper_Window_Index = 0, discard_first_imgs = 5):
    '''
    Function to load an MR dataset from GE Scanner.

    '''

    AllFiles=glob.glob(DCMDir+os.sep+'*.dcm')
    AllFiles.sort()  
    del AllFiles[:(discard_first_imgs*3)]
    
    references_files= AllFiles[:(number_of_reference*3)]

    # MK: Not sure if I understand the following section (10 lines)
    if(Lower_Window_Index==0 and Upper_Window_Index==0):
        data_files = AllFiles
    elif(Lower_Window_Index==0 and Upper_Window_Index>0):
        data_files =AllFiles[:Upper_Window_Index]
    elif(Lower_Window_Index>0 and Upper_Window_Index==0):
        data_files = references_files
        data_files.extend(AllFiles[Lower_Window_Index:])
    elif(Lower_Window_Index>0 and Upper_Window_Index>Lower_Window_Index):
        data_files = references_files
        data_files.extend(AllFiles[Lower_Window_Index:Upper_Window_Index])
    
        
    AllFiles = data_files
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
        
    
    #warnings.warn('The DICOMS are missing TriggerTime in their Metadata, \n so there is no automatic way to recover the timing ')

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
                
                
        #Change coil parameters for multi channel coil we set parameter to
        # 1, and forsingle coil we set parameter -1  
        coil_info = fdcm[0x0018, 0x1250]
        #time_triggering = fdcm[0x0018, 0x1060]
        coil_name = coil_info.value
        
        if(coil_name == "HD BodyLower"):
            reverse_coil_param = -1
        elif(coil_name == "HNS CTL345"):
            reverse_coil_param = 1
        elif(coil_name == "GPFLEX"):
            reverse_coil_param = -1
        else:
            reverse_coil_param = -1
            
        for m in range(2):
            if(len(PreSort)>4):
                # print('Trigger set manually:  (line 115 commented')
                # TriggerTime = ManualTimeBetweenImages
                Time_1 = (PreSort[3][0x0018, 0x1060].value)/1000.0
                Time_2 = (PreSort[6][0x0018, 0x1060].value)/1000.0
                TriggerTime = Time_2 - Time_1
                
            else:
                TriggerTime = ManualTimeBetweenImages
            # print(TriggerTime)
            entry={}
            entry['TimeStamp']=nDynamic*TriggerTime
            #float(im.TriggerTime)/1000.0
            Sl={}
            if m==0:
                imdata=(im.pixel_array).astype(np.float32)
                #imdata = gaussian_filter(imdata, sigma=3)
                
            else:
                cdata= (ar.pixel_array).astype(np.float32)+(ai.pixel_array).astype(np.float32) *1j
                if(coil_polarity_status==0):          
                    imdata=np.angle(cdata)
                else:  
                    imdata=-np.angle(cdata)
                #imdata = restoration.denoise_tv_chambolle(imdata, weight=0.1)


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