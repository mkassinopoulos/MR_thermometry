#!/usr/bin/env python
# coding: utf-8

# # Based on 'Demo of Proteus MR Thermometry' (Uni of Calgary, Samuel Pichardo)

# In[Section 1]:

#get_ipython().run_line_magic('matplotlib', 'inline')

# import sys
sys.path.append('D:\SOUNDPET\ThermometryLib_2022_03_21')

import matplotlib.pyplot as plt
import numpy as np
import os
import copy ;  import shutil
from Proteus.ThermometryLibrary import ThermometryLib
import logging 
from pprint import pprint
from Proteus.ThermometryLibrary.ThermometryLib import  LOGGER_NAME
from skimage import exposure
import cv2
import addcopyfighandler


# from Proteus.File_IO.H5pySimple import ReadFromH5py, SaveToH5py
# import tables
# from skimage import data, img_as_float

# import warnings


logger = logging.getLogger(LOGGER_NAME)
stderr_log_handler = logging.StreamHandler()
logger.addHandler(stderr_log_handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.ERROR) #use logging.INFO or logging.DEBUG for detailed step information for thermometry
stderr_log_handler.setFormatter(formatter)
logger.info(': INITIALIZING MORPHEUS application, powered by Proteus')


# ## Functions and classess required to prepare the MR processing    
from funcs_for_MR_processing import CompareTwoOrderedLists, UnitTest
from func_load_DICOM_Siemens import LoadDICOMSiemens




# In[Section 2: Give path_input for DICOM]:

# path_session = r'D:\SOUNDPET\ThermometryLib_2022_03_21\MR_Thermometry_Examples\2021_12_17_phantom'
# scan = path_session + r'\6_Surface_coil_200w_120s'
# scan = path_session + r'\5.Body coil 200w 120s axial'

# path_session = r'D:\SOUNDPET\ThermometryLib_2022_03_21\MR_Thermometry_Examples\2021_12_29_meat'
# scan = path_session + r'\2'

plt.close('all')

path_main = r'D:\SOUNDPET\Raw_data\\'
session = '2022_05_04'
scan_label = '5_Coronal_200w_60s_on_60s_off'

scan = path_main + '\\' + session + '\\' + scan_label
path_output = r'D:\SOUNDPET\Analysis\2022_05_05_Extract_video_Siemens\Export'
path_output_scan = path_output + '\\' + session + '\\' + scan_label
path_magn = scan + '\\Magnitude'
path_phase = scan + '\\Phase'


if os.path.isdir(path_output_scan)==0:
    os.makedirs(path_output_scan)

no_of_references = 3
ListDataExample = LoadDICOMSiemens(path_magn,path_phase,NumberSlicesCoronal =1,ManualTimeBetweenImages=5.0)

print('Total number of images (both magnitude and phase) =',len(ListDataExample))
print('Basic Metatada')
pprint(ListDataExample[0]['info'])

# For Example_C. TE=22ms --> K = -12.05  (phase change to temperature change conversion constant)

# In[Section 3: Derive thermometry]:
# We use the UnitTest class for demonstration to reprocess MRI data


ut = UnitTest() #Instantiate a parent class
ut.ep = ThermometryLib.EntryProcessing(*ut.ReturnElementsToInitializeprocessor()) #Instantiate an entry processor member on the parent class
ut.ep.ImagingFields = ut.ImagingFields #Instantiate a class full of image processing parameters

ut.ep.ImagingFields.Beta=3.0   # Siements 3T

xlim1 = 0;   xlim2 = 255
ylim1 = 0;   ylim2 = 255


x1 = 117 ; y1 = 136  ; circle_size = 3
drift_mask_DX = 70  ;   drift_mask_DY = 25
par_tolerance = 3

X1 = round(x1-(xlim2/2))
Y1 = round(y1-(ylim2/2))

ut.ep.ImagingFields.ROIs='1 C %d %d %d'  % (circle_size, X1,Y1)  
ut.ep.ImagingFields.UserDriftROIs='-1 R %d %d %d %d'  % (drift_mask_DX, drift_mask_DY, X1,Y1)
ut.ep.ImagingFields.T_tolerance = par_tolerance

ut.ep.ImagingFields.UseUserDriftMask = True
ut.ep.ImagingFields.StartReference =0
ut.ep.ImagingFields.NumberOfAverageForReference= no_of_references   # This parameter should be defined earlier (when DICOM is loaded)



for k in dir(ut.ep.ImagingFields):
    if '_' not in k:
        print(k,getattr(ut.ep.ImagingFields,k))

# We process the magntiude and phase data. We also use the CompareTwoOrderedLists to show that the repreocess thermometry is the same as in the original dataset

ut.BatchProccessorFromList(ListDataExample) #Parent class must posses a method directing the processing of entries


Main=ut.MainObject
Main_data = Main.TemporaryData['Coronal'][0]
timeD=np.array(Main_data['TimeTemperature'])
Ts = timeD[1]
nFrames = len(timeD)
print('============================ ')
print('No of frames: '+str(nFrames))
print('Acquisition time per slice: %3.2f s '  %Ts)


# In[Section 4; define function plotImages, test with a volume]:


temp_thresh = 37.3
colorbar_upper_limit = 40
# fig_tmp = plt.figure(figsize=(6.38,9.66))

def PlotImages(nDynamic, Main):
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
    plt.imshow(IMAGES['Coronal']['Temperature'][nDynamic][0]['data'],vmin=temp_thresh,vmax=54,cmap=plt.cm.jet)
    #plt.xlim(0,255)
    #plt.ylim(255,0)
    plt.xlim(xlim1,xlim2);    plt.ylim(ylim2,ylim1)
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
    for x in range(1,xlim2+1):
        for y in range(1,ylim2):
            if statmap[x,y]<temp_thresh:
                statmap[x,y] = 'nan'      
            
    plt.imshow(img_rescale,cmap=plt.cm.gray,interpolation='none')
    # imshow(statmap,vmin=40,vmax=55,cmap=plt.cm.YlOrBr, alpha = 1)
    plt.imshow(statmap,vmin=temp_thresh,vmax=colorbar_upper_limit,cmap=plt.cm.autumn, alpha = 1, animated = True)
    plt.colorbar()
    plt.title('Temp map (thresholded)')    
    
    gtitle = 'Dynamic=%d (t=%.1fs) ' % (nDynamic,  (nDynamic*Ts))
    plt.suptitle(gtitle)
    plt.show()


vol = 25
fig_tmp = plt.figure(figsize=(16,8))
# PlotImages(vol,Main,'Dynamic=' + str(vol)+' (t=' + str((vol-1)*Ts) +')')
PlotImages(vol,Main)


#  define function Plot_thermal (test with a volume):

def Plot_thermal_map(nDynamic, Main):
    IMAGES=Main.IMAGES    
    p2, p98 = np.percentile(IMAGES['Coronal']['Magnitude'][nDynamic][0]['data'], (2, 98))
    img_rescale = exposure.rescale_intensity(IMAGES['Coronal']['Magnitude'][nDynamic][0]['data'], in_range=(p2, p98))
         
    statmap = copy.deepcopy(IMAGES['Coronal']['Temperature'][nDynamic][0]['data'])
    for x in range(0,xlim2+1):
        for y in range(0,ylim2+1):
            if statmap[x,y]<temp_thresh:
                statmap[x,y] = 'nan'      
            
    plt.imshow(img_rescale,cmap=plt.cm.gray,interpolation='none')
    # imshow(statmap,vmin=40,vmax=55,cmap=plt.cm.YlOrBr, alpha = 1)
    plt.imshow(statmap,vmin=temp_thresh,vmax=colorbar_upper_limit,cmap=plt.cm.autumn, alpha = 1, animated = True)
    # cbar = plt.colorbar();     cbar.set_label('Temperature ($^\circ$C)')    
    gtitle = "t=%.1fs" % (nDynamic*Ts)  
    # plt.title(gtitle)    
    # plt.text(10, 240,gtitle,color='w',fontsize=18) 
    plt.text(10, 25,gtitle,color='w',fontsize=18) 
    plt.axis('off')     
    plt.show()

# fig_tmp = plt.figure(figsize=(7.5,3.5)),  Plot_thermal_map(vol,Main)


# In[Section 5; print figures through a loop]:

path_output_frames = path_output_scan +'\Frames'
path_output_frames_thermalmap = path_output_scan +'\Frames_thermalmap'
path_output_video = path_output + '\\' + session

if os.path.isdir(path_output_frames)==0:
    os.makedirs(path_output_frames)

if os.path.isdir(path_output_frames_thermalmap)==0:
    os.makedirs(path_output_frames_thermalmap)


# ind = [4,5,8,10]
ind = range(0,nFrames)
# ind = range(0,10)

t = time.time()
plt.close('all')  
for i in ind:
    fig_tmp = plt.figure(figsize=(16,8))
    # PlotImages(i,Main,'Dynamic=' + str(i))
    PlotImages(i,Main)
    print('Dynamic=' + str(i))
    filename = path_output_frames+'\image_'+str(i)+'.png'
    plt.savefig(filename, dpi=75)
    plt.close('all')  

elapsed = time.time() - t    
print('Time elapsed: '+str(round(elapsed))+' secs')
       
   
img_array = []
for i in ind: 
    filename = path_output_frames+'\image_'+str(i)+'.png'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
  
out = cv2.VideoWriter(path_output_video + '//' + scan_label + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size) 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

shutil.rmtree(path_output_frames, ignore_errors=False, onerror=None)
# Plot only the thermal map

plt.close('all')  
for i in ind:
    fig_tmp = plt.figure(figsize=(7.5,3.5))
    Plot_thermal_map(i,Main)
    
    print('Dynamic=' + str(i))
    # plt.show()
    filename = path_output_frames_thermalmap+'\image_'+str(i)+'.png'
    plt.savefig(filename, dpi=75,transparent=True)
    plt.close('all')




# In[Section 6: Plot timecourse of temperature in ROI]:

plt.close('all')  


frame_struct = 3
t_vol = 60    # in seconds
dynamic = int(t_vol/Ts)

image_ROI = copy.deepcopy(Main_data['MaskAverage'])*1.0
image_ROI[image_ROI==0] = np.nan

AvgTemp=np.array(Main_data['AvgTemperature'])
T10=np.array(Main_data['T10'])
T90=np.array(Main_data['T90'])

fig_tmp = plt.figure(figsize=(7.5,3.5))
Plot_thermal_map(dynamic,Main)
imshow(image_ROI,cmap=plt.cm.Greens,vmin=0.5,vmax=1.5)

filename = path_output_scan+'\ROI_for_temperature_evol.png'
plt.savefig(filename, dpi=200,transparent=True)


plt.figure(figsize=(6.8,3.9))    # Use this for separate figures [Figure 1]

plt.plot(timeD,AvgTemp,'*-')
plt.plot(timeD,T10,'*-')
plt.plot(timeD,T90,'*-')
plt.legend(['Avg. Temperature','T10','T90'])
plt.xlabel('Time (s)')
# plt.ylabel('Temperature (43$^{\circ}$)')
plt.ylabel('Temperature ($^\circ$C)')

# plt.ylim(0,90)
plt.ylim(36.5,50)
plt.grid(color = 'gray', linestyle = ':', linewidth = 1)
# plt.title('Temperature evolution in region of interest (ROI)')
filename = path_output_scan+'\Temperature_evol.png'
plt.savefig(filename, dpi=200)

# plt.tight_layout()











