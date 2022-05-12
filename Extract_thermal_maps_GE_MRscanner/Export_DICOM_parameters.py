# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:22:56 2022

@author: Michalis
"""

import os
from pathlib import PureWindowsPath,PurePosixPath,PurePath
import pydicom
import glob
import os.path
import numpy as np
import xlwt
from xlwt import Workbook
  
# Workbook is created


Experiment_Number = 1

#Get 


# mri_dicom_directory = "D:/SOUNDPET/ThermometryLib_2022_03_21/MR_Thermometry_Examples/2021_12_29_meat/2"
mri_dicom_directory = r"D:\SOUNDPET\Raw_data\2021_11_27\4.lumbart 150w 35s"




target_path = r""+mri_dicom_directory+"\*.dcm"#Set the file type we want to search for in the directory
list_of_dicom_path =glob.glob(target_path); #Search inside the dicom directory


ds = pydicom.read_file(list_of_dicom_path[3])#Assign the dicom directory file to the pydicom obj



Sequence_Info = ds[0x0008, 0x103E].value
Coil_Info = ds[0x0018,0x1250].value
Acquisition_Type =ds[0x0018,0x0023].value
TR_Info =ds[0x0018,0x0080].value
TE_Info = ds[0x0018,0x0081].value
Flip_Angle_Info =ds[0x0018,0x1314].value
Echo_Train_Length_Info =ds[0x0018,0x0091].value
Pixel_Bandwidth_Info =ds[0x0018,0x0095].value

Row_Info =ds[0x0028,0x0010].value
Column_Info =ds[0x0028,0x0011].value

pixel_spacing = ds[0x0028, 0x0030]
pixel_values = pixel_spacing.value
pixel_spacing_row = pixel_values[0]
pixel_spacing_column =pixel_values[1]

Row_In_mm = Row_Info * pixel_spacing_row
Column_In_mm = Column_Info * pixel_spacing_column

Slice_Thickness = ds[0x0018,0x0050].value

Field_Of_View_Info = str(int(Row_In_mm)) + " X " +str(int(Column_In_mm)) + " X " + str(Slice_Thickness)  


Acuisition_Matrix_Info = ds[0x0018,0x1310].value

Acuisition_Matrix_Info_String = str(Acuisition_Matrix_Info[1]) + " X " + str(Acuisition_Matrix_Info[2])


Number_of_Average_Info = ds[0x0018,0x0083].value



# Triggering_Time_Info = (ds[0x0018, 0x1060].value)/1000
Triggering_Time_Info = 'nan'

excel_file_directory ="C:/Users/Michalis/Desktop/Export Dicom Info/" + str(Experiment_Number) + "_Dicom Info.xls"
wb = Workbook()
# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1')
  

sheet1.write(0, 0, 'Sequence')
sheet1.write(0, 1, 'Type')
sheet1.write(0, 2, 'Coil')
sheet1.write(0, 3, 'Acquisition Type')
sheet1.write(0, 4, 'TR (ms)')
sheet1.write(0, 5, 'TE (ms)')
sheet1.write(0, 6, 'FLIP ANGLE')


sheet1.write(1, 0, str(Sequence_Info))
sheet1.write(1, 1, 'GE')
sheet1.write(1, 2, str(Coil_Info))
sheet1.write(1, 3, str(Acquisition_Type))
sheet1.write(1, 4, str(TR_Info))
sheet1.write(1, 5, str(TE_Info))
sheet1.write(1, 6, str(Flip_Angle_Info))

sheet1.write(2, 0, 'Echo Train Length')
sheet1.write(2, 1, 'Pixel Bandwidth (Hertz/pixel)')
sheet1.write(2, 2, 'Field of view (mm3)')
sheet1.write(2, 3, 'Acquisition Matrix Size (freq x phase)')
sheet1.write(2, 4, 'No. of Averages')
sheet1.write(2, 5, 'Triggering Time (s)')


sheet1.write(3, 0, str(Echo_Train_Length_Info))
sheet1.write(3, 1, str(Pixel_Bandwidth_Info))
sheet1.write(3, 2, str(Field_Of_View_Info))
sheet1.write(3, 3, str(Acuisition_Matrix_Info_String))
sheet1.write(3, 4, str(Number_of_Average_Info))
sheet1.write(3, 5, str(Triggering_Time_Info))
  
  
wb.save(excel_file_directory)