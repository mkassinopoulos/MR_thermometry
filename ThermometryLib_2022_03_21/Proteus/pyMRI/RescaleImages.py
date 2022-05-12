from __future__ import print_function
import numpy as np
def RescalePhase(image,info):
    '''
    We aply the golden rule to scale phase data properly
    '''
    if info['ScaleSlope']==0:
        print ("This is weird.... scaleslope =0, no scaling is applied")
        return image.astype(np.float32)
    newImage= (image.astype(np.float32) - info['ScaleIntercept'])/info['ScaleSlope']
    return newImage
    
def RescaleMagnitude(image,info):
    '''
    We aply the golden rule to scale magnitude data properly
    '''
    if info['RescaleSlope']==0:
        print ("This is weird.... RescaleSlope =0, no scaling is applied")
        return image.astype(np.float32)
    newImage= image.astype(np.float32)*info['RescaleSlope'] + info['RescaleIntercept']
    return newImage