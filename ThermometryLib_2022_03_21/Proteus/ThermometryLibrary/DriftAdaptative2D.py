from math import *
import numpy as np



LOGGER_NAME = 'MRgHIFU.by.Proteus'

import logging
logger = logging.getLogger(LOGGER_NAME)

#from numbapro import vectorize, float32, void,object_,uint8

def drift_iterate(nDOF,Tmap_ROI_List):
    Correction=np.zeros((nDOF),np.float32)
    TMapCorrected_ROI_List = Tmap_ROI_List

    #% Process the iterative drift fit
    do_iterate = True
    iter = 0
    while (do_iterate):
        PreviousCorrection = Correction
        #% that's the magic weighting which select relevant voxels only
        SumWeightMask =0
        weightMask_List=1.0/(1.0+TMapCorrected_ROI_List**2)
        SumWeightMask = SumWeightMask + weightMask_List.sum()
        if SumWeightMask==0:
            #logger.warning('TMapBaseline2: Mask for baseline drift correction is empty!!')
            return


        if self.Order == 0:
            Correction[0]=np.sum(weightMask_List*self.Tmap_ROI_List)
            Correction[0]=Correction[0]/SumWeightMask
            TMapCorrected_ROI_List=self.Tmap_ROI_List-Correction[0]


        elif self.Order == 1:
            A=np.zeros((3,3))
            B=np.zeros((3,1))
            A[0,0] = SumWeightMask
            A[0,1] = np.sum(weightMask_List*self.Fn_ROI_List[0,:].flatten())
            A[0,2] = np.sum(weightMask_List*self.Fn_ROI_List[1,:].flatten())
            A[1,1] = np.sum(weightMask_List*self.Fn_ROI_List[2,:].flatten())
            A[1,2] = np.sum(weightMask_List*self.Fn_ROI_List[3,:].flatten())
            A[2,2] = np.sum(weightMask_List*self.Fn_ROI_List[4,:].flatten())
            A[1,0] = A[0,1]
            A[2,0] = A[0,2]
            A[2,1] = A[1,2]

            B[0] = np.sum(weightMask_List*self.Tmap_ROI_List)
            B[1] = np.sum(weightMask_List*self.Tmap_ROI_List*self.Fn_ROI_List[0,:].flatten())
            B[2] = np.sum(weightMask_List*self.Tmap_ROI_List*self.Fn_ROI_List[1,:].flatten())

            Correction = np.linalg.solve(A,B)
            Correction=Correction.flatten()

            TMapCorrected_ROI_List=self.Tmap_ROI_List-(Correction[0]+Correction[1]*self.Fn_ROI_List[0,:].flatten()+Correction[2]*self.Fn_ROI_List[1,:].flatten())


        else:
            A=np.zeros((6,6))
            B=np.zeros((6,1))
            A[0,0] = SumWeightMask

            for K in range(1,6):
                A[0,K] = np.sum(weightMask_List*self.Fn_ROI_List[K-1,:].flatten())

            A[1,1] = A[0,3]
            A[1,2] = A[0,5]
            for K in range(3,6):
                A[1,K] = np.sum(weightMask_List*self.Fn_ROI_List[K+2,:].flatten())

            A[2,2] = A[0,5]
            A[2,3] = A[1,4]
            A[2,4] = A[1,5]
            A[2,5] = np.sum(weightMask_List*self.Fn_ROI_List[8,:].flatten())

            for K in range(3,6):
                A[3,K] = np.sum(weightMask_List*self.Fn_ROI_List[K+6,:].flatten())

            A[4,4] = A[3,5]
            A[4,5] = np.sum(weightMask_List*self.Fn_ROI_List[12,:].flatten())
            A[5,5] = np.sum(weightMask_List*self.Fn_ROI_List[13,:].flatten())

            B[0]= np.sum(weightMask_List*self.Tmap_ROI_List)
            for K in range(1,6):
                B[K] = np.sum(weightMask_List*self.Tmap_ROI_List*self.Fn_ROI_List[K-1,:].flatten())


            for i in range(6):
                for j in range(6):
                    if (j<i):
                        A[i,j]=A[j,i]

            Correction = np.linalg.solve(A,B)
            Correction=Correction.flatten()

            TMapCorrected_ROI_List=self.Tmap_ROI_List-Correction[0]
            for i in range(1,6):
                TMapCorrected_ROI_List-=Correction[i]*self.Fn_ROI_List[i-1,:].flatten()

        iter+=1
        CorrectionChange=np.sum(np.abs(PreviousCorrection-Correction))

        if CorrectionChange<self.DriftAccuracy:
            do_iterate=False
        if iter == self.MaxIteration:
            do_iterate=False
            #logger.debug( 'TMapBaseline2: ' + self.SelKey + ' baseline drift fitting did not convergence in %d iterations' %iter)


class DriftAdaptative2D():
    '''
    Class to keep track of the adaptive 2D algorithm for drift correction
    '''
    def __init__(self,SelKey,HistoryMaxLength,BodyTemperature,IMAGE,excl_roiCyl_radius_mm,excl_roiCyl_h_mm,Order,enableLogs):
        self.SelKey=SelKey #ID for the stack,'Coronal','Sagittal','User1', etc
        self.EnableLogs=enableLogs
        self.Order=Order           # Order of the polynimial fit: drift correction=sigma(Kij*X^i*Y^j) with i+j<=order
        self.MaxIteration = 30     # Number max of drift fit iteration
        self.DriftAccuracy = 0.01  # Temperature accuracy of the drift fit in Celsius
        self.HistoryMaxLength = HistoryMaxLength
        self.iBaselineTHistory=1
        self.BodyTemperature = BodyTemperature
        self.voxelSizeX = IMAGE['info']['VoxelSize'][0]*1e3
        self.voxelSizeY = IMAGE['info']['VoxelSize'][1]*1e3
        self.XResolution=IMAGE['data'].shape[0]
        self.YResolution=IMAGE['data'].shape[1]
        self.XYResolution=self.XResolution*self.YResolution
        self.PrevTemp=np.ones(IMAGE['data'].shape,np.float32)*BodyTemperature

        Y,X=np.meshgrid(np.linspace(-1.0,1.0,num=self.YResolution),
                        np.linspace(-1.0,1.0,num=self.XResolution))

        Lenght_norm=excl_roiCyl_h_mm/(self.voxelSizeY*self.YResolution)
        Radius_norm=excl_roiCyl_radius_mm/(self.voxelSizeX*self.XResolution*0.5)

        self.MaskCircle = ((X**2)+(Y**2)) < Radius_norm**2
        self.InvMaskCircle=self.MaskCircle==False

        self.MaskRect = np.logical_and(np.abs(X)<Lenght_norm,np.abs(Y)<Radius_norm)
        self.InvMaskRect=self.MaskRect==False

        #~ Y,X=np.meshgrid(((np.arange(self.XResolution)+1)-self.XResolution*0.5)/(self.XResolution*0.5),
                        #~ ((np.arange(self.YResolution)+1)-self.YResolution*0.5)/(self.YResolution*0.5))

        Y,X=np.meshgrid(np.linspace(-1,1.0,num=self.YResolution),
                        np.linspace(-1,1.0,num=self.XResolution))

        if Order == 0 or Order ==-1:
            self.Fn=0
        elif Order == 1:
            self.Fn=np.zeros((5,self.XResolution,self.YResolution),np.float32)
            self.A=np.zeros((3,3),np.float32)
            self.B=np.zeros((3,1),np.float32)
        elif Order == 2:
            self.Fn=np.zeros((14,self.XResolution,self.YResolution),np.float32)
            self.A=np.zeros((6,6),np.float32)
            self.B=np.zeros((6,1),np.float32)

        if Order == 1 or Order == 2:
            self.Fn[0,:,:]=X
            self.Fn[1,:,:]=Y
            self.Fn[2,:,:]=X**2
            self.Fn[3,:,:]=X*Y
            self.Fn[4,:,:]=Y**2
            if Order == 2:
                self.Fn[5,:,:]=X**3
                self.Fn[6,:,:]=(X**2)*Y
                self.Fn[7,:,:]=X*(Y**2)
                self.Fn[8,:,:]=Y**3
                self.Fn[9,:,:]=X**4
                self.Fn[10,:,:]=(X**3)*Y
                self.Fn[11,:,:]=(X**2)*(Y**2)
                self.Fn[12,:,:]=X*(Y**3)
                self.Fn[13,:,:]=Y**4
        self.maskSNRcold=None

        if Order !=-1:
            nDOF=[1,3,6][self.Order]
            self.SumAppliedCorrection = np.zeros(nDOF,np.float32)
            self.FilteredCorrection = np.zeros(nDOF,np.float32)
        self.History_Inst_Correction_Coef = [] # np.zeros((nDOF,MaxNumberOfSlice,nDyn-1))
        self.History_Cumul_Correction_Coef = [] # np.zeros((nDOF,MaxNumberOfSlice,nDyn-1))
        self.TimeList = []
        self.CorrectionList=[]
        self.PrevDyn=-1
        self.LastSliceNumber=-1
        self.Inst_Correction_Array=np.zeros((self.XResolution,self.YResolution),np.float32)
        self.Cumul_Correction_Array=np.zeros((self.XResolution,self.YResolution),np.float32)
        self.Cumul_Correction_Array0=np.zeros((self.XResolution,self.YResolution),np.float32)
        self.Cumul_Correction_Array1=np.zeros((self.XResolution,self.YResolution),np.float32)
        self.Cumul_Correction_Array2=np.zeros((self.XResolution,self.YResolution),np.float32)

        self.ResetHistory()

    def ResetHistory(self):
        if self.Order !=-1:
            nDOF=[1,3,6][self.Order]
            self.SumAppliedCorrection[:] = 0.0
            self.FilteredCorrection[:] = 0.0

        del self.History_Inst_Correction_Coef[:]
        del self.History_Cumul_Correction_Coef[:] # np.zeros((nDOF,MaxNumberOfSlice,nDyn-1))
        del self.TimeList[:]
        del self.CorrectionList[:]
        self.PrevDyn=-1
        self.LastSliceNumber=-1
        self.Inst_Correction_Array[:]=0.0
        self.Cumul_Correction_Array[:]=0.0
        self.Cumul_Correction_Array0[:]=0.0
        self.Cumul_Correction_Array1[:]=0.0
        self.Cumul_Correction_Array2[:]=0.0


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%% Function to process the drift for each slice and each dynamic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def update(self,SliceAcqTime,Tmap,maskSNR,UserMask,BodyTemperature,MaskHeating=True,UserDriftMask=None,):


        #% extract temperature in non heated region and not masked by SNR
        maskSNRcold=maskSNR.copy()

        if UserDriftMask is None :
            if MaskHeating:
                if self.SelKey=='Coronal':
                    maskSNRcold[self.MaskCircle]=False
                elif self.SelKey=='Sagittal':
                    maskSNRcold[self.MaskRect]=False
                elif UserMask is not None:
                    maskSNRcold[UserMask]=False #FOR USER1 AND USER2, the user can provide the mask where to avoid calculate the drifting

        else:
            logger.info('Updating drift using user SNR mask')
            maskSNRcold*=UserDriftMask

        self.maskSNRcold=maskSNRcold.copy() #We need this for further analysis outside this function

        if self.Order==-1:
            return

        self.Tmap_ROI_List=Tmap[maskSNRcold]-BodyTemperature

        if self.Order == 0:
            Basic0order=self.Tmap_ROI_List.mean()
        #Sep 28, 2016. This may change a bit the outcome
        #self.Tmap_ROI_List=Tmap[maskSNRcold]-self.PrevTemp[maskSNRcold]
        if self.Order > 0:
            self.Fn_ROI_List=self.Fn[:,maskSNRcold]

        self.N_voxels_Used=len(self.Tmap_ROI_List)


        nDOF=[1,3,6]
        nDOF=nDOF[self.Order]

        #% process polynomial fit correction
        self.TimeList.append(SliceAcqTime)
        self.CorrectionList.append(np.zeros(nDOF,np.float32))

        Correction=np.zeros((nDOF),np.float32)
        TMapCorrected_ROI_List = self.Tmap_ROI_List

        #% Process the iterative drift fit
        do_iterate = True
        iter = 0



        while (do_iterate):
            PreviousCorrection = Correction
            #% that's the magic weighting which select relevant voxels only
            SumWeightMask =0
            weightMask_List=1.0/(1.0+TMapCorrected_ROI_List**2) # TODO: TO review this wighting criteria
            SumWeightMask = SumWeightMask + weightMask_List.sum()
            if SumWeightMask==0:
                #logger.warning('TMapBaseline2: Mask for baseline drift correction is empty!!')
                return


            if self.Order == 0:
                Correction[0]=np.sum(weightMask_List*self.Tmap_ROI_List)
                Correction[0]=Correction[0]/SumWeightMask
                #Correction[0]= self.Tmap_ROI_List.mean()
                TMapCorrected_ROI_List=self.Tmap_ROI_List-Correction[0]
            elif self.Order == 1:
                self.A[:]=0.0
                self.B[:]=0.0
                self.A[0,0] = SumWeightMask
                self.A[0,1] = (weightMask_List*self.Fn_ROI_List[0,:]).sum()
                self.A[0,2] = (weightMask_List*self.Fn_ROI_List[1,:]).sum()
                self.A[1,1] = (weightMask_List*self.Fn_ROI_List[2,:]).sum()
                self.A[1,2] = (weightMask_List*self.Fn_ROI_List[3,:]).sum()
                self.A[2,2] = (weightMask_List*self.Fn_ROI_List[4,:]).sum()
                self.A[1,0] = self.A[0,1]
                self.A[2,0] = self.A[0,2]
                self.A[2,1] = self.A[1,2]

                self.B[0] = (weightMask_List*self.Tmap_ROI_List).sum()
                self.B[1] = (weightMask_List*self.Tmap_ROI_List*self.Fn_ROI_List[0,:]).sum()
                self.B[2] = (weightMask_List*self.Tmap_ROI_List*self.Fn_ROI_List[1,:]).sum()

                Correction = np.linalg.solve(self.A,self.B)
                Correction=Correction.flatten()

                TMapCorrected_ROI_List=self.Tmap_ROI_List-(Correction[0]+Correction[1]*self.Fn_ROI_List[0,:]+Correction[2]*self.Fn_ROI_List[1,:])


            else:
                self.A[:]=0.0
                self.B[:]=0.0
                self.A[0,0] = SumWeightMask

                for K in range(1,6):
                    self.A[0,K] = (weightMask_List*self.Fn_ROI_List[K-1,:]).sum()

                self.A[1,1] = self.A[0,3]
                #self.A[1,2] = self.A[0,5]
                self.A[1,2] = self.A[0,4] #oh my
                for K in range(3,6):
                    self.A[1,K] = (weightMask_List*self.Fn_ROI_List[K+2,:]).sum()

                self.A[2,2] = self.A[0,5]
                self.A[2,3] = self.A[1,4]
                self.A[2,4] = self.A[1,5]
                self.A[2,5] = (weightMask_List*self.Fn_ROI_List[8,:]).sum()

                for K in range(3,6):
                    self.A[3,K] = (weightMask_List*self.Fn_ROI_List[K+6,:]).sum()

                self.A[4,4] = self.A[3,5]
                self.A[4,5] = (weightMask_List*self.Fn_ROI_List[12,:]).sum()
                self.A[5,5] = (weightMask_List*self.Fn_ROI_List[13,:]).sum()

                self.B[0]= (weightMask_List*self.Tmap_ROI_List).sum()
                for K in range(1,6):
                    self.B[K] = (weightMask_List*self.Tmap_ROI_List*self.Fn_ROI_List[K-1,:]).sum()


                for i in range(6):
                    for j in range(6):
                        if (j<i):
                            self.A[i,j]=self.A[j,i]

                Correction = np.linalg.solve(self.A,self.B)
                Correction=Correction.flatten()

                TMapCorrected_ROI_List=self.Tmap_ROI_List-Correction[0]
                for i in range(1,6):
                    TMapCorrected_ROI_List-=Correction[i]*self.Fn_ROI_List[i-1,:].flatten()

            iter+=1
            CorrectionChange=np.sum(np.abs(PreviousCorrection-Correction)) #TODO: Review this criteria

            if CorrectionChange<self.DriftAccuracy or self.Order ==0:
                do_iterate=False
            if iter == self.MaxIteration:
                do_iterate=False
                logger.debug( 'DrifAdap: ' + self.SelKey + ' baseline drift fitting did not convergence in %d iterations' %iter)


        logger.debug( 'DriftAdap: ' + self.SelKey + ' final mask usage weighting is %f ' %(100.0*SumWeightMask/self.N_voxels_Used))
        logger.debug( 'DriftAdap: ' + self.SelKey + ' iteration %d, final correction change = %f' %(iter,CorrectionChange))


        RawDrift=self.SumAppliedCorrection+Correction
        logger.debug('DriftAdap: ' + self.SelKey + ',Raw Drift = ' + str(RawDrift))

        self.CorrectionList[-1] = RawDrift

        if len(self.TimeList)>2 and self.HistoryMaxLength>=2:
            self.FilteredCorrection = self.InterpolateCorrection2D(self.TimeList[-self.HistoryMaxLength:],
                                                                              self.CorrectionList[-self.HistoryMaxLength:])
            self.FilteredCorrection = self.FilteredCorrection-self.SumAppliedCorrection

        else:
            self.FilteredCorrection=Correction

        self.SumAppliedCorrection = self.SumAppliedCorrection + self.FilteredCorrection


        #logger.debug( 'TMapBaseline2: ' + self.SelKey + ' Filtered Drift =  ' + str(self.SumAppliedCorrection))


        #% apply polynonmial fit determined from previous dynamics
        self.Inst_Correction_Array = np.zeros(self.XYResolution,np.float32) - self.FilteredCorrection[0]
        if self.Order==0:
            logger.debug('Basic0order, self.FilteredCorrection[0] %f %f' %(Basic0order, self.FilteredCorrection[0]))
        for ii in range (1,nDOF):
            self.Inst_Correction_Array = self.Inst_Correction_Array - self.FilteredCorrection[ii]*self.Fn[ii-1,:].flatten()

        self.Inst_Correction_Array=np.reshape(self.Inst_Correction_Array,(self.XResolution,self.YResolution))
        #% Tmap = Tmap + self.Inst_Correction_Array
        self.History_Inst_Correction_Coef.append(-self.FilteredCorrection)

        self.Cumul_Correction_Array  += self.Inst_Correction_Array
        self.Cumul_Correction_Array0 -=  self.FilteredCorrection[0]

        if self.Order > 0:
            for ii in range (1,3):
                self.Cumul_Correction_Array1 -= np.reshape(self.FilteredCorrection[ii]*self.Fn[ii-1,:].flatten(),(self.XResolution,self.YResolution))
            if self.Order > 1:
                for ii in range (3,6):
                    self.Cumul_Correction_Array2-=np.reshape(self.FilteredCorrection[ii]*self.Fn[ii-1,:].flatten(),(self.XResolution,self.YResolution))
        if  len(self.History_Cumul_Correction_Coef)==0:
            self.History_Cumul_Correction_Coef.append(self.History_Inst_Correction_Coef[-1])
        else:
            self.History_Cumul_Correction_Coef.append(self.History_Cumul_Correction_Coef[-1]+self.History_Inst_Correction_Coef[-1])
        self.PrevTemp[:]=Tmap[:]

    def InterpolateCorrection2D(self,TimeList,CorrectionList):
        #% interpolate the drift correction as function of time
        if type(TimeList)==np.ndarray:
            TimeListArray=TimeList
        else:
            TimeListArray=np.array(TimeList)

        if type(CorrectionList)==np.ndarray:
            CorrectionListArray=CorrectionList
        else:
            CorrectionListArray=np.array(CorrectionList)
        x = TimeListArray
        y = CorrectionListArray

        p = np.polyfit(x, y, 1)

        CorrectionInterpolated=TimeListArray[-1]*p[0]+p[1] #we evaluate at the last term

        return CorrectionInterpolated
