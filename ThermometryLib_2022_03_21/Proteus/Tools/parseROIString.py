from __future__ import print_function
import re

def parseROIString(InputStr):
    '''
    % Create an structure of ROIs that will be used to perform analysis
    % by example  ' 2 C 3.4 | 3 E 9.6 7 3 4' will generate
    % ROI{1}.Slice=2
    % ROI{1}.Type='CIRCLE'
    % ROI{1}.Radius=3.4
    % ROI{1}.CenterX=0
    % ROI{1}.CenterY=0
    % ROI{2}.Slice=3
    % ROI{2}.Type='ELLIPSE'
    % ROI{2}.RadiusA=9.6
    % ROI{2}.RadiusB=7
    % ROI{2}.CenterX=3
    % ROI{2}.CenterY=4
    % ROI{2}.Rotation=0
    %
    % Multiple ROIs can be defined in the same slice, by example:
    %   ' 2 E 8 20 | 2 R 10 40 -30 0' : The first ROI is for slice 2 is an ellipse with small
    %       radius in X of 8 mm and long radius of 20 mm. The second ROI is also
    %       for slice 2 and is a square  of 10 x 40 mm centered at -30 mm from
    %       center on X and 0 mm on Y.
    %
    % Follow these specifications depending on the shape, parameters between
    % brackets mean that they are optional. Distance units are in mm and angles in degrees.
    % Rotation is always assumed at center of ROI
    % Use "|" to separate between diferent ROIs
    % Basic sintax is:
    % SliceNumber Type-of-ROI Parameters | Slicenumber Type-of-ROI Parameters | ...
    %
    % Circle:  SliceNumber C RadiusA [ CenterX CenterY]
    %   Examples. 1 C 6 : Circle on slice 1 with radius of 6 mm at center of image:
    %             3 C 5.6 3 -1.2: Circle on slice 3 with radius of 5.6 mm with center at coordinates x=+3mm
    %                           and y=-1.2 mm from center of image
    %
    % Ellipse : SliceNumber E RadiusX RadiusY [ CenterX CenterY RotationAngle]
    %   Examples. 2 E 10 20 : Ellipse on slice 2 with radiusX of 10 mm and
    %                         radiusY of 20mm at center of image
    %             4 E 4.5 12 5 6 : Ellipse on slice 4 with radiusX of 4.5 mm and
    %                              radiusY of 12, with center at 5mm on X
    %                              anf 6 mm on Y.
    %             4 E 4.5 12 5 6 34: Similar to previous example but adding a
    %                               rotation angle of the ellipse of 34 degrees
    %
    % Square : SliceNumber S Length [CenterX CenterY RotationAngle]
    %
    % Rectangle: SliceNumber S LengthX LengthY [CenterX CenterY RotationAngle]
    %
    % UserDefined: SliceNumber U Point1X Point1Y Point2X Point2Y Point3X  Point3Y [Point4X Point4Y ...]
    %           For user-defined ROI a list of at least 3 points must be
    %           provided.
    %  Example. 4 U -8 -13.13 -6.4 -14.6  6.4 -14.6 8 -13.13 8 10.8  5.45 15.2  -5.45 15.2  -8 10.8 :
    %           This list of points defines a ROI with cigar shape of 16 mm of
    %           diameter and length close to 30 mm.
    '''

    singleSp=re.compile(' +')
    Sp=re.compile(' ')

    IStr=singleSp.sub(' ',InputStr.strip(' '))

    sections=re.compile('\|')

    Parts=sections.split(IStr);


    ROIs=[];
    SErr='Error trying to parse the string for ROIs, verify correct syntax: '
    for r in Parts:
        entry={}
        Items=Sp.split(singleSp.sub(' ',r.strip(' ')))
        if len(Items)<2:
            raise ValueError(SErr+'Unsufficient parameters in ROI entry :' +r);

        entry['SliceNumber'] = abs(int(Items[0]))
        entry['NegativeMask']=int(Items[0])<0

        if Items[1]== 'C': #circle
            if len( Items)<3:
                raise ValueError('"C" needs one value for radius')

            entry['Type']='CIRCLE'
            entry['Radius']=float(Items[2])
            entry['CenterX'],entry['CenterY']=ParseForCenter(Items,3)

        elif Items[1]=='E': #ellipse
            if len( Items)<4:
                 raise  ValueError(SErr+'"E" needs two values for radius X and Y:' +r)
            entry['Type']='ELLIPSE'
            entry['RadiusX']=float(Items[2])
            entry['RadiusY']=float(Items[3])
            entry['CenterX'],entry['CenterY']=ParseForCenter(Items,4)
            entry['Rotation']=0.
            if len( Items)>=7:
                entry['Rotation']=float(Items[6]);

        elif Items[1]== 'S': #Square
            if len( Items)<3:
                raise ValueError(SErr+'"S" needs one values for length :' + r)
            entry['Type']='SQUARE'
            entry['Length']=float(Items[2])
            entry['CenterX'],entry['CenterY']=ParseForCenter(Items,3)
            entry['Rotation']=0.;
            if len( Items)>=6:
                entry['Rotation']=float(Items[5])

        elif Items[1]== 'R': #Rectangle
            if len( Items)<4:
               raise ValueError(SErr+'"S" needs two values for lengths X and Y:' + r)
            entry['Type']='RECTANGLE'
            entry['LengthX']=float(Items[2])
            entry['LengthY']=float(Items[3])
            entry['CenterX'],entry['CenterY']=ParseForCenter(Items,4)
            entry['Rotation']=0.;
            if len( Items)>=7:
                entry['Rotation']=float(Items[6])
        elif Items[1]== 'U':#User-defined
            if len( Items)<8:
                raise ValueError(SErr+'"U" needs at least 3 pairs of values for points X Y to define the ROI::' + r);
            if len(Items)%2!=0:
                raise ValueError(SErr+'"U" list of points must be in pairs:' + r)
            entry['Type']='USER'
            entry['PointsX']=[float(Items[2]),float(Items[4]),float(Items[6])]
            entry['PointsY']=[float(Items[3]),float(Items[5]),float(Items[7])]
            if len(Items)>=9:
                for m in range(8,len(Items),2):
                    entry['PointsX'].append(float(Items[m]))
                    entry['PointsY'].append(float(Items[m+1]))

        else: #we may add later other typeof ROIs, we could very easily add later user defined
            raise ValueError(SErr+'Unable to decode entry:' +r)
        ROIs.append(entry)


    FinalROIs=sorted(ROIs,key=lambda d: (d['SliceNumber'])) #we'll sure the list is list by slice and check there
    return FinalROIs



def ParseForCenter(Items,LocationX):
    x=0
    y=0
    if  len( Items)-1>=LocationX:
        x=float(Items[LocationX])
    if  len( Items)-1>=LocationX+1:
        y=float(Items[LocationX+1])
    return x,y




if __name__ == "__main__":
  InputStr=' 1 R  8.3 10 4.5 6 3| 2 C   4.4 | 3 E 12.6 7 0 0 -35 | 4 U 9.5 4 10.5 3.6  10.5 -3.6  9.5 -4  -8 -4  -10.5 -3.2  -10.5  3.2  -8 4'
  #InputStr='2 C 4 | 4 U 9.5 4 10.5 3.6  10.5 -3.6  9.5 -4  -8 -4  -10.5 -3.2  -10.5  3.2  -8 4 | 5 U 4 9.5 3.6 10.5  -3.6 10.5 -4 9.5 -4 -8  -3.2 -10.5  3.2 -10.5 4 -8'
  print ('testing with string=',InputStr)
  print (parseROIString(InputStr))
