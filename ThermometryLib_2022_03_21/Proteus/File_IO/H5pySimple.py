from __future__ import print_function, absolute_import
from future.utils import iteritems
import sys
if sys.version_info > (3, 2):
    def bCheckIfStr(v):
        return type(v) ==str
    def cStr(v):
        if type(v)==bytes:
            return v.decode("utf-8")
        else:
            return v
else:
    def bCheckIfStr(v):
        return (type(v) ==str or type(v) ==unicode)
    def cStr(v):
        return v
'''
Generic BackEnd to save hdf5 files using h5py, it seems more efficient than Pytables for "very-general-purpose saving". For better designed groups and datasets, pytables is way better
'''
import numpy
import h5py
from pydicom.uid import UID
from collections import OrderedDict

def ProcType(k,v,f,compatibility,group):
    nametype=type(v).__name__
    if group is None:
        group=f['/']
    if  type(v) in [dict,OrderedDict]:
        if isinstance(k,str):
            indexType='str'
        elif isinstance(k,unicode):
            indexType='unicode'
        elif isinstance(k,int):
            indexType='int'
        elif isinstance(k,float):
            indexType='float'
        else:
            raise TypeError('The type of key ' +str(k)+ ' is not supported ( ' + str(type(k)) + ' )')
        stk = k if (isinstance(k,str) or isinstance(k,unicode)) else str(k)
        newgroup=group.create_group(stk)
        newgroup.attrs["type"]="dict" if type(v) is dict else "OrderedDict"
        newgroup.attrs["type_key"]=indexType
        SaveToH5py(v,f,compatibility,newgroup)
    elif type(v) ==list:
        newgroup=group.create_group(k)
        newgroup.attrs["type"]="list";
        for n in range(len(v)):
            vlist=v[n]
            itemname = "item_%d"%n
            ProcType(itemname,vlist,f,compatibility,newgroup)
    elif type(v) ==tuple:
        nameList = k
        newgroup=group.create_group(k)
        newgroup.attrs["type"]="tuple"
        for n in range(len(v)):
            vlist=v[n]
            itemname = "item_%d"%n
            ProcType(itemname,vlist,f,compatibility,newgroup)
    elif type(v) == numpy.ndarray :
        ##we can apply compression rules for larger arrays
        if v.nbytes >2**10: # if array takes more 1024 bytes
            if compatibility:
                ds=group.create_dataset(k, data=v,compression="lzf") #this is very simple but accessible anywhere compressor, including matlab
            else:
                ds=group.create_dataset(k, data=v,compression="gzip",compression_opts=9) #this gives the best compression
        else:
            ds=group.create_dataset(k, data=v)
        ds.attrs["type"]="ndarray"
    elif  bCheckIfStr(v) or type(v) == numpy.string_ or \
         type(v) ==numpy.unicode_ :
        ##we'll apply compression rules for larger arrays
        ds=group.create_dataset(k, data=v)
        ds.attrs["type"]=str(type(v))
    elif numpy.isscalar(v):
        ds=group.create_dataset(k, data=v)
        ds.attrs["type"]="scalar"
    elif v is None:
        ds=group.create_dataset(k, data=0)
        ds.attrs["type"]="None"
    elif nametype == 'UID':
        sUID=str(v)
        ds=group.create_dataset(k, data=sUID)
        ds.attrs["type"]=nametype
    else:
        raise TypeError( "Datatype not handled:" + nametype)


def SaveToH5py(MyDict,f,compatibility=False,group=None):
    if bCheckIfStr(f):
        fileobj=h5py.File(f, "w")
    elif type(f)==h5py._hl.files.File:
        fileobj=f
    else:
        raise TypeError( "only string or h5py.file objects are accepted for 'f' object")

    if type(MyDict) not in [dict,OrderedDict] or type(MyDict)==list:
        raise TypeError("Only dictionaries or lists are supported to be saved")
    for (k, v )in iteritems(MyDict):
        ProcType(k,v,fileobj,compatibility,group)
    if bCheckIfStr(f):
        fileobj.close() #if we receive a string, that means that we open and close the file

def ReadFromH5py(f,group=None,typeattr=None):
    if bCheckIfStr(f):
        fileobj=h5py.File(f, "r")
    elif type(f)==h5py._hl.files.File:
        fileobj=f
    else:
        raise TypeError( "only string or h5py.file objects are accepted for 'f' object")
    if group is None:
        group=fileobj
        MyDict={}
    else:
        if typeattr=='OrderedDict':
            MyDict=OrderedDict()
        else:
            MyDict={}
    for (namevar,val) in iteritems(group):

        NameList=None
        typeattr=cStr(val.attrs["type"])
        if type(val)==h5py._hl.group.Group:
            ValGroup=ReadFromH5py(fileobj,val)
            if typeattr=="list" or  typeattr=="tuple" :
                NameList=namevar
            if NameList is not None:
                ListVal=[]
                for n in range(len(ValGroup)):
                    itemname = "item_%d"%n
                    ListVal.append(ValGroup[itemname])
                if typeattr=="list":
                    MyDict[NameList] = ListVal
                else:
                    MyDict[NameList] = tuple(ListVal)
            else:

                if "type_key" in val.attrs:
                    typekey= cStr(val.attrs["type_key"])
                    if typekey=='int':
                        snamevar=int(namevar)
                    elif typekey=='float':
                        snamevar=int(namevar)
                    elif typekey=='str':
                        snamevar=namevar
                    elif typekey=='unicode':
                        snamevar=namevar
                    else:
                        raise TypeError( "the type of dictionary key is not supported " + namevar + ' (' + typekey + ')')
                else:
                    snamevar=namevar
                MyDict[snamevar] = ValGroup
        elif typeattr == "None":
             MyDict[namevar]=None
        elif typeattr == "UID":
            MyDict[namevar]=UID(val[()])
        else:
            if cStr(val.attrs["type"])=="scalar":
                if type(val[()])==numpy.int32 or type(val[()])==numpy.int64:
                    MyDict[namevar]=int(val[()])
                else:
                    MyDict[namevar] = val[()]
            elif cStr(val.attrs["type"])=="<type 'str'>" or val.attrs["type"]=="<class 'str'>" :
                MyDict[namevar] = cStr(val[()])
            else:
                MyDict[namevar] = val[()]
    if bCheckIfStr(f):
        fileobj.close()
    return MyDict
