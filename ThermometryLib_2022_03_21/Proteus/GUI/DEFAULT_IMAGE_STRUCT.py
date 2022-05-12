def DEFAULT_IMAGE_STRUCT():
    """
    This functions returns the basic data structure that is associated to any stack of MRI data.

    :return: dictionary with elements defining a stack collection.
    """
    ImagesDict={'Magnitude':[],'Phase':[],'Temperature':[],'TimeArrival':[],'Dose':[],'MaskROI':[None],'SelPointsROI':[None],'TemperatureROIMask':[None]}
    return ImagesDict

def SINGLE_ELEM_ENTRIES():
    """
    This functions returns the entries in DEFAULT_IMAGE_STRUCT that have only a single list collection.
    Other elements are expected to have nested list `[][]`

    :return: list with elements of DEFAULT_IMAGE_STRUCT
    """
    return ['MaskROI','SelPointsROI','TemperatureROIMask']
