import numpy as np
import scipy.stats as stt

def resistant_mean(vector, threshold = 5.):
    """
    this function is (numerically) the same as the resistant_mean function in IDL
    """

    vector = vector[~np.isnan(vector)]
    if vector.size == 0:
        return(float("NaN"),float("NaN"),float("NaN"))
    else:
        clipping_apers                         = stt.sigmaclip(vector,low=threshold,high=threshold)
        objs_after_clipping                    = clipping_apers[0] #objects after trimming
        mean_objs_after_clipping               = np.nanmean(objs_after_clipping)
        stddev_objs_after_clipping             = np.nanstd(objs_after_clipping) #this is the scatter or the standard deviation
        stddev_of_the_mean_objs_after_clipping = stddev_objs_after_clipping/np.sqrt(objs_after_clipping.size-1) #this is the standard deviation of the mean
        return(mean_objs_after_clipping,stddev_objs_after_clipping,stddev_of_the_mean_objs_after_clipping)