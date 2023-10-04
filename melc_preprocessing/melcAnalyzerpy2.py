import sys
import os
import re
import glob
import numpy as np
import xml.etree.ElementTree as ET

from pylab import *                           # standard stuff
#from scipy.signal import correlate2d          # drift correction
#from scipy.misc import imresize               # image resize
#from scipy.ndimage.interpolation import shift # subpixel image shift
import tifffile  # save 8bit single-channel TIF files
#import cv2

print "### melcAnalyzer started - version 1.0 ###\n"

pIm  = imread("/data/bionets/je30bery/test_als/ALS01 - 202301161240_1/source/p_PBS_200_XF116-2_001.png")[15:-15,15:-15]
print(pIm.shape)

print(len(np.unique(pIm)))
pImRef = pIm
# save phase contrast image
#gray()
pImCorr = pIm.copy()
pImCorr = pImCorr-percentile(pImCorr,20.*0.135)
pImCorr[pImCorr<0] = 0
pImCorr = pImCorr/percentile(pImCorr,100-1.*0.135)
pImCorr[pImCorr>1] = 1
tifffile.imsave("test_py2_phase.tif", (pImCorr*255*255).astype(np.uint16))
print '+ Saved phase contrast image.'