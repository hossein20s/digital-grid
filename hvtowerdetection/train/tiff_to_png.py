import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal as gd


# get list of all subdirectories in a directory
def get_subdirs(dir):
    "Get a list of immediate subdirectories"
    return next(os.walk(dir))[1]


# get list of immediate files in a directory
def get_subfiles(dir):
    "Get a list of immediate subfiles"
    return next(os.walk(dir))[2]


# Function to stretch the 16bit image to 8bit. Play with the percent inputs.
def stretch_8bit(bands, lower_percent=0.5, higher_percent=99.5):
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.uint8)


# Save stretched 8bit array to png
def saveTopng(tifffolder, pngfolder, tifffilename, dpi=400):
    ds = gd.Open(tifffolder + '/' + tifffilename)
    r = ds.GetRasterBand(1).ReadAsArray()
    g = ds.GetRasterBand(2).ReadAsArray()
    b = ds.GetRasterBand(3).ReadAsArray()

    img = np.zeros((r.shape[0],r.shape[1],3))

    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    pngfile = pngfolder + '/' + tifffilename.split('.')[0] + '.png'
    plt.imsave(pngfile, stretch_8bit(img), dpi=dpi)


# convert images from tiff to png format
def convert_all_to_png(tifffolder, pngfolder):
    dpi=800
    # Get all geotiff files
    alltiffs = get_subfiles(tifffolder)
    count = 0
    for tifffilename in alltiffs:
        try:
            saveTopng(tifffolder, pngfolder, tifffilename, dpi=dpi)
        except:
            pass
