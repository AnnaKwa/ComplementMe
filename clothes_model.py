import numpy as np
import requests
import cv2

import matplotlib.pyplot as plt

from io import BytesIO
from scipy import misc
from sklearn.cluster import KMeans
from PIL import Image


def downsample(img, percent, max_length=0):
    '''
    reduce image size to percent of original, or
    if max_length is set, scales so that longest side is max_length pixels long
    '''
    if max_length==0:
        return imresize(img, percent)
    else:
        xdim, ydim = np.shape(img)[0], np.shape(img)[1]
        frac=max_length/np.max([xdim,ydim])
        return imresize(img, frac)


def bg_subtract(img, plot=False, include_rect=[.05,.05,1.15,1.6], flipRGB=False):
    #img = cv2.imread(image_file)
    xdim,ydim = np.shape(img)[0], np.shape(img)[1]
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    #rect = (int(.05*xdim), int(.05*xdim), int(1.15*ydim), int(1.6*ydim))
    rect = (int(include_rect[0]*xdim),int(include_rect[1]*xdim), \
            int(include_rect[2]*ydim),int(include_rect[3]*ydim))
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    #mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_masked = img*mask2[:,:,np.newaxis]


    #Get the background
    background = img - img_masked

    #Change all pixels in the background that are not black to white
    background[np.where((background > [0,0,0]).all(axis = 2))] = [0,255,0] #[255,255,255]
    final_img = background + img_masked

    if plot==True:
        plt.axis('off')
        if flipRGB==True:
            plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR), cmap='gray')
        else:
            plt.imshow(final_img)
        plt.show()
    if flipRGB==False:
        return final_img #img
    else:
        return cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)


def color_clusters(img, NumColorCenters=5):
    '''
    Returns the clt object from skLearn's k means clustering
    Input is 2 dim image with RGB tuple for each pixel (3 dim array)

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    '''
    xdim, ydim = np.shape(img)[0], np.shape(img)[1]

    flat_arr = img.reshape(xdim*ydim,3)

    clt = KMeans(n_clusters = NumColorCenters)
    return(clt.fit(flat_arr))


def dist_from_mean(rgb):
    r,g,b=rgb[0], rgb[1], rgb[2]
    mean=np.mean(rgb)
    return(np.sqrt((r-mean)**2+(g-mean)**2+(b-mean)**2 ))



def find_dominant_colors(clt, flip_RGB=False):
    centroids = clt.cluster_centers_
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster

    # omits green bg
    unique_labels=np.unique(clt.labels_)
    labels=[]
    RGBcolors=[]

    index_bg_color=None
    for i,centroid in enumerate(centroids):
        #if np.sqrt( centroid[0]**2 + (centroid[1]-255.)**2 + centroid[2]**2 )>5.5:
            labels.append(unique_labels[i])
        if flip_RGB==True:
            RGBcolors.append(centroid[::-1])
        else:
            RGBcolors.append(centroid)


    numLabels = np.arange(0, len(np.unique(labels)) +1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    #(hist, _) = np.histogram(labels, bins = numLabels)


    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    print (hist)
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster

    for (percent, color) in zip(hist, RGBcolors):
        # plot the relative percentage of each cluster
        print(percent, color)
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)

        startX = endX

    # return the bar chart

    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
