import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests

from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans



def load_img_from_url(img_url):
    '''
    reads image from url and returns it as a numpy array of RGB values
    '''
    response = requests.get(img_url)
    imgobj = Image.open(BytesIO(response.content))
    return np.array(imgobj)


def extract_colored_pixels(img_data, grey_distance_threshold=15.):
    '''
    takes in X x Y x (RGB) np array of image, flattens
    returns 1D array of pixels with stdevs from mean value > threshold (since greyscale has R=G=B)
    '''
    color_indices = np.where(np.std(img_data,axis=2) > grey_distance_threshold)
    x_colors, y_colors = color_indices[0], color_indices[1]
    colored_pixels=[img_data[x_colors[i]][y_colors[i]] for i in range(len(x_colors))]
    return colored_pixels


def get_color_palette(img, num_clusters=4):
    '''
    returns top 4 colors in picture
    ignoring percentage of each color for now
    '''
    clt = KMeans(n_clusters = num_clusters)
    clt.fit(img)
    centroids = clt.cluster_centers_

    '''
    numLabels = np.arange(0,num_clusters +1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    '''
    return centroids


def plot_palette(hist, centroids):
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster

    for (percent, color) in zip(hist,centroids):
        # plot the relative percentage of each cluster
        print(percent,color)
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)

        startX = endX

    # return the bar chart

    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
