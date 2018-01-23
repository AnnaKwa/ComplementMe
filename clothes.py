import numpy as np
import colorsys
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import misc
from sklearn.cluster import KMeans
from functools import partial
from scipy.misc import imresize
import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from oauth2client.client import GoogleCredentials


credentials = GoogleCredentials.get_application_default()
# Instantiates a client
client = vision.ImageAnnotatorClient()

def load_clothing_dict(dict_path):
    f=open(dict_path,"r")
    lines=f.readlines()
    specific_items, clothing_categories = [], []
    for x in lines:
        specific_items.append(x.split(' ')[0])
        clothing_categories.append(x.split('    ')[1])
    f.close()
    return  dict(zip(specific_items, clothing_categories))

def load_image_GCV(img_path):
    # Loads the image into memory
    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    return image

def detect_faces(image):
    '''
    Detects faces in an image.
    Input is the GCV image object from load_image_GCV
    '''
    response = client.face_detection(image=image)
    faces = response.face_annotations

    for face in faces:
        vertices = ([(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])
        upper_left_vertex = vertices[0]
        height=abs(vertices[3][1] - vertices[0][1])
        width=abs(vertices[1][0] - vertices[0][0])
        return upper_left_vertex, height, width #ertices
        #print('face bounds: {}'.format(','.join(vertices)))

def plot_face_rect(image, ULvertex, width, height, flipRGB=False):
    '''
    overlays Google Cloud Vision face detected rectange info onto image
    '''
    rect = patches.Rectangle(ULvertex, width, height, linewidth=1,edgecolor='r',facecolor='none')
    fig,ax = plt.subplots(1)
    if flipRGB==True:
        ax.imshow( BGRtoRGB(image))
    else:
        ax.imshow(image)
    ax.add_patch(rect)
    plt.show()


def BGRtoRGB(img):
    return(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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


def map2color(val, clt):
    '''
    maps cluster number to R,G,B value
    '''
    RGB_centroids=clt.cluster_centers_
    N_colors=len(RGB_centroids)
    labels= list(np.unique(clt.labels_))
    #if val not in labels:
    #    print(val, labels)
    #    raise ValueError('Error: Value not in list of clt labels.')
    ind=labels.index(val)
    return RGB_centroids[ind]



def simple_color_image(clt, img, showProgress=False):
    '''
    write a new image that replaces pixel RGBs with their cluster's centroid color
    '''

    xdim, ydim = np.shape(img)[0], np.shape(img)[1]
    flat_arr = []
    for i,pix in enumerate(clt.labels_):
        print(i)
        if i%100==0:
            clear_output()
        flat_arr.append( list( map2color(pix,clt)) )

    #mapfunc = partial(map2color, clt=clt)
    #mapped=map(mapfunc, clt.labels_)

    simple_img=np.array(flat_arr).reshape(xdim,ydim,3)
    return(simple_img)



def plot_colors(clt, flip_RGB=False):
    centroids = clt.cluster_centers_
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster

    # omits green bg
    unique_labels=np.unique(clt.labels_)
    labels=[]
    RGBcolors=[]

    index_bg_color=None
    for i,centroid in enumerate(centroids):
        #if np.sqrt( centroid[0]**2 + (centroid[1]-255.)**2 + centroid[2]**2 )>5.5:#5.5:
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

    return
