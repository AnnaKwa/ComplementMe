import colorsys
import cv2
import io
#import matplotlib.pyplot as plt
import numpy as np
import random
import requests
import time
import unittest

from keras.models import load_model
from scipy.misc import imresize
from sklearn.linear_model import LinearRegression
from PIL import Image
from sklearn.cluster import KMeans

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from oauth2client.client import GoogleCredentials


def load_image_GCV(img_path):
    # Loads the image into memory
    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    return image

def single_color_distance(data,model):
    return np.sqrt( (data-model)**2 )

def RGB_distance(data,model):
    '''
    data, model are both tuples (R,G,B)
    '''
    (R1,G1,B1) = data
    (R2,G2,B2) = model
    return np.sqrt( (R1-R2)**2 + (G1-G2)**2 + (B1-B2)**2 )

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


def pick_three_from_palette( c1,c2,c3,c4):
    # c1,...,c4 are one row from x1,...,x4 arrays (where each row c is r,g,b,std of a different image)
    (exclude1, exclude2) = random.sample([0,1,2,3],2)
    colors = [c1,c2,c3,c4]
    set1 = np.array( [list(colors[i]) for i in range(4) if i!=exclude1 ])
    set2 = np.array( [list(colors[i]) for i in range(4) if i!=exclude2 ])
    return( set1, set2)

def pick_two_from_three(setofthree):
    (exclude1, exclude2) =  (random.choice([0,1,2]), random.choice([0,1,2]))

    X1 = [list(setofthree[i]) for i in range(3) if i!=exclude1 ]
    X2 = [list(setofthree[i]) for i in range(3) if i!=exclude2 ]

    # make sure we aren't getting two sets that map same two x1,x2 colors to different y's
    if X1==X2:
        exclude2=(exclude2+1)%3
        X2 = [list(setofthree[i]) for i in range(3) if i!=exclude2 ]

    y1 = setofthree[exclude1]
    y2 = setofthree[exclude2]

    return (np.array(X1), np.array(X2), y1, y2)

def interchange_x_colors(X):
    X_reversed = np.array(list(reversed(X)))
    return (X.reshape(8) , X_reversed.reshape(8))


def random_color():
    return ( random.choice(range(255))/255., random.choice(range(255))/255., random.choice(range(255))/255. )



def get_catalog_match(RGBcolor_predicted, df, category,):
    # RGBcolor_predicted is (r,g,b) of color prediction
    # category must be in RH column of clothing_dict, e.g. accessory_neck, outerwear, shoes, etc.
    # select based on primary color in item


    category_rows = df.loc[ df['category']==category].copy()

    category_rows['rgb_dist_1'] = 999.
    category_rows['rgb_dist_2'] = 999.
    N_inv = len(category_rows)
    '''
    rgb_inv1 = category_rows['rgb1'].values
    r1 = [rgb_inv1[i][0] for i in range(N_inv)]
    b1 = [rgb_inv1[i][1] for i in range(N_inv)]
    g1 = [rgb_inv1[i][2] for i in range(N_inv)]
    rgb_inv2 = category_rows['rgb2'].values
    r2 = [rgb_inv2[i][0] for i in range(N_inv)]
    b2 = [rgb_inv2[i][1] for i in range(N_inv)]
    g2 = [rgb_inv2[i][2] for i in range(N_inv)]
    '''
    #return category_rows

    #for i in range(len(category_rows)):
    for i in category_rows['index']:
        if category_rows.at[i,'percent1'] > 0.5:
            rgb1 = category_rows.at[i,'rgb1']
            r1,g1,b1 = rgb1[0], rgb1[1], rgb1[2]
            category_rows.at[i,'rgb_dist_1'] = RGB_distance((r1, g1, b1) , RGBcolor_predicted)
        elif category_rows.at[i,'percent2'] > 0.5:
            rgb2 = category_rows.at[i,'rgb2']
            r2,g2,b2 = rgb2[0], rgb2[1], rgb2[2]
            category_rows.at[i,'rgb_dist_2'] = RGB_distance((r2, g2, b2) , RGBcolor_predicted)

    if category_rows['rgb_dist_1'].min() < category_rows['rgb_dist_2'].min():
        match_index = category_rows['rgb_dist_1'].argmin()
        min_dist = category_rows['rgb_dist_1'][match_index]
    else:
        match_index = category_rows['rgb_dist_2'].argmin()
        min_dist = category_rows['rgb_dist_2'][match_index]

    return ( category_rows.loc[match_index]['image_url'] , category_rows.loc[match_index]['product_page_url'] ,  min_dist)



class outfit_input:
    def __init__(self, location_type='local', location_path='', upload_file=None):
        if location_type=='local':
            imgobj = Image.open(location_path)
        elif location_type=='url':
            response = requests.get(location_path)
            imgobj = Image.open(io.BytesIO(response.content))
        elif location_type=='upload':
            imgobj = Image.open(io.BytesIO(upload_file))

        self.img_data_full = np.array(imgobj)
        self.imgobj = imgobj

    def write_image(self, save_path='/static/temp.jpg'):
        self.save_path = save_path
        self.imgobj.save( self.save_path )


    def crop_torso(self):
        # needs image to be saved to static directory on machine as temp.jpg
        credentials = GoogleCredentials.get_application_default()
        # Instantiates a client
        client = vision.ImageAnnotatorClient()

        #image = types.Image(content=load_image_GCV('images/test_selfie_1.jpg'))
        response = client.face_detection(image = load_image_GCV('static/temp.jpg'))
        faces = response.face_annotations

        face=faces[0]
        vertices = ([(vertex.x, vertex.y)
                        for vertex in face.bounding_poly.vertices])
        upper_left_vertex = vertices[0]
        height=abs(vertices[3][1] - vertices[0][1])
        width=abs(vertices[1][0] - vertices[0][0])
        (x0,y0), height, width= (upper_left_vertex, height, width) #ertices

        torso_x0, torso_y0, torso_height, torso_width = int(x0-width/4.5), int( y0+1.2*height), int(height*2.5), int(width*1.5)
        self.cropped_torso_img = self.img_data_full[torso_y0:torso_y0+torso_height , torso_x0:torso_x0+torso_width , :]


    def downsample(self, percent=50, max_length=500):
        self.img_data_downsampled = downsample(self.cropped_torso_img, percent=percent, max_length=max_length)


    def get_color_palette(self, nclusters=3, grey_distance_threshold=2.):
        self.num_clusters=nclusters
        color_indices = np.where(np.std(self.img_data_downsampled, axis=2)> grey_distance_threshold)
        x_colors, y_colors = color_indices[0], color_indices[1]
        colored_pixels=[self.img_data_downsampled[x_colors[i]][y_colors[i]] for i in range(len(x_colors))]

        self.clt = KMeans(n_clusters = nclusters)
        self.clt.fit(colored_pixels)
        self.color_palette = self.clt.cluster_centers_

    '''
    def plot_color_palette(self):
        centroids = self.color_palette
        numLabels = np.arange(0,self.num_clusters +1)
        (hist, _) = np.histogram(self.clt.labels_, bins = numLabels)
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
    '''

    def get_two_pairs(self):
        excluded_colors = random.sample(list(self.color_palette), 2)
        self.color_pair_0 = [color for color in self.color_palette if list(color)!=list(excluded_colors[0]) ]
        self.color_pair_1 = [color for color in self.color_palette if list(color)!=list(excluded_colors[1]) ]

    def get_linear_diff_features(self, feature_set='rgb'):
        self.features_linear_diff = []
        #self.keras_input = []
        if feature_set=='rgb':
            for color_pair in [self.color_pair_0, self.color_pair_1]:
                r1, g1, b1 = color_pair[0][0]/255.,  color_pair[0][1]/255.,  color_pair[0][2]/255.
                r2, g2, b2 = color_pair[1][0]/255.,  color_pair[1][1]/255.,  color_pair[1][2]/255.
                std1, std2 = np.std([r1,g1,b1]), np.std([r2,g2,b2])
                self.features_linear_diff.append( np.array( [ r1, g1, b1, std1, (r2-r1), (g2-g1), (b2-b1), (std2-std1) ] ).T )
        elif feature_set=='hsv':
            for c,color_pair in enumerate( [self.color_pair_0, self.color_pair_1] ):
                r1, g1, b1 = color_pair[0][0]/255.,  color_pair[0][1]/255.,  color_pair[0][2]/255.
                r2, g2, b2 = color_pair[1][0]/255.,  color_pair[1][1]/255.,  color_pair[1][2]/255.
                h1, s1, v1 = colorsys.rgb_to_hsv(r1,g1,b1)
                h2, s2, v2 = colorsys.rgb_to_hsv(r2,g2,b2)
                self.features_linear_diff.append(  np.array( [ h1, s1, v1, (h2-h1), (s2-s1), (v2-v1) ] ).T )
                if c==0:
                    self.keras_input0 =  np.array([ h1, s1, v1 , (h2-h1), (s2-s1), (v2-v1) ])
                if c==1:
                    self.keras_input1 =  np.array([ h1, s1, v1 , (h2-h1), (s2-s1), (v2-v1) ])

    def predict_color_matches(self, model_vec):
        # need model_R, model_G, model_B etc.
        model_0, model_1, model_2 = model_vec[0], model_vec[1], model_vec[2]

        ### change to commented section if not using keras model
        self.color_match_0 = [  model_0.predict( self.keras_input0.reshape(1,6) )[0] , \
                                model_1.predict(self.features_linear_diff[0])[0] , \
                                model_2.predict(self.features_linear_diff[0])[0] ]

        self.color_match_1 = [model_0.predict( self.keras_input1.reshape(1,6) ) [0] , \
                                model_1.predict(self.features_linear_diff[1])[0] , \
                                model_2.predict(self.features_linear_diff[1])[0] ]
        '''
        self.color_match_0 = [  model_0.predict(self.features_linear_diff)[0] , \
                                model_1.predict(self.features_linear_diff[0])[0] , \
                                model_2.predict(self.features_linear_diff[0])[0] ]

        self.color_match_1 = [model_0.predict(self.features_linear_diff[1])[0] , \
                                model_1.predict(self.features_linear_diff[1])[0] , \
                                model_2.predict(self.features_linear_diff[1])[0] ]
        '''

class color_dataset:
    def __init__(self, filename):
        self.filename = filename
        self.y_H, self.y_S, self.y_V = [],[],[]
        self.x1_H, self.x1_S, self.x1_V = [],[],[]
        self.x2_H, self.x2_S, self.x2_V = [],[],[]

    def convert_data_to_features(self, debug=False):
        '''
        input file: each row is the 4 color palette for one image,
                    12 columns: r1/g1/b1 ... r4/g4/b4
        '''
        r1,g1,b1, r2,g2,b2, r3,g3,b3, r4,g4,b4 =np.loadtxt(self.filename, unpack=True)

        rgb1 = np.vstack((r1,g1,b1)).T
        rgb2 = np.vstack((r2,g2,b2)).T
        rgb3 = np.vstack((r3,g3,b3)).T
        rgb4 = np.vstack((r4,g4,b4)).T

        std1 = [np.std(color) for color in rgb1]
        std2 = [np.std(color) for color in rgb2]
        std3 = [np.std(color) for color in rgb3]
        std4 = [np.std(color) for color in rgb4]

        x1 = np.vstack( (rgb1.T, std1) ).T
        x2 = np.vstack( (rgb2.T, std2) ).T
        x3 = np.vstack( (rgb3.T, std3) ).T
        x4 = np.vstack( (rgb4.T, std4) ).T

        full_X_list=[]
        full_y_list=[]

        self.set1, self.set2 = [], []
        for i in range(len(x1)):
            set1, set2 = pick_three_from_palette(x1[i], x2[i], x3[i], x4[i])
            self.set1.append(set1)
            self.set2.append(set2)
            for set in [set1, set2]:
                X1, X2, y1, y2 = pick_two_from_three(set)
                full_X_list.append(X1.reshape(8))
                full_y_list.append(y1)
                full_X_list.append(X2.reshape(8))
                full_y_list.append(y2)

        (self.x1_R, self.x1_G, self.x1_B, self.x1_std, self.x2_R, self.x2_G, self.x2_B, self.x2_std) = (np.array(full_X_list)/255.).T
        if  debug==True:
            print(full_X_list[:10])

        (self.y_R, self.y_G, self.y_B, self.y_std) = (np.array(full_y_list)/255.).T

    def RGB_to_HSV(self, debug=False):
        # converts to HSV: colorsys returns H as percent of 360 degrees
        if len(self.y_H)==0:
            for i in range(len(self.y_R)):
                h,s,v = colorsys.rgb_to_hsv(self.y_R[i], self.y_G[i], self.y_B[i])
                self.y_H.append(h)
                self.y_S.append(s)
                self.y_V.append(v)
                if debug==True:
                    if i<10:
                        print('i=',i,'  ,  H,S,V = ' ,h,s,v)
                h,s,v = colorsys.rgb_to_hsv(self.x1_R[i], self.x1_G[i], self.x1_B[i])
                self.x1_H.append(h)
                self.x1_S.append(s)
                self.x1_V.append(v)

                h,s,v = colorsys.rgb_to_hsv(self.x2_R[i], self.x2_G[i], self.x2_B[i])
                self.x2_H.append(h)
                self.x2_S.append(s)
                self.x2_V.append(v)

            self.y_H, self.y_S, self.y_V = np.array(self.y_H), np.array(self.y_S), np.array(self.y_V)
            self.x1_H, self.x1_S, self.x1_V = np.array(self.x1_H), np.array(self.x1_S), np.array(self.x1_V)
            self.x2_H, self.x2_S, self.x2_V = np.array(self.x2_H), np.array(self.x2_S), np.array(self.x2_V)



    def linear_add_features(self, feature_set='rgb'):
        '''
        since X colors 1/2 should be interchangable, can reduce number of features by half
        try: new features = r1+r2, g1+g2 ...
        feature_set = 'rgb', 'hsv', or 'both'
        normalize rgb to be out of maximum 1.0
        '''
        if feature_set=='rgb':
            self.features_linear_addition = 0.5*np.array( ( (self.x1_R + self.x2_R) ,
                                                               (self.x1_G + self.x2_G) ,
                                                               (self.x1_B + self.x2_B) ,
                                                               (self.x1_std + self.x2_std) ) ).T
        elif feature_set=='hsv':
           # omit std of R/G/B here since it is encoded in saturation
           self.features_linear_addition = 0.5*np.array( ( (self.x1_H + self.x2_H ,
                                                            self.x1_S + self.x2_S ,
                                                            self.x1_V + self.x2_V ,) )).T
        elif feature_set=='both':
            self.features_linear_addition = 0.5*np.array( ( (self.x1_R + self.x2_R),
                                                            (self.x1_G + self.x2_G),
                                                            (self.x1_B + self.x2_B),
                                                            self.x1_H + self.x2_H ,
                                                             self.x1_S + self.x2_S ,
                                                             self.x1_V + self.x2_V ,) ).T
        else:
            raise ValueError('Need to specify arg feature_set as rgb/hsv/both')



    def linear_feature_differences(self, feature_set='rgb'):
        '''
        to preserve information about individual colors, try features
        feature_set = 'rgb', 'hsv', or 'both'
        r1, g1, b1, (r2-r1), (g2-g1), (b2-b1)
        h1, s1, v1, (h2-h1), (s2-s1), (v2-v1)
        '''
        if feature_set=='rgb':
            self.features_linear_diff = np.array( ( self.x1_R,
                                                    self.x1_G,
                                                    self.x1_B,
                                                    self.x1_std,
                                                    (self.x2_R-self.x1_R),
                                                    (self.x2_G-self.x1_G),
                                                    (self.x2_B-self.x1_B),
                                                    (self.x2_std-self.x1_std)) ).T

        elif feature_set=='hsv':
            self.features_linear_diff = np.array( ( self.x1_H,
                                                    self.x1_S,
                                                    self.x1_V,
                                                    (self.x2_H-self.x1_H),
                                                    (self.x2_S-self.x1_S),
                                                    (self.x2_V-self.x1_V)) ).T
        elif feature_set=='both':
            self.features_linear_diff = np.array( ( self.x1_R,
                                                    self.x1_G,
                                                    self.x1_B,
                                                    (self.x2_R-self.x1_R),
                                                    (self.x2_G-self.x1_G),
                                                    (self.x2_B-self.x1_B),
                                                    self.x1_H,
                                                    self.x1_S,
                                                    self.x1_V,
                                                    (self.x2_H-self.x1_H),
                                                    (self.x2_S-self.x1_S),
                                                    (self.x2_V-self.x1_V) ) ).T




    def quadratic_add_features(self):
        self.features_quadratic_addition = np.array( ( np.sqrt(self.x1_R**2 + self.x2_R**2) ,
           np.sqrt(self.x1_G**2 + self.x2_G**2) ,
           np.sqrt(self.x1_B**2 + self.x2_B**2) ,
           np.sqrt(self.x1_std**2 + self.x2_std**2) ) ).T
