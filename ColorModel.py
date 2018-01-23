import numpy as np
import random
import time
import unittest

from sklearn.linear_model import LinearRegression

def single_color_distance(data,model):
    return np.sqrt( (data-model)**2 )

def RGB_distance(data,model):
    '''
    data, model are both tuples (R,G,B)
    '''
    (R1,G1,B1) = data
    (R2,G2,B2) = model
    return np.sqrt( (R1-R2)**2 + (G1-G2)**2 + (B1-B2)**2 )


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


def random_RGB_color():
    return ( random.choice(range(255)), random.choice(range(255)), random.choice(range(255)) )




class color_dataset:
    def __init__(self, filename):
        self.filename = filename

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

        (self.x1_R, self.x1_G, self.x1_B, self.x1_std, self.x2_R, self.x2_G, self.x2_B, self.x2_std) = np.array(full_X_list).T
        if  debug==True:
            print(full_X_list[:10])

        (self.y_R, self.y_G, self.y_B, self.y_std) = np.array(full_y_list).T


    def linear_add_features(self):
        '''
        since X colors 1/2 should be interchangable, can reduce number of features by half
        try: new features = r1+r2, g1+g2 ...
        '''
        self.features_linear_addition = np.array( (self.x1_R + self.x2_R ,
           self.x1_G + self.x2_G ,
           self.x1_B + self.x2_B ,
           self.x1_std + self.x2_std) ).T

    def quadratic_add_features(self):
        self.features_quadratic_addition = np.array( ( np.sqrt(self.x1_R**2 + self.x2_R**2) ,
           np.sqrt(self.x1_G**2 + self.x2_G**2) ,
           np.sqrt(self.x1_B**2 + self.x2_B**2) ,
           np.sqrt(self.x1_std**2 + self.x2_std**2) ) ).T
