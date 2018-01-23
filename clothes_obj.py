import numpy as np
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


def start_GCV():
    credentials = GoogleCredentials.get_application_default()
    # Instantiates a client
    client = vision.ImageAnnotatorClient()


class fullImage:
    def __init__(self, image_path):
        self.img0 = cv2.imread(image_path)
        self.xdim0, self.ydim0 = np.shape(self.img0)[0], np.shape(self.img0)[1]

    def detect_faces(self):
        '''
        Detects faces in an image.
        Input is the GCV image object from load_image_GCV
        '''
        face_response = client.face_detection(image=self.img0)
        faces = face_response.face_annotations

        self.face_rect_locations=[]
        for face in self.faces:
            vertices = ([(vertex.x, vertex.y)
                        for vertex in faces.bounding_poly.vertices])
            upper_left_vertex = vertices[0]
            height=abs(vertices[3][1] - vertices[0][1])
            width=abs(vertices[1][0] - vertices[0][0])
            self.face_rect_locations.append( [ upper_left_vertex, height, width ] )
