from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from importlib import reload

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from oauth2client.client import GoogleCredentials

import colorsys
import pandas as pd
import pickle
import numpy as np
import os
import random

import ColorModel
#import CatalogSearch


app = Flask(__name__)


credentials = GoogleCredentials.get_application_default()
# Instantiates a client
client = vision.ImageAnnotatorClient()


@app.route('/', methods=['GET'])
def my_form_post():
    method = request.method

    if method=='GET':
        return render_template('input_form.html')

        if file:
            return redirect('/submission')

@app.route('/noface')
def face_error():
    return render_template("face_error.html")


@app.route('/quickdemo')
def quick_demo():
    predictor_set='hsv'

    input_outfit = ColorModel.outfit_input(location_type='local', location_path='static/test_selfie_1.jpg')

    ### detect face, crop out torso region, and downsample
    input_outfit.write_image()
    input_img_file = input_outfit.save_name
    input_outfit.crop_torso()
    if input_outfit.face_detected==False:
        return redirect('/noface')
    input_outfit.downsample()

    ### get color palette based on downsampled torso region
    input_outfit.get_color_palette(grey_distance_threshold=20.)
    input_outfit.get_two_pairs()
    input_outfit.get_linear_diff_features(feature_set=predictor_set)

    inventory_df = pd.read_pickle('full_inventory.p')

    if predictor_set=='rgb':
        model_filename = 'regression_model_linearDiff_RGBpredictors.p'
        model_vec = pickle.load(open('models/'+model_filename, 'rb'))  # [model_R, model_G, model_B]

    elif predictor_set=='hsv':
        #model_H = load_model('models/keras_linearDiff_HSVpredictors_H_1node.h5')
        temp_model_vec = pickle.load(open('models/boostedTrees_model_linearDiff_HSVpredictors.p', 'rb'))
        model_H, model_S, model_V = temp_model_vec[0], temp_model_vec[1], temp_model_vec[2]
        model_vec = [model_H, model_S, model_V]

    input_outfit.predict_color_matches(model_vec)


    #print(input_outfit.color_pair_0, input_outfit.color_pair_1)

    if predictor_set=='hsv':
        h0,s0,v0 = input_outfit.color_match_0
        h1,s1,v1 = input_outfit.color_match_1
        predicted_color_0 = colorsys.hsv_to_rgb(h0,s0,v0)
        predicted_color_1 = colorsys.hsv_to_rgb(h1,s1,v1)
    else:
        predicted_color_0, predicted_color_1 = input_outfit.color_match_0, input_outfit.color_match_1

    # for now exclude shirts and pants
    return_categories = ['accessory_bag', 'accessory_head', 'accessory_neck', 'clothing_sweater', 'outerwear', 'shoes']
    return_img_urls, return_product_urls = [], []
    for category in return_categories:
        match_img_url_0, match_product_url_0, match_dist_0 = ColorModel.get_catalog_match(predicted_color_0, inventory_df, category=category)
        match_img_url_1, match_product_url_1, match_dist_1 = ColorModel.get_catalog_match(predicted_color_1, inventory_df, category=category)

        random_pick = random.randint(0,1)
        if random_pick==0:
            return_img_urls.append( match_img_url_0)
            return_product_urls.append( match_product_url_0)
        else:
            return_img_urls.append( match_img_url_1)
            return_product_urls.append( match_product_url_1)


    html_urls = zip(range(len(return_img_urls)), return_product_urls, return_img_urls)
    return render_template("index_new.html",  input_img_file = input_img_file, product_urls=html_urls)



@app.route('/submission', methods=['POST'])
def submission():
    predictor_set='hsv'

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    newfile = file.read()

    input_outfit = ColorModel.outfit_input(location_type='upload', upload_file=newfile)

    ### detect face, crop out torso region, and downsample
    input_outfit.write_image()
    input_img_file = input_outfit.save_name
    input_outfit.crop_torso()
    if input_outfit.face_detected==False:
        return redirect('/noface')
    input_outfit.downsample()

    ### get color palette based on downsampled torso region
    input_outfit.get_color_palette(grey_distance_threshold=20.)
    input_outfit.get_two_pairs()
    input_outfit.get_linear_diff_features(feature_set=predictor_set)

    inventory_df = pd.read_pickle('full_inventory.p')

    if predictor_set=='rgb':
        model_filename = 'regression_model_linearDiff_RGBpredictors.p'
        model_vec = pickle.load(open('models/'+model_filename, 'rb'))  # [model_R, model_G, model_B]

    elif predictor_set=='hsv':
        #model_H = load_model('models/keras_linearDiff_HSVpredictors_H_1node.h5')
        temp_model_vec = pickle.load(open('models/boostedTrees_model_linearDiff_HSVpredictors.p', 'rb'))
        model_H, model_S, model_V = temp_model_vec[0], temp_model_vec[1], temp_model_vec[2]
        model_vec = [model_H, model_S, model_V]

    input_outfit.predict_color_matches(model_vec)


    #print(input_outfit.color_pair_0, input_outfit.color_pair_1)

    if predictor_set=='hsv':
        h0,s0,v0 = input_outfit.color_match_0
        h1,s1,v1 = input_outfit.color_match_1
        predicted_color_0 = colorsys.hsv_to_rgb(h0,s0,v0)
        predicted_color_1 = colorsys.hsv_to_rgb(h1,s1,v1)
    else:
        predicted_color_0, predicted_color_1 = input_outfit.color_match_0, input_outfit.color_match_1

    # for now exclude shirts and pants
    return_categories = ['accessory_bag', 'accessory_head', 'accessory_neck', 'clothing_sweater', 'outerwear', 'shoes']
    return_img_urls, return_product_urls = [], []
    for category in return_categories:
        match_img_url_0, match_product_url_0, match_dist_0 = ColorModel.get_catalog_match(predicted_color_0, inventory_df, category=category)
        match_img_url_1, match_product_url_1, match_dist_1 = ColorModel.get_catalog_match(predicted_color_1, inventory_df, category=category)

        random_pick = random.randint(0,1)
        if random_pick==0:
            return_img_urls.append( match_img_url_0)
            return_product_urls.append( match_product_url_0)
        else:
            return_img_urls.append( match_img_url_1)
            return_product_urls.append( match_product_url_1)


    html_urls = zip(range(len(return_img_urls)), return_product_urls, return_img_urls)
    return render_template("index_new.html",  input_img_file = input_img_file, product_urls=html_urls)
