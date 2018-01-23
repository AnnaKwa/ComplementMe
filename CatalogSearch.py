import io
import os
import psycopg2
import time
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup as Soup
from random import randint
from selenium import webdriver
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from oauth2client.client import GoogleCredentials
# For querying postgres database
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database



def connect_to_SQLdb(db_name=None, user='postgres', password='postgres', host='/var/run/postgresql', port = '5432' ):
    '''
    Needs database name for arg db_name
    '''
    #engine = create_engine( 'postgresql://{}:{}@{}:{}/{}'.format(user, password, port, db_name, 'localhost') )
    engine = create_engine( 'postgresql://{}:{}@{}:{}/{}'.format(user, password,'localhost', port, db_name) )
    print(engine.url)

    ## create a database (if it doesn't exist)
    if not database_exists(engine.url):
        create_database(engine.url)
    print(database_exists(engine.url))
    if db_name==None:
        raise ValueError('Need database name for arg db_name.')
    con = psycopg2.connect(database = db_name, user = user, password = password, host=host)
    return con


def query_clothing_SQLdb(db_name, con, categories):
    '''
    returns pandas df of inventory items that are in categories missing from input outfit

    '''
    query= "SELECT * FROM {0} WHERE category IN {1} ".format(db_name, tuple(categories))

    data_from_sql = pd.read_sql_query(query,con)
    return data_from_sql


def start_GCV():
    credentials = GoogleCredentials.get_application_default()
    # Instantiates a client
    client = vision.ImageAnnotatorClient()


def load_clothing_dict(dict_path):
    '''
    returns (dictionary, list of unique categories)
    '''
    f=open(dict_path,"r")
    lines=f.readlines()
    specific_items, clothing_categories = [], []
    for x in lines:
        specific_items.append(x.split(' ')[0].strip('\n'))
        clothing_categories.append(x.split(' ')[1].strip('\n'))
    f.close()
    return  dict(zip(specific_items, clothing_categories)), np.unique(clothing_categories).tolist()



def detect_properties(img_location='local', local_path='', url=''):
    """
    Detects image Properties
    specify img_location as 'local' or 'url' and provide arg local_path or url
    """
    client = vision.ImageAnnotatorClient()

    if img_location=='local':
        if len(local_path)==0:
            raise ValueError('Error: need to provide a local path to image file for arg local_path')
            return
        try:
            with io.open(local_path, 'rb') as image_file:
                content = image_file.read()

            image = types.Image(content=content)
        except:
            raise ValueError('Error reading local image file ',local_path,'.')
            return

    elif img_location=='url':
        if len(url)==0:
            raise ValueError('Error: need to provide url of image for arg url')
            return
        try:
            image = types.Image()
            image.source.image_uri = url
        except:
            raise ValueError('Error accessing image url.')
            return

    response = client.image_properties(image=image)
    props = response.image_properties_annotation

    return (props)
    '''
    print('Properties:')

    for color in props.dominant_colors.colors:
        print('fraction: {}'.format(color.pixel_fraction))
        print('\tr: {}'.format(color.color.red))
        print('\tg: {}'.format(color.color.green))
        print('\tb: {}'.format(color.color.blue))
        print('\ta: {}'.format(color.color.alpha))
    '''


def detect_labels(img_location='local', local_path='', url=''):
    """
    Detects image Properties
    specify img_location as 'local' or 'url' and provide arg local_path or url
    """
    client = vision.ImageAnnotatorClient()

    if img_location=='local':
        if len(local_path)==0:
            raise ValueError('Error: need to provide a local path to image file for arg local_path')
            return
        try:
            with io.open(local_path, 'rb') as image_file:
                content = image_file.read()

            image = types.Image(content=content)
        except:
            raise ValueError('Error reading local image file ',local_path,'.')
            return

    elif img_location=='url':
        if len(url)==0:
            raise ValueError('Error: need to provide url of image for arg url')
            return
        try:
            image = types.Image()
            image.source.image_uri = url
        except:
            raise ValueError('Error accessing image url.')
            return

    response = client.label_detection(image=image)
    labels = response.label_annotations

    return labels


def group_returned_inventory(returned_inventory_items):
    '''
    inputs:
    df: pass pandas df of inventory items that were chosen to avoid overlap with the input items' categories
    '''
    unique_returned_categories = np.unique(returned_inventory_items['category'])
    returned_items_grouped=[]
    for category in unique_returned_categories:
        returned_items_grouped.append( returned_inventory_items.loc[ returned_inventory_items['category']==category ])
    return returned_items_grouped


def random_select_from_item_group(item_group, n_selections=3):
    selected_indices=[]
    max_index=len(item_group)-1

    while len(selected_indices)<n_selections:
        random_index=randint(0,max_index)
        print( random_index, selected_indices)
        if random_index not in selected_indices:
            selected_indices.append(random_index)
    return item_group.iloc[selected_indices]


class clothing_item_input:
    def __init__(self, location_type='local', location=None):
        if location==None:
            raise ValueError('Error: Need to provide image location (either local path or url).')
            return
        self.location = location
        self.color_placeholder = (0,0,0)   # RGB representation
        self.location_type=location_type
        self.item_category=None

    def fill_color(self, method='GCV', mask_bg=False):
        self.color = self.color_placeholder
        #if method='GCV':

    def label_item(self, item_dict=None):
        '''
        needs to be passed clothing category dict
        '''
        if item_dict==None:
            raise ValueError('Error: need to pass the clothing category dictionary as arg item_dict.')
            return

        self.labels = detect_labels(img_location=self.location_type, url=self.location, local_path=self.location)

        self.item_categorized=False
        for label in self.labels:
            try:
                self.item_category = item_dict[label.description]
                self.item_label = label.description
                self.item_categorized=True
            except:
                pass
            if self.item_categorized==True:
                break



class item_ensemble_input:
    '''
    need location_type='url' or 'local'
    provide list of local paths or urls
    '''
    def __init__(self, location_type='url', item_locations=[]):

        self.items = []                         # will be filled with clothing_item_input objects
        self.location_type = location_type
        self.item_locations = item_locations

    def label_items(self, item_dict):
        for i, item_location in enumerate(self.item_locations):
            self.items.append( clothing_item_input(location_type=self.location_type, location=item_location) )
            self.items[i].label_item(item_dict=item_dict)

    #def get_label_set(self):

    def find_existing_categories(self):
        self.existing_categories = [item.item_category for item in self.items]
        return(self.existing_categories)

    def find_missing_categories(self, category_list=None):
        '''
        need to provide list of clothing categories (index 1 of return from load_clothing_dict() )
        '''
        if category_list==None:
            raise ValueError('Need to provide list of clothing categories.')
            return
        self.find_existing_categories()
        self.missing_categories = [cat for cat in category_list if cat not in self.existing_categories]



class catalog_request:
    '''
    Needs:
        query_string: split by spaces
        retailer_name: currently working: 'Uniqlo', 'Nordstrom', 'Target'
        num_items: max items to search
        num_items_per_pg: optional arg for retailers that have max viewable items per page
        prettify_and_save: if you want to save the scraped html
        auto_wait: default True, will set wait time based on how long request took
    '''
    def __init__(self, \
                    query_string, \
                    retailer_name, \
                    num_items, \
                    num_items_per_pg=200, \
                    prettify_and_save=False, \
                    auto_wait=True, \
                    auto_wait_factor=3, \
                    wait_time=1,
                    item_dict=None):

        if retailer_name not in ['Uniqlo', 'Target', 'Nordstrom']:
            raise ValueError('Input error: retailer_name must be chose from Nordstrom/Target/Uniqlo.')
        self.retailer = retailer_name
        self.num_items = num_items
        self.query_string = query_string
        self.auto_wait = auto_wait
        self.auto_wait_factor = auto_wait_factor
        self.wait_time = wait_time

        self.page_urls = []
        self.image_urls = []

        self.products_df =  pd.DataFrame(columns=['image_url', 'product_page_url', 'price'])
        self.product_url_placeholder = 'product_url_goes_here'
        self.price_placeholder = 12.12

        self.total_product_count=0
        self.query_complete=False

        if item_dict==None:
            raise ValueError('Need item_dict arg for categorization.')
            return
        self.item_dict = item_dict
        self.categorize_item()

        if self.retailer=='Nordstrom':
            self.num_items_per_pg=num_items_per_pg
        if self.retailer=='Target':
            self.num_items_per_pg=96         # Target has max 96 items per page


    def extract_substring(self, full_str, start_tag, end_tag):
        '''
        extracts substring in a longer string, given the string that precedes it and string that comes after it
        e.g. extract_substring(fullstr='1234thisisthesubstring5678', start_tag='1234', end_tag='5678')
             will return 'thisisthesubstring'
        '''
        return((full_str.split(start_tag))[1].split(end_tag)[0])


    def categorize_item(self):

        list_of_query_strings = self.query_string.split(' ')
        stripped_query = [substr for substr in list_of_query_strings if (substr!='women' and substr!='womens')]
        for label in stripped_query:
            try:
                self.item_category = self.item_dict[label]
            except:
                pass


    def fill_product_info_from_page(self):
        try:
            page_imgs = self.soup.find_all('img')
        except:
            print('Error: Did you web query and soupify yet?')
            return

        self.page_product_count=0
        product_url = self.product_url_placeholder
        price = self.price_placeholder
        self.categorize_item()

        if self.retailer=='Nordstrom':
            for img in page_imgs:
                if 'store/product' in img["src"]:
                    parent_product=(img.parent)
                    product_url='https://nordstrom.com'+parent_product["href"]
                    self.products_df = self.products_df.append({'image_url': img["src"],
                                                'product_page_url': product_url,
                                                'price': price,
                                                'search_keywords': self.query_string,
                                                'category': self.item_category},
                                                ignore_index=True)
                    self.total_product_count+=1
                    self.page_product_count+=1
                    if self.total_product_count>=self.num_items:
                        self.query_complete=True
                        return

        elif self.retailer=='Target':
            for img in page_imgs:
                try:
                    if img['src'].startswith('//target.scene7') and 'fmt=' in img["src"]:
                        self.products_df = self.products_df.append({'image_url': 'https:'+img["src"],
                                                    'product_page_url': product_url,
                                                    'price': price,
                                                    'search_keywords': self.query_string,
                                                    'category': self.item_category},
                                                    ignore_index=True)
                        self.total_product_count+=1
                        self.page_product_count+=1
                        if self.total_product_count>=self.num_items:
                            self.query_complete=True
                            return
                except:
                    pass

        elif self.retailer=='Uniqlo':
            for img in page_imgs:
                try:
                    if 'prod' in img["src"]:
                        self.products_df = self.products_df.append({'image_url': img["src"],
                                                    'product_page_url': product_url,
                                                    'price': price,
                                                    'search_keywords': self.query_string,
                                                    'category': self.item_category},
                                                    ignore_index=True)
                        self.total_product_count+=1
                        self.page_product_count+=1
                        if self.total_product_count>=self.num_items:
                            self.query_complete=True
                            return
                except:
                    pass
            self.query_complete=True

        if self.page_product_count==0:
            self.query_complete=True

    def get_website_urls(self):
        '''
        fills list of urls to query for specific retailers
        '''
        image_type="ActiOn"
        query= self.query_string.split()
        query='+'.join(query)

        if self.retailer=='Nordstrom':
            n_pages = int((self.num_items-1)/self.num_items_per_pg + 1)
            for n in range(n_pages):
                pg_num=n+1
                self.page_urls.append( 'https://shop.nordstrom.com/sr?origin=keywordsearch&keyword=' \
                                        +query+'&page='+str(pg_num)+'&top='+str(self.num_items_per_pg) )
        elif self.retailer=='Target':
            n_pages = int((self.num_items-1)/96 + 1)
            for n in range(n_pages):
                pg_num=n+1
                first_item_on_pg_index = n*96
                self.page_urls.append('https://www.target.com/s?searchTerm='+query+'#sortBy=relevance&Nao='+str(first_item_on_pg_index)+'&limit=96')
        elif self.retailer=='Uniqlo':
            self.page_urls.append( 'https://www.uniqlo.com/us/en/search/?q='+query+'&lang=default' )


    def get_products(self, debug=False, driver_get_wait=False, driver_wait_time=5):
        while self.query_complete==False:
            for n_pg, page_url in enumerate(self.page_urls):
                driver = webdriver.Chrome('./chromedriver')   # pops up selenium window
                if driver_get_wait==True:
                    driver.implicitly_wait(driver_wait_time) # seconds

                if debug==True:
                    print('Working on page ',str(n_pg+1))

                if self.auto_wait==True:
                    t0 = time.time()
                    driver.get(page_url)
                    response_delay = time.time() - t0
                    if debug==True:
                        print('Page response=',str(response_delay),' s')
                else:
                    driver.get(page_url)


                self.soup = Soup(driver.page_source, 'html.parser')

                self.fill_product_info_from_page()
                driver.quit()

                # wait a bit before next page request
                if self.auto_wait==True:
                    time.sleep(self.auto_wait_factor*response_delay)  # wait 10x longer than it took them to respond
                    print('Waiting for ', str(self.auto_wait_factor*response_delay), ' sec')
                else:
                    time.sleep(self.wait_time)

        # stop looking once no product images on page
        print('Done, ', str(self.total_product_count),' products retrieved from ',str(n_pg+1), 'pages.')
