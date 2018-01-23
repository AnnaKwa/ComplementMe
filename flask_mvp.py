from flask import Flask, render_template, request
import sys
#import pandas as pd
#sys.path.append('/home/anna/Desktop/insight/project')
from importlib import reload

import CatalogSearch as cs
reload(cs)
import CatalogSearch as  cs

app = Flask(__name__)



@app.route('/')   # route decorator for function below: tells app what url triggers function
def hello_world():
    return( 'Hello, World!')

@app.route('/index')
def my_form_post():
    return render_template('input_form.html')

@app.route('/index', methods=['POST'])
def show_index():
    input_urls_string = str(request.form['input_urls']).replace(' ','')

    input_clothing_urls = input_urls_string.split(',')

    # first take input items and find what categories to return from

    #input_clothing_urls = request.form['text']
    #input_clothing_urls = ['https://n.nordstrommedia.com/ImageGallery/store/product/Zoom/19/_100847459.jpg']
    '''
    input_clothing_urls = ['https://i.pinimg.com/736x/bb/08/c8/bb08c8877de8e055b4ee4c978d7d55ee--plus-size-blouses-womens-blouses.jpg',\
                            'https://target.scene7.com/is/image/Target/52512766?wid=325&hei=325&qlt=80&fmt=pjpeg',
                            'https://i.pinimg.com/736x/b1/69/75/b169752f9cfec3d93045148db95f4d52--winter-hats-for-women-women-hats.jpg']
    '''
    input_ensemble = cs.item_ensemble_input(location_type='url', item_locations=input_clothing_urls)
    item_dict, category_list = cs.load_clothing_dict(dict_path='clothing_category_dict.txt')

    input_ensemble.label_items(item_dict=item_dict)

    exist_cat = input_ensemble.find_existing_categories()

    test_list=[]
    for elem in exist_cat:
        if elem==None:
            test_list.append('none')
        else:
            test_list.append(elem)
    input_cat_text = ' , '.join(test_list)

    input_ensemble.find_missing_categories(category_list=category_list)

    # connect to SQL database and pull inventory from desired categories
    db_name='clothing'
    con = cs.connect_to_SQLdb(db_name=db_name, host='localhost')
    not_in_set = input_ensemble.missing_categories
    returned_inventory_items = cs.query_clothing_SQLdb(db_name=db_name, con=con, categories=not_in_set)

    # separate returned inventory into groups by item category
    grouped_inventory = cs.group_returned_inventory(returned_inventory_items)

    # randomly pull few items from each group
    item_urls_to_show = []
    for item_group in grouped_inventory:
        picked_df =  cs.random_select_from_item_group(item_group, n_selections=3)
        item_urls = [ picked_df.iloc[i]['image_url'] for i in range(len(picked_df)) ]
        item_urls_to_show += item_urls
    output_cat_text=' , '.join(not_in_set)


    return render_template("index.html", input_image_urls= input_clothing_urls, output_image_urls=item_urls_to_show,
                            output_cat_text=output_cat_text ,  input_cat_text=input_cat_text )


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


'''
@app.route('/testfunc')
def func1():
    input_clothing_urls = ['https://i.pinimg.com/736x/bb/08/c8/bb08c8877de8e055b4ee4c978d7d55ee--plus-size-blouses-womens-blouses.jpg',\
                            'https://target.scene7.com/is/image/Target/52512766?wid=325&hei=325&qlt=80&fmt=pjpeg']
    input_ensemble = cs.item_ensemble_input(location_type='url', item_locations=input_clothing_urls)
    item_dict, category_list = cs.load_clothing_dict(dict_path='clothing_category_dict.txt')

    input_ensemble.label_items(item_dict=item_dict)
    input_ensemble.find_existing_categories()
    input_ensemble.find_missing_categories(category_list=category_list)
    return ' '.join(input_ensemble.existing_categories)

@app.route('/testfunc')
def func2():
    return ' '.join(input_ensemble.missing_categories)

    #db_name='clothing'
    #con = cs.connect_to_SQLdb(db_name=db_name, host='localhost')
    #clothing_matches = cs.query_clothing_SQLdb(db_name=db_name, con=con, categories=['accessory_neck','clothing_shirt'])
    #return clothing_matches.to_html()
'''
