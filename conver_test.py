import pandas as pd
import numpy as np
import pickle as pkl
import argparse as aps
from numba import jit

import os
import re
import string
import html

from collections import defaultdict

import nltk
nltk.download("stopwords")
#--------#

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

source_dir = '.'
set1 = 'ItemInfo_train.csv'
matches = 'ItemPairs_train.csv'
verbose = True
dest_dir = './set_transfer'
mapping_file = './embedding/fasttext-300.map'
category_file = 'Category.csv'
location_file = 'Location.csv'
val_prop = 0.1
test_prop = 0.1

verbose = True

if verbose:
    print('Loading datasets and maps.')

# load data
# df_pos is loaded so that it can be copied to destination directory
# replace the locationID and categoryID by regionID and parentCategoryID respectively
df1 = pd.read_csv(os.path.join(source_dir, set1), encoding = 'utf-8', engine='python', error_bad_lines=False)
#df2 = pd.read_csv( './set_transfer/ItemInfo_train_image_change.csv', encoding = 'utf-8', engine='python', error_bad_lines=False)
df_pos = pd.read_csv(os.path.join(source_dir, matches))
df_category = pd.read_csv(os.path.join(source_dir, category_file))
df_location = pd.read_csv(os.path.join(source_dir, location_file))

df1 = pd.merge(df1,df_category, how='left',  on=['categoryID'])
df1 = pd.merge(df1,df_location, how='left',  on=['locationID'])

df1 = df1.drop(['categoryID','locationID'], axis='columns')

df1 = df1.rename(columns ={'regionID': 'locationID' , 'parentCategoryID' : 'categoryID' } )
# , 'itemID_2': 'id2', 'itemID_1': 'id1'
# df_pos = df_pos.rename(columns ={'itemID_2': 'id2', 'itemID_1': 'id1'})

def manufacturer_plus_name(x):
    n = x['title']
    p = x['description']
    if isinstance(n, str) and isinstance(p, str):
        result = n + ' ' + p
    elif isinstance(n, str):
        result = n
    elif isinstance(p, str):
        result = p
    else: 
        result = np.nan
    return result

df1['title'] = df1.apply(manufacturer_plus_name, axis='columns')
df1 = df1.drop('description', axis='columns')
#df1['images_array'] = df2['images_array']
print(df1[df1['itemID'].isin([3507926])], 'test')
#df1 = df1.dropna(subset=['title', 'images_array','price','lat','lon', 'categoryID', 'locationID','attrsJSON'], inplace=True)

# load double dictionary
with open( './embedding/fasttext-300.map', 'rb' ) as f:
    map_1 = pkl.load(f)

def clean_text(x):
    "formats a single string"
    if not isinstance(x, str):
        return 'NaN'
    
    # separate possessives with spaces
    x = x.replace('\'s', ' \'s')
    
    # convert html escape characters to regular characters
    x = html.unescape(x)
    
    # separate punctuations with spaces
    def pad(x):
        match = re.findall(r'.', x[0])[0]
        match_clean = ' ' + match + ' '
        return match_clean
    rx = r'\(|\)|/|!|#|\$|%|&|\\|\*|\+|,|:|;|<|=|>|\?|@|\[|\]|\^|_|{|}|\||'
    rx += r'`|~'
    x = re.sub(rx, pad, x)
    
    # remove decimal parts of version numbers
    def v_int(x):
        return re.sub('\.\d+','',x[0])
    x = re.sub(r'v\d+\.\d+', v_int, x)
    
    return x

#Create lemmatizer and stopwords list

import spacy
nlp = spacy.load('ru2')
nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)

#Preprocess function
def preprocess_text(text):
    if type(text)==np.nan:
        return x
    elif type(text) == float:
        return np.nan

    text = re.sub(r'[:/\n?@*+,-./:;<=>?!"#$%&)(}{_-|}~\t\n]',' ', text)
    doc = nlp(text.lower())
#    print(doc.sents.lemma_)
    for s in doc.sents:
        if len(s)>200:
             tokens = [t.lemma_ for t in s[:200]]
        else:
             tokens = [t.lemma_ for t in s]
    
#     text = "".join(tokens)
    return tokens

column_idxs = [1]

if verbose:
    print('Cleaning the following columns from set1:')
    #for column in df1.columns[column_idxs]:
    #    print(column, end=' ')
    #print()


print('title cleaning and lemmatizer')
from tqdm import tqdm
array = []
#print(df1['title'][-20:])
#print(len(list(df1['title'])), 'list capacity')
#print(len(df1['title']), 'series capacity')
for row in tqdm(list(df1['title'])):
    #print(type(row))
    #df7['ti']
    array.append(preprocess_text(row) if type(row)==str else 'nan')
    #df7['title'][row] = preprocess_text( list(df7['title'])[row] )

df1['title_clean'] = pd.Series(array) 
print('title finished and attr begins')
print(df1[df1['itemID'].isin([3507926])],  'test')

df1.to_csv(os.path.join(dest_dir, 'ItemInfo_train_clean_completed.csv' ), index=False)

import numpy as np
def json2dic(x):
     if type(x)==float:
           return x
     else:
           globals = {'nan': ' nan:\' \' '}
           return ' '.join([v for key,v in eval(str(x),globals).items()])


df1['attrsJSON'] = df1['attrsJSON'].apply(json2dic)

from tqdm import tqdm
array_1 = []
#print(len(df1['title'])==len(list(df1['title']), 'list capacity'))
for row in tqdm(list(df1['attrsJSON'])):
    #print(type(row))
    #df7['ti']
    array_1.append(preprocess_text(row))
df1['attr_clean'] = pd.Series(array_1)

df1.to_csv(os.path.join(dest_dir, 'ItemInfo_train_clean_completed.csv' ), index=False)


def record2idx(x):
    if type(x)!= list:
        return x
    x = ' '.join(x).split()
    #x = x.replace('\'','')
    #x = x.replace(' ','')
    #x = list(map(str,x[1:-1].strip().split(',')))
 #   print(x)
    for i, token in enumerate(x):
        idx = map_1['word2idx'][token]
        if idx == 0:
            idx = map_1['word2idx'][token.lower()]
        if idx == 0:
            idx = map_1['word2idx'][string.capwords(token)]
        if idx == 0:
            idx = map_1['word2idx'][token.upper()]
        x[i] = idx
    return x

#if verbose:
print('Converting tokens to indices.')

print(df1['title_clean'][int(len(df1)/2)-20: int(len(df1)/2)])

import time
f = time.time()
df1['title_clean'] = df1['title_clean'].apply(record2idx)
df1['attr_clean'] = df1['attr_clean'].apply(record2idx)
print(time.time() - f)

df1.to_csv(os.path.join(dest_dir, 'ItemInfo_train_word2idx_completed.csv' ), index=False)
df_pos.to_csv(os.path.join(dest_dir, matches), index=False)







