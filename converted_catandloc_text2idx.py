import pandas as pd
import numpy as np
import pickle as pkl
import argparse as ap

import os
import re
import string
import html

from collections import defaultdict

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
#nltk.download()
nltk.download("stopwords")

from nltk.corpus import stopwords
 
russian_stopwords = stopwords.words("russian")
print(len(russian_stopwords))
#--------#

from pymystem3 import Mystem
from string import punctuation

source_dir = './'
set1 = 'ItemInfo_train.csv'
matches = 'ItemPairs_train.csv'
verbose = True
destination_path = './set_transfer'
mapping_file = './embedding/fasttext-300.map'
category_file = 'Category.csv'
location_file = 'Location.csv'
val_prop = 0.1
test_prop = 0.1
#

verbose = True

if verbose:
    print('Loading datasets and maps.')
# load data
# df_pos is loaded so that it can be copied to destination directory
# replace the locationID and categoryID by regionID and parentCategoryID respectively
df1 = pd.read_csv(os.path.join(source_dir, set1), encoding = 'utf-8', engine='python', error_bad_lines=False)
df1.insert(1, 'itemID_2', df1['itemID'])
df1.insert(1, 'itemID_1', df1['itemID'])

df_pos = pd.read_csv(os.path.join(source_dir, matches),  error_bad_lines=False)
df_category = pd.read_csv(os.path.join(source_dir, category_file))
df_location = pd.read_csv(os.path.join(source_dir, location_file))

df1 = pd.merge(df1,df_category, how='left',  on=['categoryID'])
df1 = pd.merge(df1,df_location, how='left',  on=['locationID'])

df1 = df1.drop(['categoryID','locationID'], axis='columns')

df1 = df1.rename(columns ={'regionID': 'locationID' , 'parentCategoryID' : 'categoryID' } )

column_idxs = [3, 4]

# load double dictionary
with open( './embedding/fasttext-300.map', 'rb' ) as f:
    map = pkl.load(f)

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
    doc = nlp(text.lower())
#    print(doc.sents.lemma_)
    for s in doc.sents:
        tokens = [t.lemma_ for t in s]


#    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    return text

if verbose:

    print('Cleaning the following columns from set1:')
    for column in df1.columns[column_idxs]:
        print(column, end=' ')
    print()


df1.iloc[:, column_idxs] = df1.iloc[:, column_idxs].applymap(clean_text)
print('clean text finished! ')
df1.iloc[:, column_idxs] = df1.iloc[:, column_idxs].applymap(preprocess_text)
print('preprocess test finished! ')

def record2idx(x):
    x = x.split()
    for i, token in enumerate(x):
        idx = map['word2idx'][token]
        if idx == 0:
            idx = map['word2idx'][token.lower()]
        if idx == 0:
            idx = map['word2idx'][string.capwords(token)]
        if idx == 0:
            idx = map['word2idx'][token.upper()]
        x[i] = idx
    return x

if verbose:
    print('Converting tokens to indices.')
df1.iloc[:, column_idxs] = df1.iloc[:, column_idxs].applymap(record2idx)

if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)
    if verbose:
        print('Creating destination directory')
    
df1.to_csv(os.path.join(dest_dir, 'ItemInfo_train_word2idx.csv' ), index=False)
df_pos.to_csv(os.path.join(dest_dir, matches), index=False)


