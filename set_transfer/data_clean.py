import pandas as pd
import pickle as pkl
import string
import os
import re

with open( '../embedding/fasttext-300.map', 'rb' ) as f:
    map_1 = pkl.load(f)

#print(map['word2idx'])
df1 = pd.read_csv('ItemInfo_train_clean.csv', encoding = 'utf-8', engine='python', error_bad_lines=False)

def record2idx(x):
    #x = ' '.join(x).split()
    x = x.replace('\'','')
    x = x.replace(' ','')
    x = list(map(str,x[1:-1].strip().split(',')))
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

print(df1['title_clean'][:20])

import time
f = time.time()
df1['title_clean'] = df1['title_clean'].apply(record2idx)
#print( df1['title_clean'][:20].apply(record2idx))
print(time.time() - f)



df1.to_csv(os.path.join('.', 'ItemInfo_train_word2idx.csv' ), index=False)
#df_pos.to_csv(os.path.join(dest_dir, matches), index=False)

#print(df1['title_clean'][:20])
