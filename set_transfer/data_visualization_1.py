import pandas as pd
import os
import numpy as np

source_dir = '.'
set1 = 'ItemInfo_train_word2idx.csv'
set2 = 'ItemInfo_train_image_change.csv'

#pd1 = pd.read_csv(os.path.join(source_dir, set1), encoding = 'utf-8', engine='python', error_bad_lines=False)
pd2 = pd.read_csv(os.path.join(source_dir, set2), encoding = 'utf-8', engine='python', error_bad_lines=False)

#print(pd1[:10])
print(pd2[:10])

#pd1['images_array'] = pd2['images_array']
#pd1.insert(1, 'itemID_1', pd1['itemID'])
#pd1.insert(1, 'itemID_2', pd1['itemID'])

x = sorted(set(list(pd2['images_array'].replace(np.nan, 'nan'))))
#print(x)
x.pop(x.index('nan'))
print(x[101007:101025])
print(sorted(set(list(pd2['images_array'])))[101007:101025])


pd1.to_csv(os.path.join('.', 'ItemInfo_train_completed.csv'), index=False)
print(len(x))
print(pd1.columns)

