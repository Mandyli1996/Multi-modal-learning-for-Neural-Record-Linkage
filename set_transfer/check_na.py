import pandas as pd
import numpy as np

ls = ['train_1', 'train_2', 'test_1', 'test_2', 'val_1', 'val_2']
def fillna_price(ls):
     for item in ls:
         original = pd.read_csv(item+'.csv')
         original['price']=original['price'].fillna(original['price'].mean())
         original.to_csv(item+'.csv')
fillna_price(ls)

test_1 = pd.read_csv('test_1.csv')
print(test_1['price'].isna().sum())

#print(original['title_clean'].isna().sum() , 'title')
#print(original['price'].isna().sum() , 'price')
#print(original['lon'].isna().sum(), 'lon')
#print(original[''])
