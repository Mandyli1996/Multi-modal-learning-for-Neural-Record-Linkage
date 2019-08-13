import pandas as pd
import numpy as np
import argparse as ap
import os

from collections import defaultdict
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# parser = ap.ArgumentParser()
# parser.add_argument('source_dir',
#                     help='directory containing dataset and match files to split')
# parser.add_argument('dest_dir',
#                     help='directory to save split dataset csvs')
# parser.add_argument('--set1', '-s1', default='set1.csv',
#                     help='filename of first dataset csv')
# parser.add_argument('--set2', '-s2', default='set2.csv',
#                     help='filename of second dataset csv')
# parser.add_argument('--matches', '-m', default='matches.csv',
#                     help='filename of positives matches csv')
# parser.add_argument('--neg_pos_ratio', '-npr', default = 9, type=float,
#                     help='ratio of non-matching pairs to matching pairs')
# parser.add_argument('--val_prop', '-vp', default = 0.1, type=float,
#                     help='proportion of data to allocate to validation set')
# parser.add_argument('--test_prop', '-tp', default = 0.1, type=float,
#                     help='proportion of data to allocate to test set')
# parser.add_argument('--verbose', '-v', action='store_true',
#                     help='print statistics')

# # parse arguments
# args = parser.parse_args()
# source_dir = args.source_dir
# set1 = args.set1
# set2 = args.set2
# matches = args.matches

# destination_path = args.dest_dir

# neg_pos_ratio = args.neg_pos_ratio
# val_prop = args.val_prop
# test_prop = args.test_prop

# verbose = args.verbose
source_dir = './'
set1 = 'set_transfer/ItemInfo_train_completed.csv'
matches = 'set_transfer/ItemPairs_train.csv'
verbose = True
destination_path = './set_transfer'

val_prop = 0.1
test_prop = 0.1
verbose = True


df1 = pd.read_csv(os.path.join(source_dir, set1),  encoding = 'utf-8', engine='python', error_bad_lines=False)
df1['itemID'] = df1['itemID'].astype(str)
df1['itemID_1'] = df1['itemID_1'].astype(str)
df1['itemID_2'] = df1['itemID_2'].astype(str)
# df2 = pd.read_csv(os.path.join(source_dir, set2), encoding = "latin1")
# df2['id2'] = df2['id2'].astype(str)

df_pos = pd.read_csv(os.path.join(source_dir, matches),  encoding = 'utf-8', engine='python', error_bad_lines=False)
df_pos['itemID_1'] = df_pos['itemID_1'].astype(str)
df_pos['itemID_2'] = df_pos['itemID_2'].astype(str)


df = df_pos
# calculate indices on which to split dataset into test and validation sets
test_idx = np.round(len(df) * test_prop).astype(int)
val_idx = np.round(len(df) * (test_prop + val_prop)).astype(int)

df_test = df.iloc[:test_idx,:]
df_val = df.iloc[test_idx:val_idx,:]
df_train = df.iloc[val_idx:,:]

df_train_1 = pd.merge(df_train, df1, how='left',  on=['itemID_1'])
df_train_1 = df_train_1.drop(['itemID_2_x', 'isDuplicate', 'itemID','itemID_2_y'], axis='columns')
df_train_2 = pd.merge(df_train, df1, how='left',  on=['itemID_2'])
df_train_2 = df_train_2.drop(['itemID_1_x', 'isDuplicate',  'itemID', 'itemID_1_y'], axis='columns')



# merge in relevant attributes from each dataset to id's in train, val, ...
# and test set


df_train_y = df_train['isDuplicate']

df_val_1 = pd.merge(df_val, df1, how='left',  on=['itemID_1'])
df_val_1 = df_val_1.drop(['itemID_2_x', 'isDuplicate', 'itemID','itemID_2_y' ], axis='columns')
df_val_2 = pd.merge(df_val, df1, how='left',  on=['itemID_2'])
df_val_2 = df_val_2.drop(['itemID_1_x', 'isDuplicate','itemID','itemID_1_y' ], axis='columns')
df_val_y = df_val['isDuplicate']

df_test_1 = pd.merge(df_test, df1, how='left',  on=['itemID_1'])
df_test_1 = df_test_1.drop(['itemID_2_x', 'isDuplicate', 'itemID', 'itemID_2_y'], axis='columns')
df_test_2 = pd.merge(df_test, df1, how='left',  on=['itemID_2'])
df_test_2 = df_test_2.drop(['itemID_1_x', 'isDuplicate', 'itemID', 'itemID_1_y'], axis='columns')
df_test_y = df_test['isDuplicate']


# ensure all id's match
assert(np.all(df_train_1['itemID_1'].values == df_train['itemID_1'].values))
assert(np.all(df_train_2['itemID_2'].values == df_train['itemID_2'].values))
assert(np.all(df_val_1['itemID_1'].values == df_val['itemID_1'].values))
assert(np.all(df_val_2['itemID_2'].values == df_val['itemID_2'].values))
assert(np.all(df_test_1['itemID_1'].values == df_test['itemID_1'].values))
assert(np.all(df_test_2['itemID_2'].values == df_test['itemID_2'].values))

if verbose:
    print('Training set contains {} instances'.format(len(df_train_y)))
    print('Validation set contains {} instances'.format(len(df_val_y)))
    print('Test set contains {} instances'.format(len(df_test_y)))
    
if not os.path.isdir(destination_path):
    os.mkdir(destination_path)
    if verbose:
        print('Creating destination directory.')
        
# convert 'y' Series to dataframes to avoid header import mismatches
df_train_y = pd.DataFrame(df_train_y)
df_val_y = pd.DataFrame(df_val_y)
df_test_y = pd.DataFrame(df_test_y)

# save newly split dataframes in specified destination
df_train_1.to_csv(os.path.join(destination_path, 'train_1.csv'), index=False,  encoding = 'utf-8')
df_train_2.to_csv(os.path.join(destination_path, 'train_2.csv'), index=False,  encoding = 'utf-8')
df_train_y.to_csv(os.path.join(destination_path, 'train_y.csv'), index=False,  encoding = 'utf-8')

df_val_1.to_csv(os.path.join(destination_path, 'val_1.csv'), index=False,  encoding = 'utf-8')
df_val_2.to_csv(os.path.join(destination_path, 'val_2.csv'), index=False,  encoding = 'utf-8')
df_val_y.to_csv(os.path.join(destination_path, 'val_y.csv'), index=False,  encoding = 'utf-8')

df_test_1.to_csv(os.path.join(destination_path, 'test_1.csv'), index=False,  encoding = 'utf-8')
df_test_2.to_csv(os.path.join(destination_path, 'test_2.csv'), index=False,  encoding = 'utf-8')
df_test_y.to_csv(os.path.join(destination_path, 'test_y.csv'),  encoding = 'utf-8', index=False)



