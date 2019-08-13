import numpy as np
import pandas as pd
df1 = pd.read_csv('train_1.csv')
df2 = pd.read_csv('train_y.csv')
df3 = pd.read_csv('./wrong_data/match_tra.csv')
print(len(df1), len(df2), len(df3))

