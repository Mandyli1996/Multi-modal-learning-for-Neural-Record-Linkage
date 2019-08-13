import pandas as pd
import numpy as np

original = pd.read_csv('./ItemInfo_train.csv', encoding = 'utf-8', engine='python', error_bad_lines=False)

print(original[original['itemID'].isin([5637025])])
