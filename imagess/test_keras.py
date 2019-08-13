#import TensorFlow and enable eager execution
# This code requires TensorFlow version >=1.9
import tensorflow.contrib.eager as tfe
import tensorflow as tf
tfe.enable_eager_execution()
# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
#import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import backend as K


#from tensorflow.keras.engine.topology import Layer
import numpy as np
import pandas as pd
import re
import numpy as np
import os
import time
import json
from glob import glob
#from PIL import Image
#import image
import pickle
import socket
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd 

#print('models...')
image_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, 
                                                weights='imagenet')
source_dir = '.'
set1 = 'ItemInfo_train_image.csv'

print('loading the infotrain_dataset...')
df1 = pd.read_csv(os.path.join(source_dir, set1), encoding = 'utf-8', engine='python', error_bad_lines=False)

#df1['images_array']
list_total = sorted(list(df1['images_array']))
list_total.index('nan')



#print('models finished..')



