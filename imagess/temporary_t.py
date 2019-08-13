import os
import pickle as pkl
#def file_name(file_dir):
#    for root, dirs, files in os.walk(file_dir):
#        return files

#dic ={}
#print('begin constructing the dir mapping...')
#for i in range(100):
#    a = "./"+str(i)+'/'
#    k = file_name(a)
#    for j in k:
#        dic[j] = str(i)
#with open('./dir_image.map', 'wb') as f:
#    pkl.dump(dic, f)
with open('./dir_image.map', 'rb') as f:
    dic = pkl.load(f)

print('dir mapping is finised and saved suceessfully')


# Import TensorFlow and enable eager execution
# This code requires TensorFlow version >=1.9
import tensorflow.contrib.eager as tfe
import tensorflow as tf
tfe.enable_eager_execution()
# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning

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

df1 =  pd.read_csv('./ItemInfo_train_image_change.csv', encoding = 'utf-8', engine='python', error_bad_lines=False)

list_total = list(df1['images_array'].dropna())

img_name_vector = ' '.join(list_total).split()
for i in img_name_vector:
    if type(i)==float:
        print(i, '=======float')
#img_name_vector = list_total
print('image array is finished')

import tqdm

#2. extract features from the image and then save
def load_image(image_path):
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

#使用xception来做预处理，resize image 为了能够塞入 pretrained models中 之前使用inceptionv3
# getting the unique images
# img_name_vector=img_name_vector[4]
encode_train = sorted(set(img_name_vector))

print('constructing the model...')
image_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                weights='imagenet')
#new_input = image_model.input
#hidden_layer = image_model.layers[-2].output

#image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

image_features_extract_model = image_model

print('constructing the image dataset...')
# feel free to change the batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(
                                encode_train[830296:1011000]).map(load_image).batch(16)

#801000:1011000

import numpy as np
import tensorflow as tf
import time
from tqdm import trange

print('extracting the features from dataset')

iterator = image_dataset.make_one_shot_iterator()
data_size = len(encode_train[830296:1011000])
batch_size = 16
steps_per_epoch = data_size // batch_size if data_size / batch_size == 0 else data_size // batch_size + 1
#next_element = iterator.get_next()
for epoch in range(1):
    tqr = trange(steps_per_epoch, desc="%2d" % (epoch + 1), leave=False)
    #tqr = trange(7198, desc="%2d" % (epoch + 1), leave=False)
    #for _ in tqr:
    #    img, path = iterator.get_next()

    #x = steps_per_epoch - 7198
    #tqr = trange(x, desc="%2d" % (epoch + 1), leave=False)
    for _ in tqr:
        img, path = iterator.get_next()
        path = path +'_new'
        print('path', path)
        batch_features = image_features_extract_model.predict(img)
        #print(tf.shape(batch_features))
        batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1,batch_features.shape[-1]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            #print(path_of_feature)
            np.save(path_of_feature, bf.numpy())

