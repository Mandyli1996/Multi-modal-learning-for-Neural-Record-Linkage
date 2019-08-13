import sys
import os
import re
sys.path.append('./scripts')

import numpy as np
import pandas as pd
import helpers as hp
import pickle as pkl
import itertools as it
import tensorflow as tf

from keras.layers import Input, Dense, LSTM, Embedding, Lambda, Softmax,Dot, Concatenate, \
                         Bidirectional, BatchNormalization, Dropout,Activation, Multiply, Add
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adadelta
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras import backend as K
from collections import OrderedDict
from keras.callbacks import TensorBoard 



print('deep_er_model gneration..')
from keras.layers import Dense, Activation, Multiply, Add, Lambda
import keras.initializers
    


print('deep_er_model gneration..')
def highway_layers(value, n_layers, activation="tanh", gate_bias=-3):
    dim = K.int_shape(value)[-1]
    gate_bias_initializer = keras.initializers.Constant(gate_bias)
    for i in range(n_layers):     
        gate = Dense(units=dim, bias_initializer=gate_bias_initializer)(value)
        gate = Activation("sigmoid")(gate)
        negated_gate = Lambda(
            lambda x: 1.0 - x,
            output_shape=(dim,))(gate)
        transformed = Dense(units=dim)(value)
        transformed = Activation(activation)(transformed)
        transformed_gated = Multiply()([gate, transformed])
        identity_gated = Multiply()([negated_gate, value])
        value = Add()([transformed_gated, identity_gated])
    return value

def deep_er_model_generator(data_dict,
                            embedding_file,
                            padding_limit = 100,
                            mask_zero = True,
                            embedding_trainable = False,
                            text_columns = list(), 
                            numeric_columns_1D = list(),
                            numeric_columns_2D = list(),
                            category_num_cols = list(),
                            image_url_cols = list(),
                            make_isna = True,
                            text_nan_idx = None,
                            num_nan_val = None,
                            text_compositions = ['decomposable'],
                            text_sim_metrics = ['cosine'],
                            image_sim_metrics = ['cosine'],
                            numeric_sim_metrics = ['unscaled_inverse_lp'],
                            dense_nodes = [10],
                            lstm_args = dict(units=50),
                            document_frequencies = None,
                            idf_smoothing = 2,
                            batch_norm = False,
                            dropout = 0,
                            shared_lstm = True,
                            debug = False):
    """
    Takes a dictionary of paired split DataFrames and returns a DeepER 
    model with data formatted for said model.
    
    Parameters
    ----------
    data_dict : dict
        A dictionary of dataframes (pd.DataFrame) stored with the following
        keys: train_1, val_1, test_1, train_2, val_2, test_2
    embedding_file : str
        The location and name of numpy matrix containing word vector
        embeddings.
    padding_limit : int, optional
        The maximum length of any text sequence. For any text attribute whose
        max length is below padding_limit, the max length will be used.
        Otherwise, padding_limit will be used to both pad and truncuate
        text sequences for that attribute.
    mask_zero : bool, optional
        Whether to ignore text sequence indices with value of 0. Useful for
        LSTM's and variable length inputs.
    embedding_trainable: bool, optional
        Whether to allow the embedding layer to be fine tuned.
    text_columns : list of strings, optional
        A list of names of text-based attributes
    numeric_columns : list of strings, optional
        A list of names of numeric attributes
    make_isna: bool, optional
        Whether to create new attributes indicating the presence of null values
        for each original attribute.
    text_nan_idx : int, optional
        The index corresponding to NaN values in text-based attributes.
    num_nan_val : int, optional
        The value corresponding to NaN values in numeric attributes.
    text_compositions : list of strings, optional
        List of composition methods to be applied to embedded text attributes.
        Valid options are :
            - average : a simple average of all embedded vectors
            - idf : an average of all embedded vectors weighted by normalized
                    inverse document frequency
    text_sim_metrics : list of strings, optional
        List of similarity metrics to be computed for each text-based attribute.
        Valid options are :
            - cosine
            - inverse_l1 : e^-[l1_distance]
            - inverse_l2 : e^-[l2_distance]
    numeric_sim_metrics : list of strings, optional
        List of similarity metrics to be computed for each numeric attribute.
        Valid options are :
            - scaled_inverse_lp : e^[-2(abs_diff)/sum]
            - unscaled_inverse_lp : e^[-abs_diff]
            - min_max_ratio : min / max
    dense_nodes : list of ints, optional
        Specifies topology of hidden dense layers
    lstm_args = dict, optional
        Keyword arguments for LSTM layer
    document_frequencies = tuple of length 2, optional
        Tuple of two lists of document frequencies, left side then right
    idf_smoothing : int, optional
        un-normalized idf = 1 / df ^ (1 / idf_smoothing)
        Higher values means that high document frequency words are penalized
        less.
    """
    
    ### DATA PROCESSING ###
    # initialize an empty dictionary for storing all data
    # dictionary structure will be data[split][side][column]
    sides = ['left', 'right']
    splits = ['train', 'val', 'test']
    data = dict()
    for split in splits:
        data[split] = dict()
        for side in sides:
            data[split][side] = dict()
    
    numeric_columns = numeric_columns_1D+numeric_columns_2D+category_num_cols
    columns = text_columns + numeric_columns+image_url_cols
    
    # separate each feature into its own dictionary entry
    for column in columns:
        data['train']['left'][column] = data_dict['train_1'][column]
        data['train']['right'][column] = data_dict['train_2'][column]

        data['val']['left'][column] = data_dict['val_1'][column]
        data['val']['right'][column] = data_dict['val_2'][column]

        data['test']['left'][column] = data_dict['test_1'][column]
        data['test']['right'][column] = data_dict['test_2'][column]
    import numpy as np
    def load_image(x):
        x = np.load('./imagess'+x[1:]+'.npy')
        return x
    from tqdm import tqdm
 #   for column in image_url_cols:
 #       print('image loadling...')
 #       list_t_l = []
 #       for i in tqdm(list(data['train']['left'][column])):
 #              list_t_l.append( load_image(i) )
 #       data['train']['left'][column] = pd.Series(list_t_l)
 #       #print(data['train']['left'][column][:3], '---------- the first 3', data['train']['left'][column][:1].shape)
 #       print('finished the the loading')     


 #       data['train']['right'][column] = pd.Series([load_image(i) for i in tqdm(list(data['train']['right'][column]))])
#
#        data['test']['left'][column] = pd.Series([load_image(i) for i in tqdm(list(data['test']['left'][column]))])
#        data['test']['right'][column] = pd.Series([load_image(i) for i in tqdm(list(data['test']['right'][column]))])

       
 #       data['val']['left'][column] = pd.Series([load_image(i) for i in tqdm(list(data['val']['left'][column]))])
 #       data['val']['right'][column] = pd.Series([load_image(i) for i in tqdm(list(data['val']['right'][column]))])

       # data['test']['left'][column] = pd.Series([load_image(i) for i in tqdm(list(data['test']['left'][column]))])
       # data['test']['right'][column] = pd.Series([load_image(i) for i in tqdm(list(data['test']['right'][column]))])
    
    # if enabled, create a binary column for each feature indicating whether
    # it contains a missing value. for text data, this will be a list with
    # a single index representing the 'NaN' token. for numeric data, this will
    # likely be a 0.
    if make_isna:
        for split, side, column in it.product(splits, sides, text_columns):
            isna = data[split][side][column].apply(lambda x: x == [text_nan_idx])
            isna = isna.values.astype(np.float32).reshape(-1, 1)
            isna_column = column + '_isna'
            data[split][side][isna_column] = isna
        for split, side, column in it.product(splits, sides, numeric_columns):
            isna = data[split][side][column].apply(lambda x: x == num_nan_val)
            isna_column = column + '_isna'
            isna = isna.values.astype(np.float32).reshape(-1, 1)
            data[split][side][isna_column] = isna
    
    # pad each text column according to the length of its longest entry in
    # both datasets
    maxlen = dict()
    import numpy as np
    for column in text_columns:
        #print(data['train']['left'][column][:20])
        maxlen_left = data['train']['left'][column].apply(lambda x: len(x) if type(x) != float else len([x])).max()
        
        #print(maxlen_left)
        #  data['train']['left'][column].apply(lambda x: print(x) if type(x) != float and len(x)==3151 )
        maxlen_right = data['train']['right'][column].apply(lambda x: len(x) if type(x) != float else len([x])).max()
        print(maxlen_left, maxlen_right )
        maxlength = min(padding_limit, max(maxlen_left, maxlen_right))
        #data[split][side][column] = data[split][side][column].apply(lambda x: [] if x == np.nan else x)
        for split, side in it.product(splits, sides):
            data[split][side][column] = data[split][side][column].apply(lambda x: [] if x == np.nan else x)
            data[split][side][column] = pad_sequences(data[split][side][column],maxlen=maxlength,padding='post',truncating='post')
            
        maxlen[column] = maxlength
    with open('./maxlen.map', 'wb') as f:
        pkl.dump(maxlen, f)
    # convert all numeric features to float and reshape to be 2-dimensional
    numeric_columns = numeric_columns_1D + numeric_columns_2D + category_num_cols
    
    for split, side, column in it.product(splits, sides, numeric_columns):
        feature = data[split][side][column]
        feature = feature.values.astype(np.float32).reshape(-1,1)
        data[split][side][column] = feature
    
    
    
    # format X values for each split as a list of 2-dimensional arrays
    packaged_data = OrderedDict()
    for split in splits:
        packaged_data[split] = list()
        for side, column in it.product(sides, columns):
            packaged_data[split].append(data[split][side][column])
        if make_isna:
            for side, column in it.product(sides, columns):
                packaged_data[split].append(data[split][side][column + '_isna'])
    
    
    # convert y-values
    y_train = to_categorical(data_dict['train_y'])
    y_val = to_categorical(data_dict['val_y'])
    y_test = to_categorical(data_dict['test_y'])
    
    ### MODEL BUILDING ###
    
    # each attribute of each side is its own input tensor
    # text input tensors for both sides are created before numeric input tensors
    input_tensors = dict(left=dict(), right=dict())
    for side, column in it.product(sides, text_columns):
        input_tensors[side][column] = Input(shape=(maxlen[column],))
        
    for side, column in it.product(sides, numeric_columns):
        input_tensors[side][column] = Input(shape=(1,))
    
            
    for side, column in it.product(sides, image_url_cols):
        input_tensors[side][column] = Input(shape=(1,64,2048, ),name='image'+str(side) )
    
        #create embedding layer for image features
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers.core import Flatten

    #CNN_Encoder = Sequential([Flatten(), Dense(embedding_dim), activation='relu'])
    #class CNN_Encoder(tf.keras.Model):
    #    # Since we have already extracted the features and dumped it using pickle
    #    # This encoder passes those features through a Fully connected layer
    #    def __init__(self, embedding_dim):
    #        super(CNN_Encoder, self).__init__()
    #        # shape after fc == (batch_size, 64, embedding_dim)
    #        #self.model = tf.keras.models.Sequential()
    #        #self.model.add(tf.keras.layers.Flatten())
    #        #self.model.add(tf.keras.layers.Dense(embedding_dim, activation='relu'))
    #        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')  
    #    def call(self, x): 
    #        x = tf.keras.layers.Flatten()(x)     
    #        # x = self.model(x)
    #        x = self.fc(x)
    #        return x
        
    highwaynet_img = Lambda(lambda x: highway_layers(x,2))
    
    embedding_layer_image= Sequential([Dense(300, activation='relu'), highwaynet_img])
    
    embedded_tensors = dict(left=dict(), right=dict())
    for side, column in it.product(sides, image_url_cols):
        embedded_tensors[side][column] = embedding_layer_image(input_tensors[side][column])
    
#     similarity_image_layers = list()
#     if 'cosine' in image_sim_metrics:
#         similarity_image_layer = Dot(axes=1, normalize=True)
#         similarity_image_layers.append(similarity_image_layer)
#     if 'inverse_l1' in image_sim_metrics:
#         similarity_image_layer = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))
#         similarity_image_layers.append(similarity_image_layer)
#     if 'inverse_l2' in image_sim_metrics:
#         similarity_image_layer = Lambda(lambda x: \
#                                   K.exp(-K.sqrt(K.sum(K.pow(x[0]-x[1], 2), axis=1, keepdims=True))))
#         similarity_image_layers.append(similarity_image_layer)
    
    # for each attribute, calculate similarities between left and ride sides
    
    similarity_tensors = list()   
    # create a single embedding layer for text input tensors

    highwaynet_sent = Lambda(lambda x: highway_layers(x,2))
   
    embedding_matrix = np.load(embedding_file)
    embedding_layer = Sequential()
    
    embedding_layer.add(Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                trainable=embedding_trainable,
                                mask_zero=mask_zero) )
    embedding_layer.add(highwaynet_sent)
    
    # use embedding_layer ot convert text input tensors to embedded tensors
    # and store in dictionary.
    # an embedding tensor will have shape n_words x n_embedding_dimensions
    #embedded_tensors = dict(left=dict(), right=dict())
    for side, column in it.product(sides, text_columns):
        embedded_tensors[side][column] = embedding_layer(input_tensors[side][column])
    
    text_img_tensors =dict()
    squeeze_layer = Lambda(lambda x: tf.squeeze(x, [1]) )
    reshape_layer = Lambda(lambda x: tf.reshape( x, (-1, 164,300)))
    for side in sides:
        #tf.squeeze(embedded_tensors[side][image_url_cols[0]]) 
        embedded_tensors[side][image_url_cols[0]] = squeeze_layer(embedded_tensors[side][image_url_cols[0]] )
        text_img_tensors[side] = Concatenate(axis=-2)([embedded_tensors[side][text_columns[0]], embedded_tensors[side][image_url_cols[0]] ])
        text_img_tensors[side] = reshape_layer(text_img_tensors[side])   
        
    def xdoty(x, y):
        return tf.matmul(x, tf.transpose(y, [0,2,1]) )
    
    
    score_layer = Lambda(lambda x: xdoty(x[0], x[1]))
    att_tensors =dict()
    #,, 
    left = score_layer( [text_img_tensors['left'],text_img_tensors['right']])
    att_tensors['left'] = Softmax(axis=-1)(left)
    att_tensors['right'] = Softmax(axis=-1)(score_layer([text_img_tensors['right'], text_img_tensors['left']]))
    
    alphas = dict()
    dot_layer = Lambda(lambda x: tf.matmul(x[0], x[1]) )
    alphas['left'] = dot_layer([att_tensors['left'], text_img_tensors['right']])
    alphas['right'] = dot_layer([att_tensors['right'], text_img_tensors['left']])
    
    concat_tensors = dict()
    concat_tensors['left'] = Dense(300)(Concatenate(axis=-1)([text_img_tensors['left'], alphas['left'] ]))
    concat_tensors['right'] = Dense(300)(Concatenate(axis=-1)([text_img_tensors['right'], alphas['right'] ]))
    
    composed_tensors = dict()
    composed_tensors['decomposable'] = dict()
    sum_reduce_layer = Lambda(lambda x: tf.reduce_sum(x, -2))
    composed_tensors['decomposable']['left'] = sum_reduce_layer(concat_tensors['left'])
    composed_tensors['decomposable']['right'] = sum_reduce_layer(concat_tensors['right'])
    
    similarity_layers = list()
    if 'cosine' in text_sim_metrics:
        similarity_layer = Dot(axes=1, normalize=True)
        similarity_layers.append(similarity_layer)
    if 'inverse_l1' in text_sim_metrics:
        similarity_layer = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))
        similarity_layers.append(similarity_layer)
    if 'inverse_l2' in text_sim_metrics:
        similarity_layer = Lambda(lambda x: \
                                  K.exp(-K.sqrt(K.sum(K.pow(x[0]-x[1], 2), axis=1, keepdims=True))))
        similarity_layers.append(similarity_layer)
    if 'learnable_l1' in text_sim_metrics:
        similarity_layer = Lambda(lambda x: K.abs(x[0]-x[1]))
        similarity_layers.append(similarity_layer)
    if 'concat' in text_sim_metrics:
        similarity_layer = Concatenate(axis=-1)
        similarity_layers.append(similarity_layer)    


    # for each attribute, calculate similarities between left and ride sides
    #similarity_tensors = list()
    for composition, similarity_layer in \
        it.product(text_compositions, similarity_layers):        
        similarity_tensor = similarity_layer([composed_tensors[composition]['left'],
                                              composed_tensors[composition]['right']])
        similarity_tensors.append(similarity_tensor)
            
    # reset similarity layer to empty so only numeric-based similarities are used
    similarity_layers = list()
    if 'scaled_inverse_lp' in numeric_sim_metrics:
        similarity_layer = Lambda(lambda x: K.exp(-2 * K.abs(x[0]-x[1]) / (x[0] + x[1] + 1e-5)))
        similarity_layers.append(similarity_layer)
    if 'unscaled_inverse_lp' in numeric_sim_metrics:
        similarity_layer = Lambda(lambda x: K.exp(-K.abs(x[0]-x[1])))
        similarity_layers.append(similarity_layer)
        
    for column, similarity_layer in it.product(numeric_columns, similarity_layers):
        similarity_tensor = similarity_layer([input_tensors['left'][column],
                                              input_tensors['right'][column]])
        similarity_tensors.append(similarity_tensor)
    if 'min_max_ratio' in numeric_sim_metrics:
        for column in numeric_columns:
            num_concat = Concatenate(axis=-1)([input_tensors['left'][column], input_tensors['right'][column]])
            similarity_layer = Lambda(lambda x: K.min(x, axis=1, keepdims=True) / \
                                                (K.max(x, axis=1, keepdims=True) + 1e-5))
            similarity_tensors.append(similarity_layer(num_concat))
    
    # create input tensors from _isna attributes
    input_isna_tensors = list()
    if make_isna:
        for side, column in it.product(sides, columns):
            input_isna_tensors.append(Input(shape=(1,)))
    
    num_dense_inputs = len(similarity_tensors) + len(input_isna_tensors) 
    if 'lstm ' in text_compositions or 'bi_lstm' in text_compositions:
        num_dense_inputs += lstm_args['units'] * len(text_columns)
    print('Number of inputs to dense layer: {}'.format(num_dense_inputs))
    # concatenate similarity tensors with isna_tensors.
    concatenated_tensors = Concatenate(axis=-1)(similarity_tensors + \
                                                input_isna_tensors )
    #x = Concatenate(axis=-1)( [concatenated_tensors]+ [tf.convert_to_tensor([[[1,2],[3,4]]])])
    # create dense layers starting with concatenated tensors
    dense_tensors = [concatenated_tensors]
    for n_nodes in dense_nodes:
        fc = Dense(2*n_nodes, activation='relu', name='output')
        fc_1 = Dense(n_nodes, activation='relu', name='output_2')
        #print(type(fc))
        dense_tensor = fc(dense_tensors[-1])
        dense_tensor = fc_1(dense_tensor)
#        with tf.Session() as sess:
#            print(sess.run(dense_tensor))
#            print(dense_tensor.numpy())
#        print(dense_tensor.numpy())
        if batch_norm and dropout:
            dense_tensor_bn = BatchNormalization(name='batchnormal')(dense_tensor)
            dense_tensor_dropout = Dropout(dropout)(dense_tensor_bn)
            dense_tensors.append(dense_tensor_dropout)
        else:
            dense_tensors.append(dense_tensor)
        dense_tensors.pop(0)
        
    output_tensors = Dense(2, activation='softmax')(dense_tensors[0])
    
    
    #output_tensors = text_img_tensors['left']
    product = list(it.product(sides, columns))
    if not debug:
        model = Model([input_tensors[s][tc] for s, tc in product] + input_isna_tensors,
                      [output_tensors])
    else:
        model = Model([input_tensors[s][tc] for s, tc in product] + input_isna_tensors,
                      [embedded_tensors['left'][text_columns[0]]])
 
    
    return tuple([model] + list(packaged_data.values()) + [y_train, y_val, y_test])


#==========================================================================================


import sys
import os
import re
sys.path.append('../scripts')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import helpers as hp
import pickle as pkl
import itertools as it
import keras


from collections import OrderedDict, defaultdict

from sklearn.metrics import f1_score, precision_score, recall_score,\
                            average_precision_score, roc_auc_score,\
                            roc_curve, precision_recall_curve, confusion_matrix,\
                            accuracy_score, classification_report

#from IPython.core.interactiveshell import InteractiveShell
from matplotlib import rcParams
from importlib import reload
#from model_generator import deep_er_model_generator
pd.options.display.max_colwidth = 1000

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'times new roman'

#%config InlineBackend.figure_format = 'retina'
#%matplotlib inline



reload(hp)

with open('./embedding/fasttext-300.map', 'rb') as f:
    map = pkl.load(f)

print('Data preparation..')
data_dir = os.path.join('.','set_transfer')
source_dir = os.path.join(data_dir)
data = hp.load_data(source_dir)

datasets = ['train_1', 'val_1', 'test_1', 'train_2', 'val_2', 'test_2']

#datasets = ['train_1', 'val_1', 'test_1', 'train_2', 'val_2', 'test_2']

#data['train_2']['price'] = data['train_2']['price'].apply(hp.str_to_num)
#data['val_2']['price'] = data['val_2']['price'].apply(hp.str_to_num)
#data['test_2']['price'] = data['test_2']['price'].apply(hp.str_to_num)

data['train_2']['lat'] = data['train_2']['lat'].apply(hp.str_to_num)
data['val_2']['lat'] = data['val_2']['lat'].apply(hp.str_to_num)
data['test_2']['lat'] = data['test_2']['lat'].apply(hp.str_to_num)

data['train_2']['lon'] = data['train_2']['lon'].apply(hp.str_to_num)
data['val_2']['lon'] = data['val_2']['lon'].apply(hp.str_to_num)
data['test_2']['lon'] = data['test_2']['lon'].apply(hp.str_to_num)

#doc_freqs_1, doc_freqs_2 = hp.get_document_frequencies('../data/converted/amazon-google/', mapping=map)
nan_idx = map['word2idx']['NaN']


print('model established... ')
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',  
                                         histogram_freq=1,  
                                         write_graph=True,  
                                         write_images=True)



histories = dict(acc=list(), val_acc=list(), loss=list(), val_loss=list(), precision=list(), val_precision=list(), f1_score=list(), val_f1_score=list(), auroc =list(), val_auroc=list())

#histories = dict(acc=list(), val_acc=list(), loss=list(), val_loss=list(), precision=list(), val_precision=list(), recall = list(), val_recall=list())
model, X_train, X_val, X_test, y_train, y_val, y_test = \
deep_er_model_generator(data,
                        embedding_file = './embedding/fasttext-300.matrix.npy',
                        text_columns = ['title_clean'],
                        numeric_columns_1D = ['price'],
                        numeric_columns_2D = ['lat', 'lon'],
                        category_num_cols = ['categoryID', 'locationID'],
                        image_url_cols = ['images_array'],
                        text_nan_idx=nan_idx,
                        num_nan_val=0,
                        text_sim_metrics=['cosine'],
                        text_compositions=['decomposable'],
                        numeric_sim_metrics=['scaled_inverse_lp', 'unscaled_inverse_lp', 'min_max_ratio'],
                        dense_nodes=[24],
                        document_frequencies=None,
                        idf_smoothing=2,
                        make_isna=False,
                        embedding_trainable=True,
                        padding_limit=100,
                        batch_norm=True,
                        dropout=0.75,
                        shared_lstm=True,
                        #lstm_args = dict(units=50, dropout=0.25, recurrent_dropout=0.25),
                        lstm_args=dict(units=25)
                        )

print('model complie...')
#sgd = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
import keras_metrics as km
def auroc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', km.binary_precision(),km.f1_score(),auroc ])

print('model fitting...')
import numpy as np
import time
#def load_image(x, y):
#    print(x[13])
#    x[13] = np.load('./imagess'+x[13][1:]+'.npy').reshape(1,64,2048)
#    x[6] = np.load('./imagess'+x[6][1:]+'.npy').reshape(1,64,2048)
#    return x,y
#print('loading X_train images.. ')
#t_train = time.time()
#X_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(load_image)
#print(time.time()-t_train, 'loading X_test images..')
#t_test = time.time()
#X_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).map(load_image)
#print(time.time()-t_test, 'loading X_val inages..')
#X_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(load_image)

#X_train[13] = np.array(list(X_train[13])).reshape(-1,1,64,2048)
#X_train[6] = np.array(list(X_train[6])).reshape(-1,1,64,2048)
#X_test[13] = np.array(list(X_test[13])).reshape(-1,1,64,2048)
#X_test[6] = np.array(list(X_test[6])).reshape(-1,1,64,2048)
#X_val[13] = np.array(list(X_val[13])).reshape(-1,1,64,2048)
#X_val[6] = np.array(list(X_val[6])).reshape(-1,1,64,2048)


import pickle as pkl
with open('./dataset_fit_trainnoimg.map', 'wb') as f:
      pkl.dump(X_train, f)
with open('./dataset_fit_testnoimg.map', 'wb') as f:
      pkl.dump(X_test, f)
with open('./y_fit_testnoimg.map', 'wb') as f:
      pkl.dump(y_test, f)
with open('./y_fit_trainnoimg.map', 'wb') as f:
      pkl.dump(y_train, f)
with open('./y_fit_valnoimg.map', 'wb') as f:
      pkl.dump(y_val, f)
with open('./dataset_fit_valnoimg.map', 'wb') as f:
      pkl.dump(X_val, f)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=100, shuffle=True):
        'Initialization'
        print('here is init__')
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.list_IDs[6] = list(list_IDs[6])
        self.list_IDs[13] = list(list_IDs[13])

    def load_image(self, x):
        x = np.load('./imagess'+x[1:]+'.npy')
        return x

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
    #    print((index+1)*self.batch_size, len(self.labels), '====================')
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = []
        list_y_temp = []
   #     print(len(self.list_IDs), '~~~~~~~~~~~~~~')
        for i in range(len( self.list_IDs)):
             list_IDs_temp.append(np.array( [ self.list_IDs[i][j] for j in indexes ]))
        list_y_temp =np.array([ self.labels[i] for i in indexes ])

       # list_IDs_temp = np.array([self.list_IDs[k] for k in indexes])
      #  list_y_temp = np.array([self.labels[k] for k in indexes])
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_y_temp)

        return list_IDs_temp, list_y_temp

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = list_IDs_temp
        y = list_y_temp

        # Generate data
        #print(X[6], X[6][0])
  #      test = []
  #      for i in X[6]:
  #           print(i)
  #           test.append(self.load_image(i))
        X[6] = np.array([self.load_image(i) for i in X[6]]).reshape(-1,1,64,2048)
        X[13] = np.array([self.load_image(i) for i in X[13]]).reshape(-1,1,64,2048)
        return X, y

training_generator = DataGenerator(X_train, y_train, batch_size=100)
validation_generator = DataGenerator(X_val, y_val, batch_size = 100)
print('len', len(X_train[1]))

earlystop = keras.callbacks.EarlyStopping(monitor='val_auroc', min_delta=0, patience=2, verbose=1, mode='max', baseline=None, restore_best_weights=True)

history = model.fit_generator(generator=training_generator, steps_per_epoch=len(X_train[1])//100 , epochs=10, validation_data=validation_generator, validation_steps = len(X_val[1])//100,
                    workers = 10, callbacks = [earlystop],use_multiprocessing=True, max_queue_size=20, verbose =1, shuffle=True)

#history = model.fit(X_train, y_train, epochs=8, batch_size=100,
#                    validation_data=(X_val, y_val),
#                    shuffle=True,  callbacks = [tbCallBack])

print('history finished.. ')
#with open('./model_decom_img.mod','wb') as f:
#      pkl.dump(model, f)
model.save('./model_decom_img_83.mod')


histories['acc'].extend(history.history['acc'])
histories['val_acc'].extend(history.history['val_acc'])
histories['loss'].extend(history.history['loss'])
histories['val_loss'].extend(history.history['val_loss'])
histories['precision'].extend(history.history['precision'])
histories['f1_score'].extend(history.history['f1_score'])
histories['val_precision'].extend(history.history['val_precision'])
histories['val_f1_score'].extend(history.history['val_f1_score'])
histories['auroc'].extend(history.history['auroc'])
histories['val_auroc'].extend(history.history['val_auroc'])



with open('./history_mymodel_imgdecom.map', 'wb') as f:
    pkl.dump(histories, f)





