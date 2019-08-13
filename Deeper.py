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

from keras.layers import Input, Dense, LSTM, Embedding, Lambda, Dot, Concatenate, \
                         Bidirectional, BatchNormalization, Dropout
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adadelta
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras import backend as K
from collections import OrderedDict, defaultdict
from keras.callbacks import TensorBoard 



print('deep_er_model gneration..')
def deep_er_model_generator(data_dict,
                            embedding_file,
                            padding_limit = 100,
                            mask_zero = True,
                            embedding_trainable = False,
                            text_columns = list(), 
                            numeric_columns = list(),
                            make_isna = True,
                            text_nan_idx = None,
                            num_nan_val = None,
                            text_compositions = ['average'],
                            text_sim_metrics = ['cosine'],
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
            
    columns = text_columns + numeric_columns
    
    # separate each feature into its own dictionary entry
    for column in columns:
        data['train']['left'][column] = data_dict['train_1'][column]
        data['train']['right'][column] = data_dict['train_2'][column]

        data['val']['left'][column] = data_dict['val_1'][column]
        data['val']['right'][column] = data_dict['val_2'][column]

        data['test']['left'][column] = data_dict['test_1'][column][:500]
        data['test']['right'][column] = data_dict['test_2'][column][:500]
    
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
        print(data['train']['left'][column][:20])
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
    
    # convert all numeric features to float and reshape to be 2-dimensional
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
    y_test = to_categorical(data_dict['test_y'])[:500]
    
    data_train = data['train']
    data_test = data['test']
    data_val = data['val']


    ### MODEL BUILDING ###
    
    # each attribute of each side is its own input tensor
    # text input tensors for both sides are created before numeric input tensors
    input_tensors = dict(left=dict(), right=dict())
    for side, column in it.product(sides, text_columns):
        input_tensors[side][column] = Input(shape=(maxlen[column],))
        
    for side, column in it.product(sides, numeric_columns):
        input_tensors[side][column] = Input(shape=(1,))
    
    # create a single embedding layer for text input tensors
    embedding_matrix = np.load(embedding_file)
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                trainable=embedding_trainable,
                                mask_zero=mask_zero)
    
    
    # use embedding_layer ot convert text input tensors to embedded tensors
    # and store in dictionary.
    # an embedding tensor will have shape n_words x n_embedding_dimensions
    embedded_tensors = dict(left=dict(), right=dict())
    for side, column in it.product(sides, text_columns):
        embedded_tensors[side][column] = embedding_layer(input_tensors[side][column])
    
    # initialize dictionary for storing a composition tensor for each embedding tensor
    composed_tensors = dict()
    for composition in text_compositions:
        composed_tensors[composition] = dict()
        for side in sides:
            composed_tensors[composition][side] = dict()
    
    # if enabled, reduce each embedding tensor to a quasi-1-dimensional tensor
    # with shape 1 x n_embedding_dimensions by averaging all embeddings
    if 'average' in text_compositions:
        averaging_layer = Lambda(lambda x: K.mean(x, axis=1), output_shape=(maxlen[column],))
        for side, column in it.product(sides, text_columns):
            composed_tensors['average'][side][column] = averaging_layer(embedded_tensors[side][column])
    
    # if enabled, reduce each embedding tensor to a quasi-1-dimensional tensor
    # with shape 1 x n_embedding_dimensions by taking a weighted average of all
    # embeddings. 
    if 'idf' in text_compositions:
        # store document frequency constants for each side
        dfs_constant = dict()
        dfs_constant['left'] = K.constant(document_frequencies[0])
        dfs_constant['right'] = K.constant(document_frequencies[1])
        
        # a selection layer uses an input tensor as indices to select
        # document frequencies from dfs_constant
        dfs_selection_layer = dict()
        
        # a conversion layer converts a tensor of selected document frequencies
        # to a tensor of inverse document frequencies. the larger the DF,
        # the smaller the inverse, the smallness of which is controlled by
        # idf_smoothing
        idf_conversion_layer = Lambda(lambda x: 1 / (K.pow(x, 1/idf_smoothing)))
        
        # document frequencies of 0 will result in IDF's of inf. these should
        # be converted back to 0's.
        idf_fix_layer = Lambda(lambda x: tf.where(tf.is_inf(x), tf.zeros_like(x), x))
        
        # for each IDF tensor, scale its values so they sum to 1
        idf_normalization_layer = Lambda(lambda x: x / K.expand_dims(K.sum(x, axis=1), axis=1))
        
        # take dot product between embedding tensor vectors and IDF weights
        dot_layer = Dot(axes=1)
        
        for side in sides:
            dfs_selection_layer[side] = Lambda(lambda x: K.gather(dfs_constant[side], K.cast(x, tf.int32)))
            for column in text_columns:                
                dfs_tensor = dfs_selection_layer[side](input_tensors[side][column])
                idfs_tensor = idf_conversion_layer(dfs_tensor)
                idfs_tensor_fixed = idf_fix_layer(idfs_tensor)
                idfs_tensor_normalized = idf_normalization_layer(idfs_tensor_fixed)
                composed_tensors['idf'][side][column] = dot_layer([embedded_tensors[side][column],
                                                                   idfs_tensor_normalized])
                
    # if enabled, compose embedding tensor using shared LSTM        
    if 'lstm' in text_compositions:
        if shared_lstm:
            lstm_layer = LSTM(**lstm_args)
        for side, column in it.product(sides, text_columns):
            if not shared_lstm:
                lstm_layer = LSTM(**lstm_args)
            composed_tensors['lstm'][side][column] = lstm_layer(embedded_tensors[side][column])    
    # if enambled, compose embedding tensor using bi-directional LSTM
    if 'bi_lstm' in text_compositions:
        if shared_lstm:
            lstm_layer = lstm_layer = Bidirectional(LSTM(**lstm_args), merge_mode='concat')
        for side, column in it.product(sides, text_columns):
            if not shared_lstm:
                lstm_layer = Bidirectional(LSTM(**lstm_args), merge_mode='concat')
            composed_tensors['bi_lstm'][side][column] = lstm_layer(embedded_tensors[side][column])
    
    # maintain list of text-based similarities to calculate
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
    
    # for each attribute, calculate similarities between left and ride sides
    similarity_tensors = list()
    for composition, column, similarity_layer in \
        it.product(text_compositions, text_columns, similarity_layers):        
        similarity_tensor = similarity_layer([composed_tensors[composition]['left'][column],
                                              composed_tensors[composition]['right'][column]])
        similarity_tensors.append(similarity_tensor)
        
    if 'bi_lstm' in text_compositions:
        difference_layer = Lambda(lambda x: K.abs(x[0]-x[1]))
        hadamard_layer = Lambda(lambda x: x[0] * x[1])
        for column in text_columns:
            difference_tensor = difference_layer([composed_tensors['bi_lstm']['left'][column],
                                                  composed_tensors['bi_lstm']['right'][column]])
            hadamard_tensor = hadamard_layer([composed_tensors['bi_lstm']['left'][column],
                                              composed_tensors['bi_lstm']['right'][column]])
            similarity_tensors.extend([difference_tensor, hadamard_tensor])
    
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
                                                input_isna_tensors)
    
    # create dense layers starting with concatenated tensors
    dense_tensors = [concatenated_tensors]
    for n_nodes in dense_nodes:
        fc = Dense(n_nodes, activation='relu', name='output')
        print(type(fc))
        dense_tensor = fc(dense_tensors[-1])
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
        
    output_tensors = Dense(2, activation='softmax')(dense_tensors[-1])
    
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
#import matplotlib.pyplot as plt
#import seaborn as sns
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
#from matplotlib import rcParams
#from importlib import reload
#from model_generator import deep_er_model_generator
pd.options.display.max_colwidth = 1000

#rcParams['font.family'] = 'serif'
#rcParams['font.serif'] = 'times new roman'

#%config InlineBackend.figure_format = 'retina'
#%matplotlib inline



#reload(hp)

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

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=1, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        list_IDs_temp = []
        list_y_temp = []
        #print(len(self.list_IDs), '~~~~~~~~~~~~~~')
        for i in range(len( self.list_IDs)):
             list_IDs_temp.append( self.list_IDs[i][index*self.batch_size:(index+1)*self.batch_size])
        list_y_temp = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        #X, y = self.__data_generation(list_IDs_temp, list_y_temp)

        return list_IDs_temp, list_y_temp

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = list_IDs_temp
        y = list_y_temp
        X[6] = self.list_IDs_temp[6].apply(load_image)
        X[13] = self.list_y_temp[13].apply(load_image)
        X[13] = np.array(list(X_train[13])).reshape(-1,1,64,2048)
        X[6] = np.array(list(X_train[6])).reshape(-1,1,64,2048)

        return X, y
#text_sim_metrics = ['lstm']
text_sim_metrics=['cosine']
#text_sim_metrics= ['cosine', 'inverse_l1']
#text_sim_metrics= ['learnable_l1']
print(text_sim_metrics, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
histories = dict(acc=list(), val_acc=list(), loss=list(), val_loss=list(), precision=list(), val_precision=list(),  f1_score= list(), val_f1_score=list(), auroc=list(), val_auroc=list() )
model, X_train, X_val, X_test, y_train, y_val, y_test = \
deep_er_model_generator(data,
                        embedding_file = './embedding/fasttext-300.matrix.npy',
                        text_columns = ['title_clean'],
                        numeric_columns = ['price','lat', 'lon', 'categoryID', 'locationID'],
                        text_nan_idx=nan_idx,
                        num_nan_val=0,
                        text_sim_metrics= text_sim_metrics,
                        text_compositions=['average'],
                        numeric_sim_metrics=['scaled_inverse_lp', 'unscaled_inverse_lp', 'min_max_ratio'],
                        dense_nodes=[25],
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
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', km.binary_precision(),km.f1_score(),auroc])


#adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#import keras_metrics as km
#model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', km.binary_precision(), km.binary_recall()])

print('model fitting...')
training_generator = DataGenerator(X_train, y_train, batch_size=100)
validation_generator = DataGenerator(X_val, y_val, batch_size = 100)
print('len', len(X_train[1]))

batch_size = 100
epochs = 7
earlystop = keras.callbacks.EarlyStopping(monitor='val_auroc', min_delta=0, patience=2, verbose=0, mode='max', baseline=None, restore_best_weights=True)
#checkpoint = keras.callbacks.ModelCheckpoint('./checkpoints/'+'baseline'+text_sim_metrics[0] +'.hdf5', monitor='val_auroc', verbose=1, save_best_only=True, mode='max')
history = model.fit_generator(generator=training_generator, steps_per_epoch=len(X_train[1])//batch_size , epochs= epochs, validation_data=validation_generator, validation_steps = len(X_val[1])//batch_size,
                        workers = 16, use_multiprocessing = True, max_queue_size= 40, verbose =1, shuffle=True, callbacks = [earlystop])

generator = DataGenerator(X_test, y_test, batch_size=100)
prediction = model.predict_generator(generator, steps=len(X_test[1])//100, max_queue_size=10, workers=7, use_multiprocessing=True, verbose=0)
label = y_test
dic = {}
dic['prediction'] = prediction
dic['label'] = label
dic['data'] = X_test

with open('./data_baseline'+ text_sim_metrics[-1]+'.map', 'wb') as f:
    pkl.dump(dic, f)

#history = model.fit_generator(generator=training_generator, steps_per_epoch=len(X_train[1])//100 , epochs=13, validation_data=validation_generator, 
#                                       validation_steps = len(X_val[1])//100, workers = 10, use_multiprocessing=True, max_queue_size=20, verbose =1, shuffle=True)

#history = model.fit(X_train, y_train, epochs=12, batch_size=100,
#                    validation_data=(X_val, y_val),
#                    shuffle=True,  callbacks = [tbCallBack])
print('history finished.. ')
#with open('./model_baselinelstm'+ text_sim_metrics[-1]+'.map', 'wb') as f:
#    pkl.dump(model, f)
model.save('./model_baselinelstm.h5')
#histories['acc'].extend(history.history['acc'])
#histories['val_acc'].extend(history.history['val_acc'])
#histories['loss'].extend(history.history['loss'])
#histories['val_loss'].extend(history.history['val_loss'])
#histories['precision'].extend(history.history['precision'])
#histories['recall'].extend(history.history['recall'])
#histories['val_precision'].extend(history.history['val_precision'])
#histories['val_recall'].extend(history.history['val_recall'])

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

with open('./history_baselinelstm'+ text_sim_metrics[-1]+'.map', 'wb') as f:
    pkl.dump(histories, f)




