import sys
import os
import re
import tensorflow as tf
sys.path.append('../scripts')

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import helpers as hp
import pickle as pkl
import itertools as it
import keras

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
from keras.engine.topology import Layer


print('deep_er_model gneration..')
from keras.layers import Dense, Activation, Multiply, Add, Lambda
import keras.initializers


class matrix_cor(Layer):
	"""
credit to author from: https://www.jianshu.com/p/6c34045216fb 
The function of this layer is to construct a class of a defined layer
	"""
	def __init__(self, output_dim, **kwargs):
        	self.output_dim = output_dim
        	super(matrix_cor, self).__init__(**kwargs)

	def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        	self.kernel_1 = self.add_weight(name='kernel', 
                                      shape=(self.output_dim, input_shape ),
                                      initializer='uniform',
                                      trainable=True)

        	super(matrix_cor, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
        	temp_t = tf.multiply(self.kernel_1, x)
        	return temp_t

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

def gru( units,dropout=0.4, directional=True, return_sequences= True, gpu= 1 ):
  # If you have a GPU, we recommend using the CuDNNGRU layer (it provides a 
  # significant speedup).
    if directional== False:       
     #   if tf.test.is_gpu_available():
        if gpu ==0:
            return keras.layers.CuDNNGRU(units,
                                        return_sequences=return_sequences, 
                                        recurrent_initializer='glorot_uniform')
        else:
            return keras.layers.GRU(units, dropout= dropout,
                                   return_sequences=return_sequences, 
                                   recurrent_activation='sigmoid', 
                                   recurrent_initializer='glorot_uniform')
    else:
     #   if tf.test.is_gpu_available():
        if gpu==0:
            return keras.layers.Bidirectional(keras.layers.CuDNNGRU(units, 
                                        return_sequences=return_sequences, 
                                        recurrent_initializer='glorot_uniform'), merge_mode='concat')
        else:
            return keras.layers.Bidirectional(keras.layers.GRU(units, dropout= dropout,
                                   return_sequences=return_sequences, 
                                   recurrent_activation='sigmoid', 
                                   recurrent_initializer='glorot_uniform'),  merge_mode='concat')

def summary_vector(input_pair_1, input_pair_2 ):
    text_img_tensors = dict()
    text_img_tensors['left'] = input_pair_1
    text_img_tensors['right'] = input_pair_2
    def xdoty(x, y):
        return tf.matmul(x, tf.transpose(y, [0,2,1]) )

    score_layer = Lambda(lambda x: xdoty(x[0], x[1]))

    att_tensors =dict() 
    left = score_layer( [text_img_tensors['left'],text_img_tensors['right']])
    att_tensors['left'] = Softmax(axis=-1)(left)
    att_tensors['right'] = Softmax(axis=-1)(score_layer([text_img_tensors['right'], text_img_tensors['left']]))

    #gru_1 = gru(units=300,dropout=0.1, directional=True, return_sequences = True)
    from keras_self_attention import SeqSelfAttention
    gru_1 = SeqSelfAttention(attention_activation='sigmoid')
    rnn_1_layer = Lambda(lambda x: gru_1(x)) 

    alphas = dict()
    g1 = dict()
    dot_layer = Lambda(lambda x: tf.matmul(x[0], x[1]) )
    temp_left = Dropout(0.15)( text_img_tensors['left'])
    g1['left'] = rnn_1_layer(temp_left)
    #g1['left'] = Dropout(0.35)( g1['left'])
    temp_right = Dropout(0.15)( text_img_tensors['right'])
    g1['right'] = rnn_1_layer(temp_right)
    #g1['right'] = Dropout(0.35)( g1['right'])    

    alphas['left'] = dot_layer([att_tensors['left'], g1['right'] ])
    alphas['right'] = dot_layer([att_tensors['right'], g1['left'] ])

    concat_tensors = dict()
    #differ_layer = Lambda(lambda x: highway_layers(tf.math.subtract(x[0], x[1]),2) )
    concat_tensors['left'] = Dense(600)(Concatenate(axis=-1)([g1['left'], alphas['left'] ]))
    concat_tensors['right'] = Dense(600)(Concatenate(axis=-1)([g1['right'], alphas['right'] ]))    

   # concat_tensors['left'] = differ_layer([g1['left'], alphas['left'] ])
   # concat_tensors['right'] = differ_layer([g1['right'] , alphas['right'] ])

    gru_2 = gru(units=300,dropout=0.1, directional=True, return_sequences= False)
#     units=300
#     dropout=0.4
#     directional=True
#     return_sequences= False
#     gru_2 = keras.layers.GRU(units, dropout= dropout,
#                                    return_sequences=return_sequences, 
#                                    recurrent_activation='sigmoid', 
#                                    recurrent_initializer='glorot_uniform')
    rnn_2_layer = Lambda(lambda x: tf.transpose( tf.reshape(gru_2(x), (-1, 1,K.int_shape(gru_2(x))[-1]) ), [0,2,1] ) )
    squeeze_layer = Lambda(lambda x: tf.squeeze(x, [1]) )
    g2 = dict()
    temp_1 = Dropout(0.15)( text_img_tensors['left'])
    g2['left'] = rnn_2_layer(temp_1)
    #g2['left'] = Dropout(0.35)( g2['left'])
    temp_2 = Dropout(0.15)( text_img_tensors['right'])
    g2['right'] = rnn_2_layer( temp_2 )
    #g2['right'] = Dropout(0.35)( g2['right'])
    #highwaynet_compared = Lambda(lambda x: highway_layers(x,2))

    composed_tensors = dict()
    composed_tensors['hybrid'] = dict()
    #broadcast_layer = Lambda(lambda x: tf.broadcast_to(x, [ None, 
    #                                                    K.int_shape(concat_tensors['left'])[1],K.int_shape(x)[-1] ]) )
    attention_layer = Dense(K.int_shape(g2['left'])[-1])
    g2['left'] = attention_layer(g2['left'])
    g2['right'] = attention_layer(g2['right'])

    composed_tensors['hybrid']['left'] = dot_layer([concat_tensors['left'], g2['right']])
    composed_tensors['hybrid']['right'] = dot_layer([concat_tensors['right'], g2['left']])

    #highway_weight_layer = Lambda(lambda x: highway_layers(x, 2))
    #composed_tensors['hybrid']['left'] = Dense(1)(highway_weight_layer(composed_tensors['hybrid']['left']  ))
    #composed_tensors['hybrid']['right'] = Dense(1)(highway_weight_layer(composed_tensors['hybrid']['right'] ))  

    softmax_layer = Lambda(lambda x: tf.transpose(Softmax(axis=-2)(x), [0, 2,1]))
    composed_tensors['hybrid']['left'] = softmax_layer(composed_tensors['hybrid']['left'] )
    composed_tensors['hybrid']['right'] = softmax_layer(composed_tensors['hybrid']['right'] )

    composed_tensors['hybrid']['left'] = dot_layer([composed_tensors['hybrid']['left'], concat_tensors['left'] ])
    composed_tensors['hybrid']['right'] = dot_layer([composed_tensors['hybrid']['right'], concat_tensors['right'] ]) 

    composed_tensors['hybrid']['left'] = squeeze_layer(composed_tensors['hybrid']['left'])
    composed_tensors['hybrid']['right'] = squeeze_layer(composed_tensors['hybrid']['right'])    
    return composed_tensors
        
def deeper_generator(
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
                            text_compositions = ['average'],
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
    
    maxlen = {'title_clean': 100} 
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
        averaging_layer = Lambda(lambda x: K.mean(x, axis=1) )
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
    if 'learnable_l1' in text_sim_metrics:
        similarity_layer = Lambda(lambda x: K.abs(x[0]-x[1]))
        similarity_layers.append(similarity_layer)
    if 'concat' in text_sim_metrics:
        similarity_layer = Concatenate(axis=-1)
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
    
    return model


def deeper_img_generator(
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
                            text_compositions = ['average'],
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
    print(text_sim_metrics , '#############################################')
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
    
    maxlen = {'title_clean': 100} 
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
        
    embedding_layer_image= Sequential([Flatten(), Dense(600, activation='relu')])
    embedded_tensors = dict(left=dict(), right=dict())
    for side, column in it.product(sides, image_url_cols):
        embedded_tensors[side][column] = embedding_layer_image(input_tensors[side][column])
    
    similarity_image_layers = list()
    if 'cosine' in image_sim_metrics:
        similarity_image_layer = Dot(axes=1, normalize=True)
        similarity_image_layers.append(similarity_image_layer)
    if 'inverse_l1' in image_sim_metrics:
        similarity_image_layer = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))
        similarity_image_layers.append(similarity_image_layer)
    if 'inverse_l2' in image_sim_metrics:
        similarity_image_layer = Lambda(lambda x: \
                                  K.exp(-K.sqrt(K.sum(K.pow(x[0]-x[1], 2), axis=1, keepdims=True))))
        similarity_image_layers.append(similarity_image_layer)
    if 'learnable_l1' in image_sim_metrics:
        similarity_layer = Lambda(lambda x: K.abs(x[0]-x[1]))
        similarity_layers.append(similarity_layer)
    if 'concat' in image_sim_metrics:
        similarity_layer = Concatenate(axis=-1)
        similarity_layers.append(similarity_layer)        
    # for each attribute, calculate similarities between left and ride sides
    similarity_tensors = list()
    for column, similarity_layer in \
        it.product(image_url_cols, similarity_image_layers):        
        similarity_tensor = similarity_layer([embedded_tensors['left'][column],
                                              embedded_tensors['right'][column]])
        similarity_tensors.append(similarity_tensor)
    
    
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
    #embedded_tensors = dict(left=dict(), right=dict())
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
        averaging_layer = Lambda(lambda x: K.mean(x, axis=1) )
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
    if 'learnable_l1' in text_sim_metrics:
        similarity_layer = Lambda(lambda x: K.abs(x[0]-x[1]))
        similarity_layers.append(similarity_layer)
    if 'concat' in text_sim_metrics:
        similarity_layer = Concatenate(axis=-1)
        similarity_layers.append(similarity_layer)

    # for each attribute, calculate similarities between left and ride sides
    #similarity_tensors = list()
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
    concatenated_tensors = Concatenate(axis=-1)(similarity_tensors )
    print(keras.backend.int_shape(concatenated_tensors), '***************************************' )    
    # create dense layers starting with concatenated tensors
    dense_tensors = [concatenated_tensors]
    for n_nodes in dense_nodes:
        fc = Dense(n_nodes, activation='relu', name='output')
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
        
    output_tensors = Dense(2, activation='softmax')(dense_tensors[-1])
    
    product = list(it.product(sides, columns))
    if not debug:
        model = Model([input_tensors[s][tc] for s, tc in product] + input_isna_tensors,
                      [output_tensors])
    else:
        model = Model([input_tensors[s][tc] for s, tc in product] + input_isna_tensors,
                      [embedded_tensors['left'][text_columns[0]]])
    
    return model

def deeper_img_decom_generator(
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
    print(text_sim_metrics , '#############################################')
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
    
    maxlen = {'title_clean': 100} 
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
 
    
    return model


def deeper_img_decom_advance_generator(
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
                            text_compositions = ['hybrid'],
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
    print(text_sim_metrics , '#############################################')   
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
    maxlen = {'title_clean': 100} 
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
#######################################
######################################
######################################
#######################################
    text_img_tensors =dict(left=dict(), right=dict())
    squeeze_layer = Lambda(lambda x: tf.squeeze(x, [1]) )
    reshape_layer = Lambda(lambda x: tf.reshape( x, (-1, 164,300)))
    for side in sides:
        #tf.squeeze(embedded_tensors[side][image_url_cols[0]]) 
        embedded_tensors[side][image_url_cols[0]] = squeeze_layer(embedded_tensors[side][image_url_cols[0]] )
        #text_img_tensors[side] = Concatenate(axis=-2)([embedded_tensors[side][text_columns[0]], embedded_tensors[side][image_url_cols[0]] ])
        #text_img_tensors[side] = reshape_layer(text_img_tensors[side])   
        
   
    #embedded_tensors[side][text_columns[0]], embedded_tensors[side][image_url_cols[0]]
    
    t2t = summary_vector(embedded_tensors['left'][text_columns[0]], embedded_tensors['right'][text_columns[0]] )
    t2i = summary_vector(embedded_tensors['left'][text_columns[0]], embedded_tensors['right'][image_url_cols[0]])
    i2t = summary_vector(embedded_tensors['left'][image_url_cols[0]], embedded_tensors['right'][text_columns[0]] )
    i2i = summary_vector(embedded_tensors['left'][image_url_cols[0]], embedded_tensors['right'][image_url_cols[0]])
    
    composed_tensors = dict()
    composed_tensors['hybrid'] = dict()
    
    for side in sides:
        composed_tensors['hybrid'][side] = Concatenate(axis=-1)([ t2t['hybrid'][side], t2i['hybrid'][side], i2t['hybrid'][side],i2i['hybrid'][side] ])
        
        
    
    
    
    
    
##################################################
##################################################
##################################################
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
        
    output_tensors = Dense(2, activation='softmax')(dense_tensors[0])
    
    
    #output_tensors = text_img_tensors['left']
    product = list(it.product(sides, columns))
    if not debug:
        model = Model([input_tensors[s][tc] for s, tc in product] + input_isna_tensors,
                      [output_tensors])
    else:
        model = Model([input_tensors[s][tc] for s, tc in product] + input_isna_tensors,
                      [embedded_tensors['left'][text_columns[0]]])
    
    return model


def deeper_img_decom_vbi_generator(
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
                            text_compositions = ['vbi'],
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
    print(text_sim_metrics , '#############################################')
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
    maxlen = {'title_clean': 100} 
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
    squeeze_layer = Lambda(lambda x: tf.squeeze(x, [1]))
    embedded_tensors = dict(left=dict(), right=dict())

    for side, column in it.product(sides, image_url_cols):
        embedded_tensors[side][column] = embedding_layer_image(input_tensors[side][column])
        embedded_tensors[side][column] = squeeze_layer( embedded_tensors[side][column])
    
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

    #highwaynet_sent = Lambda(lambda x: highway_layers(x,2))
   
    embedding_matrix = np.load(embedding_file)
    embedding_layer = Sequential()
    
    embedding_layer.add(Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                trainable=embedding_trainable,
                                mask_zero=mask_zero) )
    #embedding_layer.add(highwaynet_sent)
    
    # use embedding_layer ot convert text input tensors to embedded tensors
    # and store in dictionary.
    # an embedding tensor will have shape n_words x n_embedding_dimensions
    #embedded_tensors = dict(left=dict(), right=dict())
    directionals = ['positive', 'inverse']
    for side, column in it.product(sides, text_columns):
        embedded_tensors[side][column] = embedding_layer(input_tensors[side][column])
  
    text_img_tensors = list()
    def bidirect_layer( in_question_words, in_passage_words, ll=50):
        # ========Bilateral Matching=====
        matrix = matrix_cor(ll)
        matrix.build( 2*ll )
        match_representation = bilateral_match_func(in_question_words, in_passage_words, matrix)

        return match_representation
    
    def bilateral_match_func(in_question_repres, in_passage_repres, matrix):
        match_representation_1 = uni_match_func(in_question_repres, in_passage_repres, matrix)
     #   match_representation_2 = uni_match_func(in_passage_repres, in_question_repres, matrix)
        
        #match_representation = Concatenate(axis=-1)([match_representation_1, match_representation_2])
        
        return match_representation_1
    
    def uni_match_func(in_question_repres, in_passage_repres, matrix, units = 50, ll=50):   
        gru_question_layer = gru( units, directional=True, return_sequences= True) #[batch_size, number of word or imagevector, 2h]
        gru_passage_layer = gru( units,directional=True, return_sequences= False)
        gru_question = gru_question_layer(in_question_repres)
        gru_passage = gru_passage_layer(in_passage_repres)
        gru_question = tf.expand_dims(gru_question, 2)
        gru_question = tf.tile(input=gru_question, multiples=[1, 1, ll, 1])
        gru_question = matrix(gru_question)
        print(K.int_shape(gru_question))
        gru_question = tf.reshape(gru_question, [-1, tf.shape(in_question_repres)[1], 2*ll, units])
        gru_passage = tf.expand_dims(gru_passage, 1)
        gru_passage = tf.tile(input=gru_passage, multiples=[1, ll, 1])
        gru_passage = matrix(gru_passage)
        gru_passage = tf.expand_dims(gru_passage, 1)
        gru_passage = tf.tile(input=gru_passage, multiples=[1, tf.shape(gru_question)[1], 1, 1])
        gru_passage = tf.reshape(gru_passage, [-1, tf.shape(in_question_repres)[1], 2*ll, units])
        

        unscale_cos = tf.reduce_sum(tf.multiply(gru_passage, gru_question), -1)
        norm_question = tf.sqrt(tf.reduce_sum(tf.square(gru_question), -1))
        norm_passage = tf.sqrt(tf.reduce_sum(tf.square(gru_passage), -1))
        cos_m = unscale_cos / (norm_passage * norm_question + 0.0000001)
        #cos_m = tf.squeeze(cos_m, [-1])
        #cos_m = tf.reshape(cos_m, [-1, tf.shape(in_question_repres)[1], ll, 2*units])
        gru_agg_layer = gru( units,directional=True, return_sequences= False)
        summary_vector = gru_agg_layer(cos_m)
        return summary_vector
        
        
    mbm_layer = Lambda(lambda x: bidirect_layer(x[0], x[1]))
    mbm_layer_1 = Lambda(lambda x: bidirect_layer(x[0], x[1]))
    mbm_layer_2 = Lambda(lambda x: bidirect_layer(x[0], x[1]))
    text_img_tensors.append(mbm_layer([embedded_tensors['left'][text_columns[0]], embedded_tensors['right'][text_columns[0]] ]))
    print(K.int_shape( embedded_tensors['right'][image_url_cols[0]] ))
    text_img_tensors.append( mbm_layer_1([embedded_tensors['left'][text_columns[0]], embedded_tensors['right'][image_url_cols[0]] ]))
    text_img_tensors.append( mbm_layer_1([embedded_tensors['right'][text_columns[0]], embedded_tensors['left'][image_url_cols[0]] ]) )
    text_img_tensors.append( mbm_layer_2([embedded_tensors['left'][image_url_cols[0]], embedded_tensors['right'][image_url_cols[0]] ]) )
       

    #highwaynet_compared = Lambda(lambda x: highway_layers(x,2))
    
    composed_tensors = dict()
    composed_tensors['vbi'] = Concatenate(axis=-1)(text_img_tensors)
    
    similarity_tensors.append(composed_tensors['vbi'] )
            
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
    concatenated_tensors = Concatenate(axis=-1)(similarity_tensors  )
    
    #x = Concatenate(axis=-1)( [concatenated_tensors]+ [tf.convert_to_tensor([[[1,2],[3,4]]])])
    # create dense layers starting with concatenated tensors
    dense_tensors = [concatenated_tensors]
    print(keras.backend.int_shape(concatenated_tensors), '***************************************')
    
    for n_nodes in dense_nodes:
        fc = Dense(2*n_nodes, activation='relu', name='output')
        #fc_1 = Dense(n_nodes, activation='relu', name='output_2')
        #print(type(fc))
        dense_tensor = fc(dense_tensors[-1])
        #dense_tensor = fc_1(dense_tensor)
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
    #output_tensors =  composed_tensors['vbi'] 
    
    #output_tensors = text_img_tensors['left']
    product = list(it.product(sides, columns))
    if not debug:
        model = Model([input_tensors[s][tc] for s, tc in product] + input_isna_tensors,
                      [output_tensors])
    else:
        model = Model([input_tensors[s][tc] for s, tc in product] + input_isna_tensors,
                      [embedded_tensors['left'][text_columns[0]]])
    
#    return tuple([model] + list(packaged_data.values()) + [y_train, y_val, y_test])
    
    return model

