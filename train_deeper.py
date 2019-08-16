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
from deeper_models import deeper_img_generator, deeper_generator, deeper_img_decom_generator, deeper_img_decom_advance_generator, deeper_img_decom_vbi_generator
import argparse
import tensorflow as tf
print('deep_er_model gneration..')
from keras.layers import Dense, Activation, Multiply, Add, Lambda
import keras.initializers

import sys
import os
import re
sys.path.append('../scripts')

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import helpers as hp
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
        self.list_IDs[6] = list(self.list_IDs[6])
        self.list_IDs[13] = list(self.list_IDs[13])
    def load_image(self, x):
        try:
                x = np.load('./imagess'+x[1:]+'_new.npy')
                flag = True
        except FileNotFoundError as exception:
                x = np.load('./imagess/60/13520160.jpg_new.npy')
                flag = True
        except OSError as exception:
                x = np.load('./imagess/60/13520160.jpg_new.npy')
                flag = True
        if flag:
             return x

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        #print('this is index=', self.indexes[index*self.batch_size:(index+1)*self.batch_size],'==============================')
        # Generate indexes of the batch
    #    print((index+1)*self.batch_size, len(self.labels), '====================')
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = []
        list_y_temp = []
   #     print(len(self.list_IDs), '~~~~~~~~~~~~~~')

        for i in range(len( self.list_IDs)):
             K = self.list_IDs[i]
             list_IDs_temp.append(np.array( [ K[j] for j in indexes ]))
        #print(list_IDs_temp)
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






def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('multi-modal on deep neural record linkage')
    parser.add_argument('--model_name', default='Deeper_img', help='batch size of the dataset')
    parser.add_argument('--learning_rate', default=0.001, help='buffer size')
    # Add data arguments
    parser.add_argument('--image_url_cols', default='images_array', help='buffer size')
    parser.add_argument('--borad_log_dir', default=1, help='source language')
    parser.add_argument('--text_sim_metrics', type= int, default=1, help='target language')
    parser.add_argument('--text_compositions', default= 'average', help='train model on a tiny dataset')

    # Add model arguments
    parser.add_argument('--dropout', default=0.75, help='score of attention:default,general,concat')

    # Add optimization arguments
   # parser.add_argument('--models', default="GRU", help='force stop training at specified epoch')
    parser.add_argument('--patience', default=2, help='clip threshold of gradients')
    # Add checkpoint arguments
    parser.add_argument('--batch_size', default=100, help='path to save logs')
    parser.add_argument('--epoches', default=10, help='path to save checkpoints')
    parser.add_argument('--workers', default=16, help='filename to load checkpoint')
    parser.add_argument('--max_queue_size', default= 40, help='save a checkpoint every N epochs')
    parser.add_argument('--dense_node', default= 25, help='save a checkpoint every N epochs')

    # Parse twice as model arguments are not known the first time
    args = parser.parse_args()
    #ARCH_CONFIG_REGISTRY[args.arch](args)
    return args




def main(args):
    model_name = args.model_name
    lr = args.learning_rate
    image_url_cols = [args.image_url_cols]
    borad_log_dir = str(args.borad_log_dir)
    text_sim_metrics = int(args.text_sim_metrics)
    if text_sim_metrics == 1:
        text_sim_metrics = ['cosine']
    if text_sim_metrics == 2:
        text_sim_metrics = ['cosine', 'inverse_l1']
    if text_sim_metrics == 3:
        #text_sim_metrics = ['learnable_l1']
        text_sim_metrics = ['concat']
    if text_sim_metrics == 4:
        text_sim_metrics = ['inverse_l1']
    print(text_sim_metrics, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    text_compositions = [args.text_compositions]
    dropout = args.dropout
    image_url_cols = [args.image_url_cols]
    patience = args.patience
    batch_size = args.batch_size 
    epochs = args.epoches
    workers= args.workers
    max_queue_size = args.max_queue_size
   
    
    pd.options.display.max_colwidth = 1000

 #   rcParams['font.family'] = 'serif'
 #   rcParams['font.serif'] = 'times new roman'

    #%config InlineBackend.figure_format = 'retina'
    #%matplotlib inline
 #   reload(hp)

    with open('./embedding/fasttext-300.map', 'rb') as f:
        map = pkl.load(f)

    print('Data preparation..')
    data_dir = os.path.join('.','set_transfer')
    source_dir = os.path.join(data_dir)
    nan_idx = map['word2idx']['NaN']


    print('model established... ')
    log_dir = './Graph/'+ borad_log_dir
    import keras
    tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir,  
                                             histogram_freq=1,  
                                             write_graph=True,  
                                             write_images=True)




    histories = dict(acc=list(), val_acc=list(), loss=list(), val_loss=list(), precision=list(), val_precision=list(), f1_score=list(), val_f1_score=list(), auroc =list(), val_auroc=list())
 #   from 
    #with tf.device('/cpu:0'):
    if model_name =='Deeper':
          print(model_name, '################')
          model = deeper_generator(
                    embedding_file = './embedding/fasttext-300.matrix.npy',
                    text_columns = ['title_clean'],
                    numeric_columns = ['price','lat', 'lon', 'categoryID', 'locationID'],
                    text_nan_idx=nan_idx,
                    num_nan_val=0,
                    text_sim_metrics=text_sim_metrics,
                    text_compositions= ['average'],
                    numeric_sim_metrics=['scaled_inverse_lp', 'unscaled_inverse_lp', 'min_max_ratio'],
                    dense_nodes=[args.dense_node],
                    document_frequencies=None,
                    idf_smoothing=2,
                    make_isna=False,
                    embedding_trainable=True,
                    padding_limit=100,
                    batch_norm=True,
                    dropout=dropout,
                    shared_lstm=True,
                    #lstm_args = dict(units=50, dropout=0.25, recurrent_dropout=0.25),
                    lstm_args=dict(units=25)
                    )

    if model_name =='Deeper_img':
            print(model_name, '################')
            model = deeper_img_generator(
                        embedding_file = './embedding/fasttext-300.matrix.npy',
                        text_columns = ['title_clean'],
                        numeric_columns_1D = ['price'],
                        numeric_columns_2D = ['lat', 'lon'],
                        category_num_cols = ['categoryID', 'locationID'],
                        image_url_cols = ['images_array'],
                        text_nan_idx=nan_idx,
                        num_nan_val=0,
                        text_sim_metrics= text_sim_metrics,
                        text_compositions= ['average'],
                        image_sim_metrics = ['cosine', 'inverse_l1', 'inverse_l2'],
                        numeric_sim_metrics=['scaled_inverse_lp', 'unscaled_inverse_lp', 'min_max_ratio'],
                        dense_nodes=[args.dense_node],
                        document_frequencies=None,
                        idf_smoothing=2,
                        make_isna=False,
                        embedding_trainable=True,
                        padding_limit=100,
                        batch_norm = True,
                        dropout = dropout,
                        shared_lstm=True,
                        #lstm_args = dict(units=50, dropout=0.25, recurrent_dropout=0.25),
                        lstm_args=dict(units=25)
                        )
    if model_name =='Deeper_img_decom':
            print(model_name, '################')
            model = deeper_img_decom_generator(
                        embedding_file = './embedding/fasttext-300.matrix.npy',
                        text_columns = ['title_clean'],
                        numeric_columns_1D = ['price'],
                        numeric_columns_2D = ['lat', 'lon'],
                        category_num_cols = ['categoryID', 'locationID'],
                        image_url_cols = ['images_array'],
                        text_nan_idx=nan_idx,
                        num_nan_val=0,
                        text_sim_metrics= text_sim_metrics,
                        text_compositions= ['decomposable'],
                        numeric_sim_metrics=['scaled_inverse_lp', 'unscaled_inverse_lp', 'min_max_ratio'],
                        dense_nodes=[args.dense_node],
                        document_frequencies=None,
                        idf_smoothing=2,
                        make_isna=False,
                        embedding_trainable=True,
                        padding_limit=100,
                        batch_norm=True,
                        dropout=dropout,
                        shared_lstm=True,
                        #lstm_args = dict(units=50, dropout=0.25, recurrent_dropout=0.25),
                        lstm_args=dict(units=25)
                        )
    if model_name == 'Deeper_img_decom_advanced':
            print(model_name, '################')
            model = deeper_img_decom_advance_generator(
                    embedding_file = './embedding/fasttext-300.matrix.npy',
                    text_columns = ['title_clean'],
                    numeric_columns_1D = ['price'],
                    numeric_columns_2D = ['lat', 'lon'],
                    category_num_cols = ['categoryID', 'locationID'],
                    image_url_cols = ['images_array'],
                    text_nan_idx=nan_idx,
                    num_nan_val=0,
                    text_sim_metrics= text_sim_metrics,
                    text_compositions= ['hybrid'],
                    numeric_sim_metrics=['scaled_inverse_lp', 'unscaled_inverse_lp', 'min_max_ratio'],
                    dense_nodes=[args.dense_node],
                    document_frequencies=None,
                    idf_smoothing=2,
                    make_isna=False,
                    embedding_trainable=True,
                    padding_limit=100,
                    batch_norm=True,
                    dropout=dropout,
                    shared_lstm=True,
                    #lstm_args = dict(units=50, dropout=0.25, recurrent_dropout=0.25),
                    lstm_args=dict(units=25)
                    )
    if model_name == 'Deeper_img_decom_vbi':
            print(model_name, '################')
            model = deeper_img_decom_vbi_generator(
                    embedding_file = './embedding/fasttext-300.matrix.npy',
                    text_columns = ['title_clean'],
                    numeric_columns_1D = ['price'],
                    numeric_columns_2D = ['lat', 'lon'],
                    category_num_cols = ['categoryID', 'locationID'],
                    image_url_cols = ['images_array'],
                    text_nan_idx=nan_idx,
                    num_nan_val=0,
                    text_sim_metrics= text_sim_metrics,
                    text_compositions= ['vbi'],
                    numeric_sim_metrics=['scaled_inverse_lp', 'unscaled_inverse_lp', 'min_max_ratio'],
                    dense_nodes=[args.dense_node],
                    document_frequencies=None,
                    idf_smoothing=2,
                    make_isna=False,
                    embedding_trainable=True,
                    padding_limit=100,
                    batch_norm=True,
                    dropout=dropout,
                    shared_lstm=True,
                    #lstm_args = dict(units=50, dropout=0.25, recurrent_dropout=0.25),
                    lstm_args=dict(units=25)
                    )

    print('model complie...')
    #sgd = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
    import keras_metrics as km
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    from keras.utils import multi_gpu_model as gpu
    # model = gpu(model, 2)


    def auroc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', km.binary_precision(),km.f1_score(),auroc])

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

    with open('./y_fit_trainimg.map', 'rb') as f:
        y_train = pkl.load(f)
    with open('./y_fit_valimg.map', 'rb') as f:
        y_val = pkl.load(f)

    with open('./dataset_fit_trainimg.map', 'rb') as f:
        X_train = pkl.load(f)

    with open('./dataset_fit_valimg.map', 'rb') as f:
        X_val = pkl.load(f)

    with open('./dataset_fit_testimg.map', 'rb') as f:
        X_test = pkl.load(f)

    with open('./y_fit_testimg.map', 'rb') as f:
        y_test = pkl.load(f)   

    # import pickle as pkl
    # with open('./dataset_fit_trainnoimg.map', 'wb') as f:
    #       pkl.dump(X_train, f)
    # with open('./dataset_fit_testnoimg.map', 'wb') as f:
    #       pkl.dump(X_test, f)
    # with open('./y_fit_testnoimg.map', 'wb') as f:
    #       pkl.dump(y_test, f)
    # with open('./y_fit_trainnoimg.map', 'wb') as f:
    #       pkl.dump(y_train, f)
    # with open('./y_fit_valnoimg.map', 'wb') as f:
    #       pkl.dump(y_val, f)
    # with open('./dataset_fit_valnoimg.map', 'wb') as f:
    #       pkl.dump(X_val, f)
    training_generator = DataGenerator(X_train, y_train, batch_size=batch_size)
    validation_generator = DataGenerator(X_val, y_val, batch_size = batch_size)
    print('len', len(X_train[1]))
    import keras
    earlystop = keras.callbacks.EarlyStopping(monitor='val_auroc', min_delta=0, patience=2, verbose=1, mode='max', baseline=None, restore_best_weights=True)
   # checkpoint = keras.callbacks.ModelCheckpoint('./checkpoints/'+ model_name + borad_log_dir+'.hdf5', monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')
    history = model.fit_generator(generator=training_generator, steps_per_epoch=len(X_train[1])//batch_size , epochs= 3, validation_data=validation_generator, validation_steps = len(X_val[1])//batch_size,
                         workers = workers, use_multiprocessing = True, max_queue_size= max_queue_size, verbose =1, shuffle=True, callbacks = [earlystop])
    model_log = './model_log/'+ model_name + borad_log_dir +'h5'
 #   with open( model_log, 'wb') as f:
 #         pkl.dump(model, f)
    model.save(model_log)
    #model.load_weights('./model_log/'+ model_name + borad_log_dir +'.tmod')
    generator = DataGenerator(X_test, y_test, batch_size=100)
    prediction = model.predict_generator(generator, steps=len(X_test[1])//100, max_queue_size=10, workers=7, use_multiprocessing=True, verbose=0)
    label = y_test
    dic = {}
    dic['prediction'] = prediction
    dic['label'] = label
    dic['data'] = X_test

    with open('./data_'+model_name +'.map', 'wb') as f:
         pkl.dump(dic, f)

    #from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    #p, r, th = precision_recall_curve(y_test, prediction)
    #auc = auc(r, p)
    #ap = average_precision_score(y_test, prediction)
    
    #dic['auc'] = auc
    #dic['ap'] = ap
    #with open('./data_'+model_name + text_sim_metrics[-1]+'.map+auc', 'wb') as f:
    #     pkl.dump(dic, f)

    #history = model.fit(X_train, y_train, epochs=8, batch_size=100,
    #                    validation_data=(X_val, y_val),
    #                    shuffle=True,  callbacks = [tbCallBack])
# len(X_train[1])//batch_size 
    print('history finished.. ')

#    model_log = './model_log/'+ model_name + borad_log_dir +'.tmod'
 #   with open( model_log, 'wb') as f:
 #         pkl.dump(model, f)
#    model.save(model_log)

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

    history_log = './history_log/'+ model_name + borad_log_dir + '.tmap'
    with open(history_log, 'wb') as f:
        pkl.dump(histories, f)

    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
   # p, r, th = precision_recall_curve(y_test, prediction)
   # auc = auc(r, p)
   # ap = average_precision_score(y_test, prediction)
   # print(ap, auc , '===========ap and auc============')
   # dic['auc'] = auc
   # dic['ap'] = ap
   # with open('./data_'+model_name + text_sim_metrics[-1]+'.map+auc', 'wb') as f:
   #      pkl.dump(dic, f)

if __name__ == '__main__':
    args = get_args()
    main(args)


