import pickle as pkl
import keras
    
with open('y_fit_trainoimg.map', 'rb') as f:
    y_train = pkl.load(f)
with open('y_fit_testnoimg.map', 'rb') as f:
    y_val = pkl.load(f)

with open('dataset_fit_trainnoimg.map', 'rb') as f:
    X_train = pkl.load(f)

with open('dataset_fit_testnoimg.map', 'rb') as f:
    X_val = pkl.load(f)

with open('model_savenoimg.mod', 'rb') as f:
    model = pkl.load(f)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',  
                                         histogram_freq=1,  
                                         write_graph=True,  
                                         write_images=True)

histories = dict(acc=list(), val_acc=list(), loss=list(), val_loss=list())

import numpy as np

    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def load_image(x):
        x = np.load('./imagess'+x[1:]+'.npy')
        return x

    def __init__(self, list_IDs, labels, batch_size=100, shuffle=False):
        'Initialization'
        print('here is init__')
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        print((index+1)*self.batch_size, len(self.labels), '====================')
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = []
        list_y_temp = []
        print(len(self.list_IDs), '~~~~~~~~~~~~~~')
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
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = list_IDs_temp
        y = list_y_temp
        
        # Generate data
        X[6] = np.array([load_image(i) for i in X[6]]).reshape(-1,1,64,2048)
        X[13] = np.array([load_image(i) for i in X[13]]).reshape(-1,1,64,2048) 
        return X, y
    
training_generator = DataGenerator(X_train, y_train, batch_size=10)
#validation_generator = DataGenerator(X_val, y_val)
print('len', len(X_train[1]))
history = model.fit_generator(generator=training_generator, steps_per_epoch=len(X_train[1])//10 , epochs=2,
                    workers = 5, verbose =1)



#history = model.fit(X_train, y_train, epochs=8, batch_size=100,
#                    validation_data=(X_val, y_val),
#                    shuffle=True,  callbacks = [tbCallBack])
print('history finished.. ')

histories['acc'].extend(history.history['acc'])
histories['val_acc'].extend(history.history['val_acc'])
histories['loss'].extend(history.history['loss'])
histories['val_loss'].extend(history.history['val_loss'])

with open('./history_1.map', 'wb') as f:
    pkl.dump(histories, f)

