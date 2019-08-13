from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from collections import defaultdict
import pandas as pd
import pickle as pkl


model = KeyedVectors.load('./araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model')

model.save( './embedding/{}-300.gensim'.format('fasttext'))


# create and save numpy embedding matrix with initial row of zeros
print('creating embedding matrix')
embedding_matrix = model.vectors
embedding_matrix = np.vstack([np.zeros(300), embedding_matrix])
np.save(file='./embedding/{}-300.matrix'.format('fasttext'),
        arr=embedding_matrix)

model_name = 'fasttext'
# create and save two maps of corpus vocabulary
print('creating maps')
vocab = ['<unk>'] + list(model.vocab.keys())
word2idx = defaultdict(int, zip(vocab, range(len(vocab))))
idx2word = dict(zip(range(len(vocab)), vocab))

# manually encode NaN's as unknown
for nan in ['NaN', 'NAN', 'nan', 'Nan']:
    word2idx[nan] = 0

map = dict()
map['word2idx'] = word2idx
map['idx2word'] = idx2word

with open('./embedding/{}-300.map'.format(model_name), 'wb') as f:
    pkl.dump(map, f)
    #a kind of saving methods that save map into f file.
