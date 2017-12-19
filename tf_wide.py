import sys, random, os, itertools, gc, collections
import numpy as np
from operator import itemgetter as ig
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from scipy.sparse import csr_matrix

trainFile = 'data/dat.train'
validFile = 'data/dat.valid'
testFile  = 'data/dat.test'
max_codes = 521

'''
Tensorflow Start the learning for wide
'''

def load_to_sparse(loadFile):
    '''
    Transform the codes and labids into SparseTensor
    '''
    posn = []
    y_out = []

    for c, row in enumerate(open(loadFile)):
        
        row = map(int, row.split(' '))
        y, x  = row[0], row[1:]
        y_out.append(y)
        
        for code in x:
            posn.append([c, code, 1])
             
    # Make a spare matrix of the rows and cols
    indices = [p[:2] for p in posn]
    vals    = [v[2] for v in posn]
    X_sparse_out = tf.SparseTensor(indices, vals, dense_shape=(c+1, max_codes+1))

    return X_sparse_out, y_out


def input_fn(X, y):
    feature_cols = {'codes': X}
    label = tf.constant(y, shape=[len(y)])
    return feature_cols, label

codes = tf.feature_column.categorical_column_with_identity('codes', num_buckets=max_codes)
wide_columns = [codes]

model = tf.estimator.LinearClassifier(
        model_dir='/tmp',
        optimizer = 'Ftrl',
        feature_columns=wide_columns)


for i in range(10):
    model.train(input_fn=lambda: input_fn(*load_to_sparse(trainFile)), steps = 1)
    results = model.evaluate(input_fn=lambda: input_fn(*load_to_sparse(validFile)), steps = 1)
    print("%s: %s", 'auc', results['auc'])
