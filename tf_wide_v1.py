import sys, random, os, itertools, gc, collections
import numpy as np
from operator import itemgetter as ig
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from scipy.sparse import csr_matrix
from math import exp, log, sqrt

trainFile = 'data/dat.train'
validFile = 'data/dat.valid'
testFile  = 'data/dat.test'
max_codes = 521


'''
tingrtu FTRL proximal
@ https://www.kaggle.com/c/avazu-ctr-prediction/discussion/10927
'''

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g

# Params
alpha = .01  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized
D  = max_codes
epoch = 20

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D)

# start training
for e in xrange(epoch):
    loss = 0.
    count = 0
    
    for c, row in enumerate(open(trainFile)):
        
        row = map(int, row.split(' '))
        y, x  = row[0], row[1:]
        p = learner.predict(x)
        learner.update(x, p, y)

    # validate
    y_predval = []
    y_valid = []
    for c, row in enumerate(open(validFile)):
        row = map(int, row.split(' '))
        y, x  = row[0], row[1:]
        p = learner.predict(x)
        y_valid.append(y)
        y_predval.append(p)
    print 'epoch: %s auc: %s'%(e, roc_auc_score(y_valid, y_predval))

# Predict
y_predtst = []
y_test = []
for c, row in enumerate(open(testFile)):
    row = map(int, row.split(' '))
    y, x  = row[0], row[1:]
    p = learner.predict(x)
    y_test.append(y)
    y_predtst.append(p)
    
print 50*'-'
print 'Kaggle test auc: %s'%(roc_auc_score(y_test, y_predtst))

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
