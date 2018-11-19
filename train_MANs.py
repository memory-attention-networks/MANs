import sys

import Models
from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2,l1
import pdb
import numpy as np
import lmdb
import threading
import os
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
import MANs_9
from keras.layers import GlobalAveragePooling2D,Dense,Activation
from keras.models import Model,model_from_json

import tensorflow as tf

from keras import backend as K

from keras.models import Model

from keras.layers import Input, merge

from keras.layers.core import Lambda


# seed 1234 is used for reproducibility
np.random.seed(seed=1234)
os.environ['CUDA_VISIBLE_DEVICES']='0,1'


## 0:Cross View, 1:Cross Subject
subject = 1

## Data root
if subject:
  data_root = 'data_ntu_subject/'
else:
  data_root = 'data_ntu_view/'

out_dir_name = 'MANs_subject'

## Parameters
loss = 'categorical_crossentropy'
lr = 0.01
momentum = 0.9

activation = "relu"
optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=True)
#optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
reg = l2(1.e-4)

batch_size = 128
epochs = 50
n_classes = 60
scale=50

if subject:
  samples_per_epoch = 39889
  samples_per_validation = 16390
else:
  samples_per_epoch = 37462
  samples_per_validation = 18817



class data_iter:
  """Takes an iterator/generator and makes it thread-safe by
  serializing call to the `next` method of given iterator/generator.
  """
  def __init__(self, it):
    self.it = it
    self.lock = threading.Lock()

  def __iter__(self):
    return self

  def __next__(self):
    with self.lock:
      return self.it.__next__()

def data_generator(f):
  """A decorator that takes a generator function and makes it thread-safe.
  """
  def g(*a, **kw):
      return data_iter(f(*a, **kw))
  return g


@data_generator
def train_data():

  lmdb_file_train_x = os.path.join(data_root, 'Xtrain_lmdb')
  lmdb_file_train_y = os.path.join(data_root, 'Ytrain_lmdb')

  lmdb_env_x = lmdb.open(lmdb_file_train_x)
  lmdb_txn_x = lmdb_env_x.begin()
  lmdb_cursor_x = lmdb_txn_x.cursor()

  lmdb_env_y = lmdb.open(lmdb_file_train_y)
  lmdb_txn_y = lmdb_env_y.begin()
  lmdb_cursor_y = lmdb_txn_y.cursor()
  
  X = np.zeros((batch_size,scale,scale,3))
  Y = np.zeros((batch_size,n_classes))
  batch_count = 0
  temp=1
  while True:
    indices = list(range(0,samples_per_epoch))
    np.random.shuffle(indices)
    
    for index in indices:
      value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()))
      label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index).encode()))

      x = value.reshape((scale, scale,3))
      x.setflags(write=1)

      X[batch_count] = np.reshape(x, [scale, scale,3])
      Y[batch_count] = label
      batch_count += 1

      if batch_count == batch_size:
        ret_x = X
        ret_y = Y
        X = np.zeros((batch_size, scale, scale,3))
        Y = np.zeros((batch_size,n_classes))
        temp+=1
        batch_count = 0
        yield (ret_x,ret_y)



@data_generator
def test_data():

  lmdb_file_test_x = os.path.join(data_root, 'Xtest_lmdb')
  lmdb_file_test_y = os.path.join(data_root, 'Ytest_lmdb')

  lmdb_env_x = lmdb.open(lmdb_file_test_x)
  lmdb_txn_x = lmdb_env_x.begin()
  lmdb_cursor_x = lmdb_txn_x.cursor()

  lmdb_env_y = lmdb.open(lmdb_file_test_y)
  lmdb_txn_y = lmdb_env_y.begin()
  lmdb_cursor_y = lmdb_txn_y.cursor()

  X = np.zeros((batch_size, scale, scale,3))
  Y = np.zeros((batch_size,n_classes))
  batch_count = 0
  temp=1
  while True:
    indices = list(range(0,samples_per_validation))
    np.random.shuffle(indices)

    for index in indices:
      value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()))
      label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index).encode()))

      x = value.reshape((scale,scale, 3))
      x.setflags(write=1)

      X[batch_count] = x.reshape((scale, scale,3))
      Y[batch_count] = label
      batch_count += 1

      if batch_count == batch_size:
        ret_x = X
        ret_y = Y
        X = np.zeros((batch_size, scale, scale,3))
        Y = np.zeros((batch_size,n_classes))
        batch_count = 0
        temp+=1
        yield (ret_x,ret_y)

def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice no.
    [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]

def to_multi_gpu(model, n_gpus=4):

    """Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.
    Each GPU gets a slice of the input batch, applies the model on that
    slice
    and later the outputs of the models are concatenated to a single
    tensor,
    hence the user sees a model that behaves the same as the original.
    """

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])
    towers = []
    device=[0,1,2,3]
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(device[g])):
            slice_g = Lambda(slice_batch, lambda shape: shape,
    arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = merge(towers, mode='concat', concat_axis=0)

    return Model(inputs=[x], outputs=merged)

def train():


    model=MANs_9.MANs_model()

    model=to_multi_gpu(model,2)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    if not os.path.exists('weights/'+out_dir_name):
      os.makedirs('weights/'+out_dir_name)
    weight_path = 'weights/'+out_dir_name+'/{epoch:03d}_{val_acc:0.3f}.hdf5'

    #serialize weight to h5
    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=5,
                                  verbose=1,
                                  mode='auto',
                                  cooldown=3,
                                  min_lr=0.0001)
    csv_logger = CSVLogger('log_MANs_9_subject.csv', append=True, separator=';')
    callbacks_list = [checkpoint,reduce_lr,csv_logger]

    model.fit_generator(train_data(),
                        steps_per_epoch=samples_per_epoch/batch_size+1,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_data=test_data(),
                        validation_steps=samples_per_validation/batch_size+1,
                        workers=1,
                        initial_epoch=0
                       )



if __name__ == "__main__":
  train()
