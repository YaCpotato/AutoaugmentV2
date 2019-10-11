import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import models, layers, datasets, utils, backend, optimizers, initializers
backend.set_session(session)
from transformations import get_transformations
import numpy as np
import time
from keras.models import Sequential
import wide_resnet
from keras.utils import np_utils

(Xtr, Ytr), (Xts, Yts) = datasets.cifar10.load_data()
Xtr = Xtr.astype('float32')
Xts = Xts.astype('float32')
Xtr /= 255
Xts /= 255
Ytr = np_utils.to_categorical(Ytr, 10)
Yts = np_utils.to_categorical(Yts, 10)

def model():
	model=Sequential()
	model.add(wide_resnet.WideResidualNetwork(depth=28, width=8, dropout_rate=0.1,include_top=True, weights=None,input_shape=None,classes=10, activation='softmax'))
	return model

model = model()
model.compile(optimizers.SGD(decay=1e-4), 'categorical_crossentropy', ['accuracy'])
model.fit(Xtr,Ytr,batch_size=128,epochs=1,validation_split=0.1)
model.evaluate(Xts,Yts)
