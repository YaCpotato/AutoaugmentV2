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
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

(Xtr, Ytr), (Xts, Yts) = datasets.cifar10.load_data()
Xtr = Xtr.astype('float32')
Xts = Xts.astype('float32')
Xtr /= 255
Xts /= 255
Ytr = np_utils.to_categorical(Ytr, 10)
Yts = np_utils.to_categorical(Yts, 10)

def model():
	model=Sequential()
	model.add(wide_resnet.WideResidualNetwork(depth=40, width=2, dropout_rate=0.1,include_top=True, weights=None,input_shape=None,classes=10, activation='softmax'))
	return model

model = model()
model.compile(optimizers.SGD(decay=1e-4), 'categorical_crossentropy', ['accuracy'])
start = time.time()
result = model.fit(Xtr,Ytr,batch_size=128,epochs=120,verbose=2,validation_split=0.1)
end = time.time() - start
print(end)
print('--------')
print(result.history)
score = model.evaluate(Xts,Yts,verbose=2)
print('===Final Test Score===')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
