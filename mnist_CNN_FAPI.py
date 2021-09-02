import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import mnist


(x_train, y_train),(x_test,y_test) = mnist.load_data()

lbl_unique = len(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
input_shape = (image_size, image_size, 1)

x_train = np.reshape(x_train, [-1,image_size,image_size,1]) / 255
x_test = np.reshape(x_test, [-1,image_size,image_size,1]) / 255 

filters = 64
kernel_size = 3
batch_size = 200
dropout = 0.2

model_path = './save_model'
if not os.path.exists(model_path):
    os.mkdir(model_path)

model_checkpoint = ModelCheckpoint(filepath = model_path +'/cp.h5',
                                   monitor = 'accuracy',
                                   mode='max',
                                   save_best_only = True,
                                   save_weights_only=False,
                                   verbose = 1)
early_stopping = EarlyStopping(monitor = 'accuracy', patience = 10)
call_backs = [model_checkpoint,early_stopping]

input_d = Input(shape = input_shape)
x = Conv2D(filters = filters, kernel_size = kernel_size, activation = 'relu')(input_d)
x = MaxPooling2D(pool_size = 2)(x)
x = Conv2D(filters = filters, kernel_size = kernel_size, activation = 'relu')(x)
x = Flatten()(x)
x = Dropout(dropout)(x)
x = Dense(lbl_unique)(x)
output_D = Activation('softmax')(x)

model = Model(input_d,output_D)
model.summary()

model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'categorical_crossentropy')
model.fit(x_train,y_train, batch_size = batch_size, epochs = 10, callbacks=call_backs)
_, acc = model.evaluate(x_test,y_test, batch_size = batch_size, verbose = 0)

print("\nTesting accuracy: %.1f%%" % (100.0 * acc))