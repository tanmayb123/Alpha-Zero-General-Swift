import os

import numpy as np

import tensorflow as tf
  
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def i_get_model():
    input_boards = Input(shape=(8, 8))
    x_image = Reshape((8, 8, 1))(input_boards)                # batch_size  x board_x x board_y x 1
    h_conv1 = Activation('relu')(Conv2D(512, 3, padding='same', use_bias=False)(x_image))         # batch_size  x board_x x board_y x num_channels
    h_conv2 = Activation('relu')(Conv2D(512, 3, padding='same', use_bias=False)(h_conv1))         # batch_size  x board_x x board_y x num_channels
    h_conv3 = Activation('relu')(Conv2D(512, 3, padding='valid', use_bias=False)(h_conv2))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
    h_conv4 = Activation('relu')(Conv2D(512, 3, padding='valid', use_bias=False)(h_conv3))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
    h_conv4_flat = Flatten()(h_conv4)
    s_fc1 = Dropout(0.3)(Activation('relu')(Dense(1024, use_bias=False)(h_conv4_flat)))  # batch_size x 1024
    s_fc2 = Dropout(0.3)(Activation('relu')(Dense(512, use_bias=False)(s_fc1)))          # batch_size x 1024
    pi = Dense(65, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
    v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

    model = Model(inputs=input_boards, outputs=[pi, v])
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(0.003))

    return model

def get_swift_weights(model):
    w = []
    for layer in model.layers:
        w += layer.get_weights()
    return w

def get_model():
    if os.path.isfile("/home/tajymany/tpu_lock"):
        return i_get_model()
    
    os.popen("touch /home/tajymany/tpu_lock")

    resolver =  tf.distribute.cluster_resolver.TPUClusterResolver("grpc://10.213.40.50:8470")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    with strategy.scope():
        m = i_get_model()

    return m

def train_model(model, data, batch_size, epochs):
    i = data[0].shape[0] % batch_size
    indata = data[0][:-i]
    pidata = data[1][:-i]
    vdata = data[2][:-i]
    model.fit(indata, (pidata, vdata), batch_size=batch_size, epochs=epochs, verbose=1)
