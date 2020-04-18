import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def fix_pi(pi, valids):
    return tf.math.softmax(pi - ((1 - valids) * 1000))

def conv_block(x):
    x = Conv2D(64, 3, padding='same', use_bias=False, activation='relu')(x)
    return x

def res_block(x):
    orig_x = x
    x = Conv2D(64, 3, padding='same', use_bias=False, activation='relu')(x)
    x = Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = Add()([x, orig_x])
    x = Activation('relu')(x)
    return x

class PyOthelloModel:
    def __init__(self):
        self.model = self.get_model()
        self.data = None
        self.episodeDataCounts = []

    def i_get_model(self):
        input_boards = Input(shape=(8, 8))
        input_valids = Input(shape=(65,))
        x_image = Reshape((8, 8, 1))(input_boards)                # batch_size  x board_x x board_y x 1
        x = conv_block(x_image)
        for i in range(10):
            x = res_block(x)
        h_conv4_flat = Flatten()(x)
        s_fc1 = Dropout(0.3)(Activation('relu')(Dense(1024, use_bias=False)(h_conv4_flat)))  # batch_size x 1024
        s_fc2 = Dropout(0.3)(Activation('relu')(Dense(512, use_bias=False)(s_fc1)))          # batch_size x 1024
        prepi = Dense(65, name='pi')(s_fc2)   # batch_size x self.action_size
        postpi = Lambda(lambda x: fix_pi(x[0], x[1]))([prepi, input_valids])
        v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        model = Model(inputs=[input_boards, input_valids], outputs=[postpi, v])
        model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(0.006))

        model.summary()

        return model

    def get_swift_weights(self):
        w = []
        for layer in self.model.layers:
            w += layer.get_weights()
        return w

    def get_model(self):
        if os.path.isfile("/home/tajymany/tpu_lock"):
            return self.i_get_model()
        
        os.popen("touch /home/tajymany/tpu_lock")

        resolver =  tf.distribute.cluster_resolver.TPUClusterResolver(tpu='mediumboy', project='asktanmayb', zone='us-central1-a')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

        with strategy.scope():
            m = self.i_get_model()

        return m

    def train_model(self, batch_size, epochs):
        i = self.data[0].shape[0] % batch_size
        if i > 0:
            boarddata = self.data[0][:-i]
            validsdata = self.data[1][:-i]
            pidata = self.data[2][:-i]
            vdata = self.data[3][:-i]
        else:
            boarddata = self.data[0]
            validsdata = self.data[1]
            pidata = self.data[2]
            vdata = self.data[3]
        p = np.random.permutation(len(boarddata))
        boarddata = boarddata[p]
        validsdata = validsdata[p]
        pidata = pidata[p]
        vdata = vdata[p]
        print([x.shape for x in [boarddata, validsdata, pidata, vdata]])
        self.model.fit([boarddata, validsdata], [pidata, vdata], batch_size=batch_size, epochs=epochs, verbose=1)

    def add_data(self, data):
        self.episodeDataCounts.append(data[0].shape[0])

        if self.data is None:
            self.data = data
            return

        for i in range(4):
            self.data[i] = np.concatenate([self.data[i], data[i]], axis=0)

        if len(self.episodeDataCounts) > 1:
            self.remove_first_episode()

    def remove_first_episode(self):
        for i in range(4):
            self.data[i] = self.data[i][self.episodeDataCounts[0]:]
        del self.episodeDataCounts[0]
