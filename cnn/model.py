# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import numpy as np

import keras.backend as K
from keras.layers import Conv2D, Dense, BatchNormalization, Flatten, Input, Activation, Add
from keras import Model
from keras.initializers import TruncatedNormal
from keras.regularizers import l2
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.optimizers import SGD
from keras.models import save_model, load_model
from keras.callbacks import TensorBoard
import os
import cnn.config as C
from game.common import BOARD_SIZE, HISTORY_CHANNEL
import shutil


class ReversiModel(object):
    def __init__(self, mode='challenger'):
        self.model = None
        try:
            if mode == 'challenger':
                self.load_challenger_model()  # always train the challenger model and bat with the defender
            elif mode == 'defender':
                self.load_defender_model()
        except Exception as e:
            print(e)

    def __del__(self):
        K.clear_session()

    def remove_model(self):
        try:
            os.remove(C.model_challenger_path)
        except FileNotFoundError as e:
            print(e)
        try:
            os.remove(C.model_defender_path)
        except FileNotFoundError as e:
            print(e)

    def rebuild_model(self):
        self.remove_model()
        self.build_model()

    def build_model(self):
        # if self.model:
        #     print('model already loads from file.h5')
        #     return
        input_x = x = Input((BOARD_SIZE, BOARD_SIZE, HISTORY_CHANNEL * 2 + 1))
        x = Conv2D(
            input_shape=(BOARD_SIZE, BOARD_SIZE, HISTORY_CHANNEL * 2 + 1),
            filters=C.cnn_filter_num,
            kernel_size=C.cnn_filter_size,
            kernel_initializer=TruncatedNormal(stddev=0.1),
            kernel_regularizer=l2(C.l2_reg),
            bias_initializer=TruncatedNormal(stddev=0.1),
            # activation='relu',
            padding='same',
            data_format='channels_last',
        )(x)

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        # placeholder for residual block
        for _ in range(C.res_layer_num):
            x = self._build_residual_block(x)

        pre_out = x

        # for policy output
        x = Conv2D(
            filters=2,
            kernel_size=1,
            kernel_initializer=TruncatedNormal(stddev=0.1),
            kernel_regularizer=l2(C.l2_reg),
            bias_initializer=TruncatedNormal(stddev=0.1),
            # activation='relu',
            # padding='same',
            data_format='channels_last',
        )(pre_out)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        policy = Dense(
            BOARD_SIZE ** 2,
            kernel_initializer=TruncatedNormal(stddev=0.1),
            kernel_regularizer=l2(C.l2_reg),
            bias_initializer=TruncatedNormal(stddev=0.1),
            activation="softmax",
            name="policy",
        )(x)

        # for value output
        x = Conv2D(
            filters=1,
            kernel_size=1,
            kernel_initializer=TruncatedNormal(stddev=0.1),
            bias_initializer=TruncatedNormal(stddev=0.1),
            data_format="channels_last",
            kernel_regularizer=l2(C.l2_reg)
        )(pre_out)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(
            256,
            kernel_regularizer=l2(C.l2_reg),
            kernel_initializer=TruncatedNormal(stddev=0.1),
            bias_initializer=TruncatedNormal(stddev=0.1),
            activation="relu",
        )(x)
        value = Dense(
            1,
            kernel_regularizer=l2(C.l2_reg),
            kernel_initializer=TruncatedNormal(stddev=0.1),
            bias_initializer=TruncatedNormal(stddev=0.1),
            activation="tanh",
            name="value"
        )(x)
        self.model = Model(input_x, [policy, value], name='reversi_model')
        self.compile_model()
        self.save_defender_model()
        shutil.copy(C.model_defender_path, C.model_challenger_path)
        # print(policy_out is value_out)

    def compile_model(self):
        self.model.compile(
            optimizer=SGD(lr=C.ln_rate, momentum=C.ln_momentum),
            loss=[loss_for_policy, loss_for_value],
            # loss_weights=[1, 1],
        )

    @staticmethod
    def _build_residual_block(x):
        in_x = x
        x = Conv2D(
            filters=C.cnn_filter_num,
            kernel_size=C.cnn_filter_size,
            kernel_initializer=TruncatedNormal(stddev=0.1),
            kernel_regularizer=l2(C.l2_reg),
            bias_initializer=TruncatedNormal(stddev=0.1),
            # activation='relu',
            padding='same',
            data_format='channels_last'
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(
            filters=C.cnn_filter_num,
            kernel_size=C.cnn_filter_size,
            kernel_initializer=TruncatedNormal(stddev=0.1),
            kernel_regularizer=l2(C.l2_reg),
            bias_initializer=TruncatedNormal(stddev=0.1),
            # activation='relu',
            padding='same',
            data_format='channels_last'
        )(x)
        x = BatchNormalization()(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    def save_defender_model(self):
        save_model(self.model, C.model_defender_path)

    def save_challenger_model(self):
        save_model(self.model, C.model_challenger_path)

    def load_challenger_model(self):
        self.model = load_model(C.model_challenger_path,
                                custom_objects={'loss_for_policy': loss_for_policy, 'loss_for_value': loss_for_value})

    def load_defender_model(self):
        self.model = load_model(C.model_defender_path,
                                custom_objects={'loss_for_policy': loss_for_policy, 'loss_for_value': loss_for_value})


def load_data_set():
    data = np.loadtxt(C.features_path)
    target = np.loadtxt(C.labels_path)
    x = data.reshape(-1, BOARD_SIZE, BOARD_SIZE, HISTORY_CHANNEL * 2 + 1)
    y = target[:, :-1]
    z = target[:, -1].reshape(-1, 1)
    return x, y, z


def loss_for_policy(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)


def loss_for_value(y_true, y_pred):
    return (y_true - y_pred)**2
    # return mean_squared_error(y_true, y_pred)


def train(epochs, batch_size=256, shuffle=True):
    x, y, z = load_data_set()
    model = load_model(C.model_challenger_path,
                       custom_objects={'loss_for_policy': loss_for_policy, 'loss_for_value': loss_for_value})
    model.fit(x=x, y=[y, z], batch_size=batch_size, epochs=epochs, shuffle=shuffle)


def test1():
    model = ReversiModel()
    # model.build_model()
    x, y, z = load_data_set()
    # print('states...', x.shape)
    # print('prob', y.shape)
    # print('winner', z.shape)
    # model.model.fit(x=x, y=[y, z], batch_size=C.batch_size, epochs=100, shuffle=True)
    # model.save_challenger_model()
    a = model.model.predict(x[0].reshape(-1, 8, 8, 3))[0].reshape(8,8)
    print(a)


def test3():
    x, y, z = load_data_set()
    model = load_model('../models/challenger.h5',
                       custom_objects={'loss_for_policy': loss_for_policy, 'loss_for_value': loss_for_value})
    tb_callback = TensorBoard(log_dir='../log/', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(x=x, y=[y, z], batch_size=256, epochs=10, shuffle=True, callbacks=[tb_callback], initial_epoch=0)


if __name__ == '__main__':
    # test1()
    ReversiModel().rebuild_model()


