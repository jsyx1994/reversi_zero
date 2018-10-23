# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""control self-play, train and evaluate"""

from game.common import *
from manager.config import *
from cnn.model import load_data_set, ReversiModel
import numpy as np
import cnn.config as C

def limit_history_data(sz):
    features = np.loadtxt(C.features_path)
    target = np.loadtxt(C.labels_path)
    maximum_training_data_size = opt_data_threshold << 1
    if sz > maximum_training_data_size:     # cut half
        features = features[-maximum_training_data_size:]
        target = target[-maximum_training_data_size:]

    with open(label_path, "w") as f:
        np.savetxt(f, target, fmt='%f')
    with open(features_path, "w") as f:
        np.savetxt(f, features, fmt='%i')     # (tup) can save the array row wise

def train(epochs, batch_size=batch_size, shuffle=True):
    x, y, z = load_data_set()
    sz = len(z)
    print('data size:', sz)
    limit_history_data(sz)
    if sz >= opt_data_threshold:
        model_lock.acquire()    # request the model
        try:    # load the model from the disk, need to block if other processes are using it
            reversi_model = ReversiModel(mode='challenger')
            model = reversi_model.model
        except Exception as e:  # except an error
            model_lock.release()
            print(e)
        else:   # not error then fit
            model_lock.release()
            model.fit(x=x, y=[y, z], batch_size=batch_size, epochs=epochs, shuffle=shuffle)
            model_lock.acquire()
            try:
                reversi_model.save_challenger_model()
            except Exception as e:
                print(e)
            finally:
                model_lock.release()
    else:
        print('not enough data, training sleep...')
        time.sleep(opt_wait_date_time)


def train4ever(epochs=opt_epochs, batch_size=batch_size, shuffle=True):
    while 1:
        train(
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle
        )
        print('have a rest %s secs:' % opt_idle_time)
        time.sleep(opt_idle_time)


if __name__ == '__main__':
    train(2)


