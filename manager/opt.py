# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""control self-play, train and evaluate"""

from conf import model_lock, data_lock, log_lock
from manager.config import *
from cnn.model import load_data_set, ReversiModel
import numpy as np
import cnn.config as C
import time

def limit_history_data():
    data_lock.acquire()
    try:
        features = np.loadtxt(C.features_path)
        target = np.loadtxt(C.labels_path)
    finally:
        data_lock.release()
    maximum_training_data_size = opt_data_threshold << 1
    sz = len(target)
    if sz > maximum_training_data_size:     # cut half
        features = features[-maximum_training_data_size:]
        target = target[-maximum_training_data_size:]

    data_lock.acquire()
    try:
        with open(label_path, "w") as f:
            np.savetxt(f, target, fmt='%f')
            f.close()
        with open(features_path, "w") as f:
            np.savetxt(f, features, fmt='%i')     # (tup) can save the array row wise
            f.close()
    except Exception as e:
        print(e)
    finally:
        data_lock.release()

def train(epochs, batch_size=batch_size, shuffle=True):
    limit_history_data()
    x, y, z = load_data_set()
    sz = len(z)
    print('data size:', sz)
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
        try:
            train(
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle
            )
        except Exception as e:
            log_lock.acquire()
            try:
                with open(error_log, 'a+') as f:
                    f.write(str(e) + '\n')
            except FileNotFoundError:
                pass
            finally:
                log_lock.release()
            # print(e)
        print('have a rest %s secs:' % opt_idle_time)
        time.sleep(opt_idle_time)


if __name__ == '__main__':
    train(2)


