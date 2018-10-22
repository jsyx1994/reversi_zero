# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""control self-play, train and evaluate"""

from game.common import *
from manager.config import *
from cnn.model import load_data_set, ReversiModel


def train(epochs, batch_size=batch_size, shuffle=True):
    x, y, z = load_data_set()
    print('data size:', len(z))
    if len(z) >= opt_data_threshold:
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
    train(100)


