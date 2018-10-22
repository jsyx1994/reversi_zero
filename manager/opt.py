# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""control self-play, train and evaluate"""

# from multiprocessing import Process
from game.common import *
from manager.config import *
from cnn.model import load_data_set, ReversiModel


def train(epochs, batch_size=batch_size, shuffle=True):
    x, y, z = load_data_set()
    print(len(z))
    if len(z) >= 5000:
        # print(1)
        model_lock.acquire()    # request the model
        # print(2)
        try:    # load the model from the disk, need to block if other processes are using it
            reversi_model = ReversiModel(mode='challenger')
            model = reversi_model.model
            print(3)
            print(model is reversi_model.model)
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
        time.sleep(60)


def train4ever(epochs=opt_epochs, batch_size=batch_size, shuffle=True):
    while 1:
        train(
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle
        )


if __name__ == '__main__':
    pass
    # mp.set_start_method('forkserver')
    # import manager.eval as ev
    # Process(target=ev.eval, args=(1,)).start()


