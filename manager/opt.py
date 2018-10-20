# -*- coding: utf-8 -*-
# !/usr/bin/env python3

from game.common import *
from cnn.config import batch_size
from cnn.model import load_data_set
import threading
from cnn.model import ReversiModel


def train(epochs, batch_size=batch_size, shuffle=True):
    x, y, z = load_data_set()
    print(len(z))
    if len(z) >= 5000:
        model_lock.acquire()    # request the model
        try:    # load the model from the disk, need to block if other processes are using it
            reversi_model = ReversiModel(mode='challenger')
            model = reversi_model.model
            # print(model is reversi_model.model)
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
        print('training sleep...')
        time.sleep(60)


if __name__ == '__main__':
    threading.Thread(target=train, args=(1,)).start()
    time.sleep(5)
    model_lock.acquire()
    print(1111)
    model_lock.release()
