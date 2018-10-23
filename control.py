# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""control the parallelism"""
from multiprocessing import Process


def self_play_pool(pool_size=4):
    for i in range(pool_size):
        Process(target=sp.play_games, args=(-1,)).start()


if __name__ == '__main__':
    # threading.Thread(target=train, args=(1,)).start()
    # Process(target=train, args=(1,)).start()
    import manager.eval as ev
    import manager.opt as op
    import manager.selfplay as sp
    self_play_pool(1)
    # Process(target=op.train4ever, args=()).start()
    # Process(target=ev.eval4ever, args=()).start()

    # time.sleep(5)
    # model_lock.acquire()
    # print(1111)
    # model_lock.release()
