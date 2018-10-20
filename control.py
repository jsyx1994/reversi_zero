# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""control self-play, train and evaluate"""

import manager.selfplay as sp
import manager.eval as ev
from manager import opt as op
from multiprocessing import Process


def main():
    # psp = Process(target=sp.play_games(1))
    # pev = Process(target=ev.eval(1))
    # pop = Process(target=op.train(1))
    # psp.start()
    # pev.start()
    # pop.start()
    op.train(1)
    # ev.eval(1)
    from conf import root_dir
    print(root_dir)


if __name__ == '__main__':
    main()


