# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""this contains the different strategies"""

import game.common as cm
import random


class SmartRandom(object):
    def __init__(self):
        pass

    @staticmethod
    def smart_random(board, threshold=200):
        count = 0
        for _ in range(threshold):
            rdx = random.randint(0, cm.BOARD_SIZE - 1)
            rdy = random.randint(0, cm.BOARD_SIZE - 1)
            count += 1
            if board.disc_place(color=board.side_color, x=rdx, y=rdy, check_only=True):
                return rdx, rdy
        moves = board.generate_moves(board.side_color)
        if moves:
            return random.choice(moves)
        else:
            return None
