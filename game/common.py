# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""This defines the common variables in the project"""
BOARD_SIZE = 8

WHITE = -1
BLACK = 1
EMPTY = 0

TIE = 0

DEBUG = 0

COMMUNICATION = 'Json'

HISTORY_CHANNEL = 1/2
DIR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BOARD_ELE_INDEX = list((i, j) for i in range(0, BOARD_SIZE) for j in range(0, BOARD_SIZE) if (i, j) != (3, 3) or (i, j) != (4, 4) or (i, j) != (3, 4) or (i, j) != (4, 3))

cut_line = '-------------------------------------------------------------------------------------\n'
