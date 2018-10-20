# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""execute a single decision"""

from game.board import Board
from mcts.uctAlg import UCTAlg
import numpy as np

from cnn.model import ReversiModel
reversi_model = ReversiModel()
reversi_model.model.predict(np.random.random((1, 8, 8, 3)))


def main():
    board = Board()
    json_input = input()
    board.restore_position_from_json_string(json_input=json_input)
    board.print_board()
    # your strategy taken


def uct_algorithm():
    uct_alg = UCTAlg(reversi_model)
    uct_alg.run()
    print()


if __name__ == '__main__':
    uct_algorithm()
    # main()
