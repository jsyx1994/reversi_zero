# -*- coding: utf-8 -*-
# !/usr/bin/env python3

from game.board import Board, WHITE, BLACK,BOARD_SIZE
from cnn.config import  channel
import pure_MCTS
import mcts.uctAlg
import json
import numpy as np
from cnn.model import ReversiModel
player = ReversiModel(mode='defender')
player.model.predict(np.random.random((1, BOARD_SIZE, BOARD_SIZE, channel)))

class Play(object):
    def __init__(self):
        # global bp, wp, board, black_prompt, white_prompt, features, labels, reversi_model   # use occupied to update board history
        # json style to input

        black_prompt = '{"requests":[{"x":-1,"y":-1}],"responses":[]}'
        white_prompt = '{"requests":[],"responses":[]}'
        self.bp = json.loads(black_prompt)
        self.wp = json.loads(white_prompt)
        self.board = Board()

    def play(self, turn):
        """the black side first to place a disc, add output to black_prompt responses and to white_prompt requests, vice versa"""
        # global black_prompt, white_prompt, res, pi, board
        if turn % 2 == 0:
            prompt, requests_add, responses_add, color_to_play = self.bp, self.bp, self.wp, BLACK
            print("pure")
            res = pure_MCTS.UCTAlg(json=prompt).run(time_limit=1)
        else:
            prompt, requests_add, responses_add, color_to_play = self.wp, self.wp, self.bp, WHITE
            print("alpha")
            res = mcts.uctAlg.UCTAlg(predict_model=player, json=prompt, mode='comp').run(time_limit=1)[0]
        print(res)
        self.board.disc_place(color_to_play, res[0], res[1])  # record steps to  board

        dct = {'x': res[0], 'y': res[1]}
        requests_add["responses"].append(dct)
        responses_add["requests"].append(dct)

if __name__ == '__main__':
    play = Play()
    turn = 0
    while not play.board.game_over:
        print('Round %d:' % turn)
        play.board.print_board()
        play.play(turn)
        turn += 1

    play.board.print_board()
    winner = play.board.judge()
    print('winner is %s' % ('Black' if winner == BLACK else ('White' if winner == WHITE else 'TIE')))