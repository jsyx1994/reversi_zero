# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import json
import numpy as np

from game.common import *
from manager.config import eval_rounds
from mcts.uctAlg import UCTAlg
from cnn.model import ReversiModel
from game.board import Board
import shutil
from cnn.config import model_challenger_path, model_defender_path
from manager.config import eval_log_path
import datetime

IDLE_TIME = 600


hand = BLACK

from manager.config import selfplay_monitor

def init():
    global bp, wp, board, black_prompt, white_prompt    # use occupied to update board history
    # global occupied
    black_prompt = '{"requests":[{"x":-1,"y":-1}],"responses":[]}'
    white_prompt = '{"requests":[],"responses":[]}'
    bp = json.loads(black_prompt)
    wp = json.loads(white_prompt)
    board = Board()
    if DEBUG:
        print('History info ...')
        # print_history()
        print()
        print('Feature info ...')


def record_steps(color_to_play, x, y):
    """record board state in selfplay.py, which maintains a Board instance"""
    global board
    board.disc_place(color_to_play, x, y)


def play(turn):
    """the black side first to place a disc, add output to black_prompt responses and to white_prompt requests, vice versa"""
    global black_prompt, white_prompt, res, pi, board, hand, challenger_model, defender_model
    if turn % 2 == 0:
        prompt, requests_add, responses_add, color_to_play = bp, bp, wp, BLACK
        if hand == BLACK:   # the first player is black
            model = challenger_model
        else:
            model = defender_model
    else:
        prompt, requests_add, responses_add, color_to_play = wp, wp, bp, WHITE
        if hand == WHITE:
            model = challenger_model
        else:
            model = defender_model

    results = UCTAlg(predict_model=model, json=prompt, mode='comp').run()

    res = results[0]      # (3, 2)
    pi = results[1]

    record_steps(color_to_play, res[0], res[1])

    dct = {'x': res[0], 'y': res[1]}
    requests_add["responses"].append(dct)
    responses_add["requests"].append(dct)
    # black_prompt, white_prompt = json.dumps(bp), json.dumps(wp)
    if DEBUG:
        print('The subprocess responses ...', res)
        print('the possibility...', pi)
        print('%s round has played' % ('black' if color_to_play == BLACK else 'white'))
        # print('Black prompt...', black_prompt)
        # print('White prompt...', white_prompt)


def play_games(rounds):
    """play games according to rounds para"""
    def play_one_game():
        init()
        global board
        turn = 0
        while not board.game_over:
            if selfplay_monitor:
                print('Round %d:' % turn)
                board.print_board()
                print()
            play(turn)
            turn += 1
            if DEBUG:
                print('Is game over? ', board.game_over)
        global hand
        if selfplay_monitor:
            print('Round %d the terminal turn:' % turn)
            board.print_board()
        winner = board.judge()
        if selfplay_monitor:
            print(hand, winner)
        global challenger_wins
        if hand == winner:
            challenger_wins += 1
            print('challenger wins')
        hand *= -1  # one battle ends, to evaluate the model accurately, models should exchange hands
        print('winner is %s' % ('Black' if winner == BLACK else ('White' if winner == WHITE else 'TIE')))

    if rounds < -1:
        print('please select correct rounds to begin')
    elif rounds is -1:
        while 1:
            play_one_game()
    else:
        for _ in range(rounds):
            play_one_game()


def update_generation():
    model_lock.acquire()
    try:
        shutil.copy(src=model_challenger_path, dst=model_defender_path)
    except Exception as e:
        model_lock.release()
        print('update model failed...')
    else:
        print('model updated')
        ts = datetime.datetime().now()
        with open(eval_log_path, 'a') as f:
            f.write(str(ts) + ' model updated\n')
            f.close()
    finally:
        model_lock.release()


def eval(rounds):
    global challenger_wins
    challenger_wins = 0
    model_lock.acquire()
    global challenger_model, defender_model
    try:
        challenger_model = ReversiModel(mode='challenger')
        defender_model = ReversiModel(mode='defender')

        challenger_model.model.predict(np.random.random((1, BOARD_SIZE, BOARD_SIZE, 3)))
        defender_model.model.predict(np.random.random((1, BOARD_SIZE, BOARD_SIZE, 3)))
    except Exception as e:
        model_lock.release()
        print(e)
    else:
        model_lock.release()
        N = rounds
        print('Evaluate starts, rounds = ', N)
        play_games(N)
        challenger_winning_rate = challenger_wins/N
        if challenger_winning_rate > .55:
            update_generation()
        print('challenger winning rate:', challenger_wins/N)


def eval4ever(rounds=eval_rounds):
    while 1:
        eval(rounds=rounds)
        print('Have a {} seconds rest...'.format(IDLE_TIME))
        time.sleep(IDLE_TIME)


if __name__ == '__main__':
    eval(1)
    # if challenger_wins/N > 0.55:
    #     update_generation()
