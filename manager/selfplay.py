# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""this control the self-play of the game and """
from game.common import *
import json
from game.board import Board
import numpy as np
from cnn.model import ReversiModel
from mcts.uctAlg import UCTAlg



def init():
    global bp, wp, board, black_prompt, white_prompt, data, labels, reversi_model   # use occupied to update board history
    # global occupied
    black_prompt = '{"requests":[{"x":-1,"y":-1}],"responses":[]}'
    white_prompt = '{"requests":[],"responses":[]}'
    bp = json.loads(black_prompt)
    wp = json.loads(white_prompt)
    board = Board()
    labels = np.reshape([], (-1, BOARD_SIZE ** 2))
    data = np.reshape([], (-1, BOARD_SIZE ** 2 * (HISTORY_CHANNEL * 2 + 1)))

    model_lock.acquire()
    try:
        reversi_model.load_challenger_model()
        reversi_model.model.predict(np.random.random((1, BOARD_SIZE, BOARD_SIZE, 3)))
    finally:
        model_lock.release()

    if DEBUG:
        print('History info ...')
        # print_history()
        print()
        print('Feature info ...')
    calc_features(BLACK)


def record_steps(color_to_play, x, y):
    """record board state in selfplay.py, which maintains a Board instance"""
    global board
    board.disc_place(color_to_play, x, y)
    calc_features(-color_to_play)


def record_possibilities(pi):
    global labels
    labels = np.vstack((labels, pi))
    if DEBUG:
        print('the labels...', labels.shape)


def add_winner_and_writeIO(winner, turn):
    """1 if winner is current player which can be calculated by turn else 0 for tie or -1 for lost"""
    global data, labels, board
    winner_stack = []
    # turn = labels.shape[0]
    # winner = board.judge()
    for i in range(turn):
        current_player = BLACK if i % 2 == 0 else WHITE
        player_reward = 1 if current_player == winner else (0 if winner == TIE else -1)
        winner_stack.append(player_reward)
    winner_stack = np.asarray(winner_stack).reshape((turn, 1))

    labels = np.hstack((labels, winner_stack))
    if DEBUG:
        print('label shape...', labels.shape)
        print('final labels...', labels)

    with open("../labels.out", "a+") as f:
        np.savetxt(f, labels, fmt='%f')
    with open("../features.out", "a+") as f:
        np.savetxt(f, data, fmt='%i')     # (tup) can save the array row wise


def play(turn):
    """the black side first to place a disc, add output to black_prompt responses and to white_prompt requests, vice versa"""
    global black_prompt, white_prompt, res, pi, board
    if turn % 2 == 0:
        prompt, requests_add, responses_add, color_to_play = bp, bp, wp, BLACK
    else:
        prompt, requests_add, responses_add, color_to_play = wp, wp, bp, WHITE

    results = UCTAlg(predict_model=reversi_model, json=prompt, mode='stoch').run()

    res = results[0]      # (3, 2)
    pi = results[1]

    record_steps(color_to_play, res[0], res[1])
    record_possibilities(pi)

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


def calc_features(color_to_play):
    global board, data
    one_piece = [0 for _ in range((HISTORY_CHANNEL << 1) + 1)]
    one_feature = []
    bd = board.board
    for j in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            e = bd[i][j]
            if e == color_to_play:
                one_piece[0], one_piece[1] = 1, 0
            elif e == EMPTY:
                one_piece[0], one_piece[1] = 0, 0
            else:
                one_piece[0], one_piece[1] = 0, 1
            one_piece[-1] = 1 if color_to_play == BLACK else 0
            one_feature = np.hstack((one_feature, one_piece))
            # print(one_feature)
    data = np.vstack((data, one_feature))
    # print(data.shape)


def transform_history_matrix():
    """a board state can be transformed by the rotation and transposition to accelerate the data-generation"""
    def rotation():
        pass

    def transposition():
        pass


def play_games(rounds):
    """play games according to rounds para"""
    def play_one_game():
        init()
        global board
        turn = 0
        while not board.game_over:
            print('Round %d:' % turn)
            board.print_board()
            play(turn)
            turn += 1
            if DEBUG:
                print('Is game over? ', board.game_over)
        print('Round %d the terminal turn:' % turn)
        board.print_board()
        winner = board.judge()
        global data
        data = np.delete(data, -1, 0)
        add_winner_and_writeIO(winner, turn)
        print('winner is %s' % ('Black' if winner == BLACK else ('White' if winner == WHITE else TIE)))

    global reversi_model
    # the model will not change in one battle
    reversi_model = ReversiModel()

    if rounds < -1:
        print('please select correct rounds to begin')
    elif rounds is -1:
        while 1:
            play_one_game()
    else:
        for _ in range(rounds):
            play_one_game()


def idel():
    model_lock.acquire()
    time.sleep(10)
    model_lock.release()


def main():
    # from manager.eval import eval
    # import threading
    # t = threading.Thread(target=eval)
    # t.start()
    play_games(rounds=100)


if __name__ == '__main__':
    main()
