# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""this control the self-play of the game and """

from game.common import *
from conf import *
import json
from game.board import Board
import numpy as np
from cnn.model import ReversiModel
from mcts.uctAlg import UCTAlg
from manager.config import features_path, label_path, error_log

from manager.config import selfplay_monitor
from manager.config import eval_timelimit, selfplay_timelimit

class Play(object):
    def __init__(self):
        # global bp, wp, board, black_prompt, white_prompt, features, labels, reversi_model   # use occupied to update board history
        # json style to input

        black_prompt = '{"requests":[{"x":-1,"y":-1}],"responses":[]}'
        white_prompt = '{"requests":[],"responses":[]}'
        self.bp = json.loads(black_prompt)
        self.wp = json.loads(white_prompt)
        self.board = Board()
        self.labels = np.reshape([], (-1, BOARD_SIZE ** 2))  # one turn has (8,8,3) -> 64 * 3
        self.features = np.reshape([], (-1, BOARD_SIZE ** 2 * (HISTORY_CHANNEL * 2 + 1)))  # one turn has 64 + 1
        # all_objects = muppy.get_objects()
        # sum1 = summary.summarize(all_objects)
        # summary.print_(sum1)

    def record_steps(self, color_to_play):
        """record board state in selfplay.py, which maintains a Board instance"""
        self.calc_features(color_to_play)  # produce input data before change state

    def record_possibilities(self, pi):
        self.labels = np.vstack((self.labels, pi))
        if DEBUG:
            print('the labels...', self.labels.shape)

    def add_winner_and_writeIO(self, winner, turn):
        """1 if winner is current player which can be calculated by turn else 0 for tie or -1 for lost"""
        winner_stack = []
        # turn = labels.shape[0]
        # winner = board.judge()
        for i in range(turn):
            current_player = BLACK if i % 2 == 0 else WHITE
            player_reward = 1 if current_player == winner else (0 if winner == TIE else -1)
            winner_stack.append(player_reward)
        winner_stack = np.asarray(winner_stack).reshape((turn, 1))

        self.labels = np.hstack((self.labels, winner_stack))
        if DEBUG:
            print('label shape...', self.labels.shape)
            print('final labels...', self.labels)

        data_lock.acquire()
        try:
            with open(label_path, "a+") as f:
                np.savetxt(f, self.labels, fmt='%f')
            with open(features_path, "a+") as f:
                np.savetxt(f, self.features, fmt='%i')  # (tup) can save the array row wise
        except Exception as e:
            print(e)
        finally:
            data_lock.release()

    def play(self, player1, player2, turn, p1isp2):
        """the black side first to place a disc, add output to black_prompt responses and to white_prompt requests, vice versa"""
        # global black_prompt, white_prompt, res, pi, board
        if turn % 2 == 0:
            prompt, requests_add, responses_add, color_to_play = self.bp, self.bp, self.wp, BLACK
            current2play = player1
        else:
            prompt, requests_add, responses_add, color_to_play = self.wp, self.wp, self.bp, WHITE
            current2play = player2

        if p1isp2:  # self play, use stochastic policy
            results = UCTAlg(predict_model=current2play, json=prompt, mode='stoch').run(time_limit=selfplay_timelimit)
        else:   # eval, use deterministic policy
            results = UCTAlg(predict_model=current2play, json=prompt, mode='comp').run(time_limit=eval_timelimit)

        res, pi = results[0], results[1]  # (3, 2)

        if p1isp2:
            self.record_steps(color_to_play)
            self.record_possibilities(pi)

        self.board.disc_place(color_to_play, res[0], res[1])  # record steps to  board

        dct = {'x': res[0], 'y': res[1]}
        requests_add["responses"].append(dct)
        responses_add["requests"].append(dct)
        if DEBUG:
            print('The subprocess responses ...', res)
            print('the possibility...', pi)
            print('%s round has played' % ('black' if color_to_play == BLACK else 'white'))

    def calc_features(self, color_to_play):
        one_piece = [0 for _ in range((HISTORY_CHANNEL << 1) + 1)]
        one_feature = []
        bd = self.board.board
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
        self.features = np.vstack((self.features, one_feature))
        # print(data.shape)

    def transform_history_matrix(self):
        """a board state can be transformed by the rotation and transposition to accelerate the data-generation"""
        def rotation():
            pass

        def transposition():
            pass


class SelfPlay(object):
    """An instance in one battle"""

    def __init__(self, player1='defender', player2='defender'):
        """
        use defender model to continue generating data until it was replaced by challenger model
        default self play <=> player1 is player2
        """
        memory_gpu(.001)  # set the memory of gpu
        assert (player1 is 'challenger') or (player1 is 'defender')
        assert (player2 is 'defender') or (player2 is 'challenger')
        self.p1 = player1
        self.p2 = player2
        model_lock.acquire()
        try:
            # the model will not change in one battle
            self.player1 = ReversiModel(mode=player1)  # every time start a new battle, reload the newest model
            self.player2 = ReversiModel(mode=player2)
        except Exception as e:
            model_lock.release()
            print(e)
        else:
            self.player1.model.predict(np.random.random((1, BOARD_SIZE, BOARD_SIZE, 3)))  # initialize the generator to accelerate the simulation
            self.player2.model.predict(np.random.random((1, BOARD_SIZE, BOARD_SIZE, 3)))
        finally:
            model_lock.release()

    def play_one_game(self):
        play = Play()
        turn = 0
        while not play.board.game_over:
            if selfplay_monitor:
                print('Round %d:' % turn)
                play.board.print_board()
            play.play(self.player1, self.player2, turn, self.p1 is self.p2)
            turn += 1
            if DEBUG:
                print('Is game over? ', play.board.game_over)
        if selfplay_monitor:
            print('Round %d the terminal turn:' % turn)
            play.board.print_board()
        winner = play.board.judge()
        if self.p1 is self.p2:    # only in self play program produce input and output
            print('record')
            play.add_winner_and_writeIO(winner, turn)
        if selfplay_monitor:
            print('winner is %s' % ('Black' if winner == BLACK else ('White' if winner == WHITE else 'TIE')))
        return winner


def play_games(rounds):
    """play games according to rounds """
    try:
        if rounds < -1:
            print('please select correct rounds to begin')
        elif rounds is -1:
            while 1:
                SelfPlay().play_one_game()
        else:
            for _ in range(rounds):
                SelfPlay().play_one_game()
    except Exception as e:
        log_lock.acquire()
        try:
            with open(error_log, 'a+') as f:
                f.write(str(e) + '\n')
        except FileNotFoundError:
            pass
        finally:
            log_lock.release()
        # print(e)


def main():
    # play_games(-1)
    SelfPlay().play_one_game()


if __name__ == '__main__':
    main()
