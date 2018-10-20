# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""This is the rule operation of the reversi board game"""
import functools
import numpy as np
import json
import time
import random
from game.common import *
effective_points = [[0 for x in range(2)] for x in range(BOARD_SIZE)]


class Board(object):
    """This defines the class of thre reversi board"""
    def __init__(self):
        self.board = []
        self.black_piece = 0
        self.white_piece = 0
        # self.empty_poses = copy.copy(BOARD_ELE_INDEX)
        self.side_color = EMPTY
        # self.BOARD_HISTORY = np.zeros(shape=(BOARD_SIZE, BOARD_SIZE, HISTORY_CHANNEL), dtype=np.int32)
        # self.occupied = None

        self.initialize_board()
        random.shuffle(BOARD_ELE_INDEX)     # shuffle the searching direction and position to make it more efficient
        random.shuffle(DIR)

    def print_board(self):
        """debug the output of the current state"""
        print_sp = functools.partial(print, end='  ')
        print_sp(' ')
        for i in range(BOARD_SIZE):
            print_sp(i)
        print()
        for i in range(BOARD_SIZE):
            print_sp(i)
            for j in range(BOARD_SIZE):
                e = self.board[j][i]
                print_sp('●') if e == BLACK else print_sp('○') if e == WHITE else print_sp('·')
            print()

    def initialize_board(self):
        """put the beginning state of the game"""
        self.board = np.zeros(shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int)   # another way of defining board: [[for x in range(cm.BOARD_SIZE)] for x in range(cm.BOARD_SIZE)]
        self.board[3][3] = self.board[4][4] = WHITE
        self.board[4][3] = self.board[3][4] = BLACK
        self.black_piece = 2
        self.white_piece = 2

        # self.occupied = {(3, 3), (4, 4), (3, 4), (4, 3)}
        # self.BOARD_HISTORY[3][3][0] = WHITE
        # self.BOARD_HISTORY[4][4][0] = WHITE
        # self.BOARD_HISTORY[3][4][0] = BLACK
        # self.BOARD_HISTORY[4][3][0] = BLACK

    def set_color(self, color):
        self.side_color = color

    def turn_color(self):
        self.set_color(-self.side_color)

    @staticmethod
    def boarder_check(ix, iy):
        return 0 <= ix < 8 and 0 <= iy < 8

    @staticmethod
    def move_step(x, y, d0, d1):
        return x + d0, y + d1

    def dir_prob(self, x, y, d):
        pass

    def disc_place(self, color, x, y, check_only=False):
        # start = time.time()
        # self.board[x][y] != EMPTY
        # print("self.board use:", time.time() - start)
        if x < 0:
            return False
        if self.board[x][y] != EMPTY:
            return False
        # self.empty_poses.remove((x, y))     # delete the possible from search the
        is_valid = False
        c = 0
        start = time.time()

        for d in DIR:
            i, j = self.move_step(x, y, d[0], d[1])
            # i = x + d[0]
            # j = y + d[1]
            curr_count = 0
            while 1:
                if not self.boarder_check(i, j):
                    curr_count = 0
                    break
                check = self.board[i][j]
                if check == -color:
                    curr_count += 1
                    effective_points[curr_count][0] = i
                    effective_points[curr_count][1] = j
                    i, j = self.move_step(i, j, d[0], d[1])
                elif check == EMPTY:
                    curr_count = 0
                    break
                else:
                    break

            if curr_count != 0:
                is_valid = True
                if check_only:
                    return is_valid
                if color == BLACK:
                    self.black_piece += curr_count
                    self.white_piece -= curr_count
                else:
                    self.black_piece -= curr_count
                    self.white_piece += curr_count
                while curr_count > 0:
                    i = effective_points[curr_count][0]
                    j = effective_points[curr_count][1]
                    self.board[i][j] *= -1
                    curr_count -= 1
        # print('for d in dir:', time.time() - start)
        if is_valid:
            self.board[x][y] = color
            if color == BLACK:
                self.black_piece += 1
            else:
                self.white_piece += 1
        return is_valid

    def restore_position_from_json_string(self, json_input):
        """using json to control communication"""
        line_input = json.loads(json_input)
        self.restore_position_from_json(line_input)

    def restore_position_from_json(self, line_input):
        requests = line_input['requests']
        responses = line_input['responses']
        if requests[0]['x'] == -1:
            self.side_color = BLACK
        else:
            self.side_color = WHITE
        turn = len(responses)
        for t in range(turn):   # history info
            x = requests[t]['x']
            y = requests[t]['y']
            if x >= 0:
                self.disc_place(-self.side_color, x, y)
                # self.occupied.add((x, y))
                # self.history_update()

            x = responses[t]['x']
            y = responses[t]['y']
            if x >= 0:
                self.disc_place(self.side_color, x, y)
                # self.occupied.add((x, y))
                # self.history_update()

        x = requests[turn]['x']
        y = requests[turn]['y']
        if x >= 0:
            self.disc_place(- self.side_color, x, y)


    def generate_moves(self, side_color):
        # start = time.time()
        # moves = list(filter(lambda indexs: self.disc_place(side_color, indexs[0], indexs[1], check_only=True), BOARD_ELE_INDEX))
        moves = []
        for indexs in BOARD_ELE_INDEX:
            if self.disc_place(side_color, indexs[0], indexs[1], check_only=True):
                # yield indexs
                moves.append(indexs)
        # print('generate_moves use:', time.time()-start)
        return moves

    def check_if_has_valid_move(self, color):
        for indexs in BOARD_ELE_INDEX:
            if self.disc_place(color, indexs[0], indexs[1], check_only=True):
                return True
        return False

    def is_valid_move(self, side_color, x, y):
        """check if the move of the side_color at the coord is valid"""
        return self.disc_place(side_color, x, y, check_only=True)

    def judge(self):
        if self.black_piece > self.white_piece:
            return BLACK
        elif self.white_piece > self.black_piece:
            return WHITE
        else:
            return TIE

    @property
    def game_over(self):
        return (self.black_piece + self.white_piece >= BOARD_SIZE ** 2) or (not(
                    self.check_if_has_valid_move(BLACK) or self.check_if_has_valid_move(WHITE)))


def main():
    b = Board()
    Input = input()
    b.restore_position_from_json_string(Input)
    b.print_board()
    # print(b.BOARD_HISTORY)


if __name__ == '__main__':
    main()
