# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""This implements the UCT algorithm"""
import functools
import random
import math
import numpy as np


import time
import copy
import json

BOARD_SIZE = 8

WHITE = -1
BLACK = 1
EMPTY = 0

TIE = 0

DEBUG = 0

COMMUNICATION = 'Json'

HISTORY_CHANNEL = 1
DIR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BOARD_ELE_INDEX = list((i, j) for i in range(0, BOARD_SIZE) for j in range(0, BOARD_SIZE) if (i, j) != (3, 3) or (i, j) != (4, 4) or (i, j) != (3, 4) or (i, j) != (4, 3))

cut_line = '-------------------------------------------------------------------------------------\n'
effective_points = [[0 for x in range(2)] for x in range(BOARD_SIZE)]


class UCT(object):
    """Tree definition but not including expand, select, simulate and backup"""
    def __init__(self):
        # common tree structure
        self.state = None   # Board information from game.board, including black and white pieces and the whole board info, used to identity a node
        self.parent = None  # the parent of the node
        self.children = []  # the children of the node
        self.parent_action = None   # parent's action to reach this state, used to make decisions, a binary tuple
        # self.weight = float('inf')  # initialize the max fla

        # UCT characters
        self.visit_time = 0     # the times a node is visited throughout the whole process
        self.total_reward = 0   # how many times winning throughout whole process
        self.psa = 0    # prior experience

    # common operations of tree
    def initialize_state(self, state):
        self.state = copy.deepcopy(state)

    def set_parent(self, uct_node):
        self.parent = uct_node

    def add_child(self, uct_node):
        self.children.append(uct_node)

    def rand_choose_child(self):
        return random.choice(self.children)

    @property
    def mean_reward(self):
        if self.visit_time == 0:
            return 0
        else:
            return self.total_reward/self.visit_time

    @property
    def my_parent(self):
        return self.parent

    @property
    def my_board(self):
        return self.state

    @property
    def my_children(self):
        return self.children




class Board(object):
    """This defines the class of the reversi board"""
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
        center = int(BOARD_SIZE / 2)
        self.board[center-1][center-1] = self.board[center][center] = WHITE   # place the board according to position
        self.board[center][center-1] = self.board[center-1][center] = BLACK
        self.black_piece = 2
        self.white_piece = 2

    def set_color(self, color):
        self.side_color = color

    def turn_color(self):
        self.set_color(-self.side_color)

    @staticmethod
    def boarder_check(ix, iy):
        """
        Check the frontier
        :param ix:
        :param iy:
        :return:
        """
        return 0 <= ix < BOARD_SIZE and 0 <= iy < BOARD_SIZE

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


class SmartRandom(object):
    def __init__(self):
        pass

    @staticmethod
    def smart_random(board, threshold=200):
        count = 0
        for _ in range(threshold):
            rdx = random.randint(0, 8 - 1)
            rdy = random.randint(0, 8 - 1)
            count += 1
            if board.disc_place(color=board.side_color, x=rdx, y=rdy, check_only=True):
                return rdx, rdy
        moves = board.generate_moves(board.side_color)
        if moves:
            return random.choice(moves)
        else:
            return None


class UCTAlg(object):
    """alg implements including expand, select, simulate and backup etc."""

    def __init__(self, json=None, mode='comp'):
        """init the root node, configure the IO"""
        uct = UCT()
        uct.state = Board()

        if json is None:
            json_input = input()
            uct.state.restore_position_from_json_string(json_input=json_input)
        else:
            uct.state.restore_position_from_json(json)
        self.root_node = uct
        self.mode = mode
        self.working_node = self.root_node

    def run(self, time_limit):
        start = time.time()
        while time.time() - start < time_limit:
            self.select(self.working_node)
        if self.mode == 'comp':
            return self.deterministical_decide()
        elif self.mode == 'stoch':
            return self.stochastical_decide()

    def set_working_node(self, node):
        """be flexible to adapt to different situations, aka. may asyc"""
        self.working_node = node

    def select(self, working_node):
        while working_node.my_children:  # while has child
            working_node = self.choose_best_child(uct=working_node)  # select until has no child
        # start = time.time()
        self.simulate(working_node)
        self.expand(working_node)
        # print('expand use:', time.time() - start)

    def expand(self, uct):
        """while the selection encounter a leaf node, expand its children and choose the best child of this node"""
        state = uct.my_board
        if state.game_over:
            winner = state.judge()
            reward = 1 if winner == state.side_color else -1
            self.backup(uct, reward, winner)  # 操蛋这里的winner之前写成STATE.SIDE_COLOR了
        moves = state.generate_moves(uct.state.side_color)  # 可以与神经网络输出并行

        children = uct.my_children
        if not moves:
            node = UCT()  # if it has no move to go, then create a fake node which just change the color
            node.initialize_state(uct.state)  # also copy the board history and occupied discs
            node.state.turn_color()
            node.parent_action = None
            node.parent = uct

            node.psa = 1  # pass for sure
            children.append(node)
        else:
            for move in moves:
                node = UCT()
                curr_state = uct.state
                node.initialize_state(curr_state)
                new_state = node.state
                new_state.disc_place(curr_state.side_color, x=move[0], y=move[1])
                new_state.turn_color()
                node.parent_action = move
                node.parent = uct
                children.append(node)
        # return self.choose_best_child(uct=uct)

    def simulate(self, uct):
        simu_board = copy.deepcopy(uct.my_board)  # not time-consuming
        while not simu_board.game_over:
            action = SmartRandom.smart_random(simu_board)
            if action:
                simu_board.disc_place(simu_board.side_color, action[0], action[1])
                simu_board.turn_color()
            else:
                simu_board.turn_color()
        winner = simu_board.judge()
        # print('winner', winner)
        self.backup(uct, 1, winner)

    def backup(self, uct, reward, winner):
        """update the reward of the player_color"""
        while uct is not None:
            uct.visit_time += 1
            board = uct.my_board
            node_color = board.side_color
            if winner == node_color:
                uct.total_reward -= reward
            else:
                uct.total_reward += reward  #正奖励应该更新赢家节点之后的动作
                # uct.total_reward -= reward
            uct = uct.my_parent

    def deterministical_decide(self):
        """choose the actions from the root according to the maximum visiting times"""
        children = self.root_node.my_children
        best = children[0]
        for c in children:
            if c.visit_time > best.visit_time:
                best = c
        return best.parent_action

    def APV_equation(self, uct):
        """return Q + U"""
        q_v_ = uct.total_reward
        nsa = uct.visit_time
        sigma_nsb = uct.my_parent.visit_time - 1
        psa = uct.psa
        if nsa == 0:
            return float('inf')
        equation = q_v_ / nsa + 2 * psa * math.sqrt(sigma_nsb) / nsa
        return equation

    def get_weight(self, uct):
        return self.APV_equation(uct)

    def choose_best_child(self, uct):
        children = uct.my_children
        best_child = random.choice(children)
        best_weight = self.get_weight(best_child)
        for child in children:
            child_weight = self.get_weight(child)
            if child_weight > best_weight:
                best_child = child
                best_weight = child_weight
        return best_child


if __name__ == '__main__':

    # print(reversi_model.model.predict(np.random.random((1, BOARD_SIZE, BOARD_SIZE, 3))))
    uct = UCTAlg()
    uct.run(time_limit=1)
    decision = uct.deterministical_decide()
    try:
        prompt = '{"response":{"x":' + str(decision[0]) + ',' + '"y":' + str(decision[1]) + '}}'
    except Exception as e:
        prompt = '{"response":{"x":' + str(-1) + ',' + '"y":' + str(-1) + '}}'
    print(prompt)
    # print(json.loads(prompt), end='')
    # del reversi_model
    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)
