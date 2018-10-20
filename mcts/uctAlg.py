# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""This implements the UCT algorithm"""

import copy
import random
import math
import time
from game.common import *
from game.smart_random import SmartRandom
from mcts.ucTree import UCT
from game.board import Board
import numpy as np

C = 2
TIME_LIMIT = .1


class UCTAlg(object):
    """alg implements including expand, select, simulate and backup etc."""
    def __init__(self, predict_model, json=None, mode='comp'):
        """init the root node, configure the IO"""
        self.root_node = UCT()
        self.root_node.state = Board()
        self.mode = mode
        if json is None:
            json_input = input()
            self.root_node.state.restore_position_from_json_string(json_input=json_input)
        else:
            self.root_node.state.restore_position_from_json(json)
        self.working_node = self.root_node
        self.reversi_model = predict_model.model

    def run(self):
        start = time.time()
        while time.time() - start < TIME_LIMIT:
            self.select(self.working_node)
            # self.backup(best, 0, 0)
            # print('select use:', time.time() - start)
            # start = time.time()
            # start = time.time()
        # self.root_node.my_board.print_board()
        print('total simulate time:', self.root_node.visit_time)
        if self.mode == 'comp':
            return self.deterministical_decide()
        elif self.mode == 'stoch':
            return self.stochastical_decide()


        # print(c)
        # print('simulation use:', time.time() - start)
    def stochastical_decide(self):
        pi = np.zeros(shape=(BOARD_SIZE, BOARD_SIZE))
        children = self.root_node.my_children
        sum_all = self.root_node.visit_time - 1

        if (children[0].parent_action is None) or (sum_all == 0):
            return (-1, -1), list(pi.flatten())
        a = [x for x in range(len(children))]
        p = []

        print(end='       ')
        for c in children:
            coord = c.parent_action
            x, y = coord[0], coord[1]
            pi[x][y] = (c.visit_time/sum_all)
            # a.append(coord)
            p.append(pi[x][y])
            print(coord, end='             ')
        print()
        prob = [x * 100 for x in p]
        print(prob)
        print(cut_line)
        selected_index = np.random.choice(a=a, p=p)
        action = children[selected_index].parent_action
        return action, list(pi.flatten())

    def deterministical_decide(self):
        pi = np.zeros(shape=(BOARD_SIZE, BOARD_SIZE))
        children = self.root_node.my_children
        best = children[0]
        sum_all = self.root_node.visit_time - 1
        if (best.parent_action is None) or (sum_all == 0):  # according to the alg, 'no action' always have a node, and this is a fake child
            # print('(-1, -1)')
            # print(list(pi.flatten()))
            return (-1, -1), list(pi.flatten())
        for c in children:
            coord = c.parent_action
            x = coord[0]
            y = coord[1]
            pi[x][y] = (c.visit_time/sum_all)
            if c.visit_time > best.visit_time:
                best = c

        return best.parent_action, list(pi.flatten())
        # print(best.parent_action)
        # print(list(pi.flatten()))

    def set_working_node(self, node):
        """be flexible to adapt to different situations, aka. may asyc"""
        self.working_node = node

    def select(self, working_node):
        while working_node.my_children:    # while has child
            working_node = self.choose_best_child(uct=working_node)     # select until has no child
        # start = time.time()
        self.expand(working_node)
        # print('expand use:', time.time() - start)

    def expand(self, uct):
        """while the selection encounter a leaf node, expand its children and choose the best child of this node"""
        state = uct.my_board
        if state.game_over:
            # print('game over')
            self.backup(uct, 1, state.side_color)
        feature_input = self.calc_features(state)
        prediction = self.reversi_model.predict(feature_input.reshape(-1, BOARD_SIZE, BOARD_SIZE, 3))
        p = prediction[0].reshape(8, 8)
        v = prediction[1].reshape(1)
        self.backup(uct, v[0], state.side_color)    # 使用多*程？
        moves = state.generate_moves(uct.state.side_color)
        children = uct.my_children
        if not moves:
            node = UCT()    # if it has no move to go, then create a fake node which just change the color
            node.initialize_state(uct.state)    # also copy the board history and occupied discs
            state = node.state
            state.turn_color()
            node.parent_action = None
            node.parent = uct

            node.psa = 1    # pass for sure
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

                node.psa = p[move[0]][move[1]]
                children.append(node)
        # return self.choose_best_child(uct=uct)

    def simulate(self, uct):
        simu_board = copy.deepcopy(uct.my_board)    # not time-consuming
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

    def backup(self, uct, reward, player_color):
        """update the reward of the player_color"""
        while uct is not None:
            uct.visit_time += 1
            board = uct.my_board
            # if player_color == BLACK:    # black wins the game
            #     if board.side_color == BLACK:     # because
            #         uct.total_reward -= reward
            #     else:
            #         uct.total_reward += reward
            # elif player_color == WHITE:
            #     if board.side_color == WHITE:
            #         uct.total_reward += reward
            #     else:
            #         uct.total_reward -= reward
            board_color = board.side_color
            if player_color == board_color:
                uct.total_reward += reward
            else:
                pass
                # uct.total_reward -= reward
            uct = uct.my_parent

    def add_noise(self):
        pass

    def APV_equation(self, uct):
        q_v_ = uct.total_reward
        nsa = uct.visit_time
        sigma_nsb = uct.my_parent.visit_time - 1
        psa = uct.psa
        if nsa == 0:
            return float('inf')
        if uct is self.root_node:
            pass
        equation = q_v_/nsa + C * psa * math.sqrt(sigma_nsb) / (nsa + 1)
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

    @staticmethod
    def calc_features(s):
        one_piece = [0 for _ in range((HISTORY_CHANNEL << 1) + 1)]
        one_feature = []
        bd = s.board
        for j in range(BOARD_SIZE):
            for i in range(BOARD_SIZE):
                e = bd[i][j]
                if e == s.side_color:
                    one_piece[0], one_piece[1] = 1, 0
                elif e == EMPTY:
                    one_piece[0], one_piece[1] = 0, 0
                else:
                    one_piece[0], one_piece[1] = 0, 1
                one_piece[-1] = 1 if s.side_color == BLACK else 0
                one_feature = np.hstack((one_feature, one_piece))
        return one_feature
