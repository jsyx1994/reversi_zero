# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""This implements the UCT algorithm"""

import random
import math
from game.common import *
from mcts.ucTree import UCT
from game.board import Board
import numpy as np

from mcts.config import *
from manager.config import selfplay_monitor
import time

class UCTAlg(object):
    """alg implements including expand, select, simulate and backup etc."""

    def __init__(self, predict_model, json=None, mode='comp'):
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
        self.reversi_model = predict_model.model


    def run(self, time_limit):
        start = time.time()
        while time.time() - start < time_limit:
            self.select(self.working_node)
        if selfplay_monitor:
            print('total simulate time:', self.root_node.visit_time)
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
        self.expand(working_node)
        # print('expand use:', time.time() - start)

    def expand(self, uct):
        """while the selection encounter a leaf node, expand its children and choose the best child of this node"""
        state = uct.my_board
        # the mother of the true real reward
        if state.game_over:
            winner = state.judge()
            reward = 1 if winner == state.side_color else -1
            self.backup(uct, reward, winner)
        #s = time.time()
        feature_input = self.calc_features(state)
        #print('calc feature use:', time.time() - s)
        # s = time.time()
        prediction = self.reversi_model.predict(feature_input.reshape(-1, BOARD_SIZE, BOARD_SIZE, 2))
        # print('predict use:', time.time() - s)
        p = prediction[0].reshape(BOARD_SIZE, BOARD_SIZE)
        v = prediction[1].reshape(1)
        # print(p, v)
        self.backup(uct, v[0], state.side_color)
        #s = time.time()
        moves = state.generate_moves(uct.state.side_color)  # 可以与神经网络输出并行
        #print('generate use:', time.time() - s)

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

                node.psa = p[move[0]][move[1]]
                children.append(node)
        # return self.choose_best_child(uct=uct)

    def backup(self, uct, reward, winner):
        """update the reward of the player_color"""
        while uct is not None:
            uct.visit_time += 1
            board = uct.my_board
            node_color = board.side_color
            if winner == node_color:
                uct.total_reward -= reward
            else:
                uct.total_reward += reward  #更新赢家节点之后的动作
                # uct.total_reward -= reward
            uct = uct.my_parent

    def stochastical_decide(self):
        """choose the action according to the distribution of prob of visiting times and this is for self play to encourage exploration"""
        pi = np.zeros(shape=(BOARD_SIZE, BOARD_SIZE))
        children = self.root_node.my_children
        sum_all = self.root_node.visit_time - 1

        if (children[0].parent_action is None) or (sum_all == 0):
            return (-1, -1), list(pi.flatten())
        a = [x for x in range(len(children))]
        p = []
        noise = []

        for c in children:
            coord = c.parent_action
            x, y = coord[0], coord[1]
            pi[x][y] = (c.visit_time / sum_all)
            noise.append(c.visit_time)
            p.append(pi[x][y])
            if selfplay_monitor:
                print(str(coord).center(20), end='')
        if selfplay_monitor:
            print()

        p = self.add_noise(noise, p)

        prob = [str(x * 100) for x in p]
        if selfplay_monitor:
            for pb in prob:
                print(pb.center(20), end='')
            print()
            print(cut_line)
        selected_index = np.random.choice(a=a, p=p)
        action = children[selected_index].parent_action
        return action, list(pi.flatten())

    def deterministical_decide(self):
        """choose the actions from the root according to the maximum visiting times"""
        pi = np.zeros(shape=(BOARD_SIZE, BOARD_SIZE))
        children = self.root_node.my_children
        best = children[0]
        sum_all = self.root_node.visit_time - 1
        if (best.parent_action is None) or (
                sum_all == 0):  # according to the alg, 'no action' always have a node, and this is a fake child
            # print('(-1, -1)')
            # print(list(pi.flatten()))
            return (-1, -1), list(pi.flatten())
        for c in children:
            coord = c.parent_action
            x = coord[0]
            y = coord[1]
            pi[x][y] = (c.visit_time / sum_all)
            if c.visit_time > best.visit_time:
                best = c

        return best.parent_action, list(pi.flatten())
        # print(best.parent_action)
        # print(list(pi.flatten()))

    @staticmethod
    def add_noise(noise, p):
        """add dirichlet noise"""
        noise = np.asarray(noise, dtype=np.float64)
        noise += epsilon
        noise = np.random.dirichlet(noise, 1)
        p = np.asarray(p)
        if DEBUG:
            print('before')
            p = np.asarray(p)
            prob = str(p * 100)  # convert to percentage
            for pb in prob:
                print(pb.center(20), end='')
            print(list(prob))
            print('after')
        p = list((1 - noise_eps) * p + noise_eps * noise.flatten())     # adding dirichlet noise eps = .25
        return p

    def APV_equation(self, uct):
        """return Q + U"""
        q_v_ = uct.total_reward
        nsa = uct.visit_time
        sigma_nsb = uct.my_parent.visit_time - 1
        psa = uct.psa
        if nsa == 0:
            return float('inf')
        equation = q_v_ / nsa + Cpuct * psa * math.sqrt(sigma_nsb) / nsa
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
        """
        simplified features
        :param s:
        :return: total board feature
        """
        one_piece = [0 for _ in range(2)]   # (x,y) x: if my disc or enermy, y: who i am
        one_feature = []
        bd = s.board
        for j in range(BOARD_SIZE):
            for i in range(BOARD_SIZE):
                e = bd[i][j]  # by row
                if e == s.side_color:
                    one_piece[0] = 1
                elif e == EMPTY:
                    one_piece[0] = 0
                else:
                    one_piece[0] = -1
                one_piece[1] = 1 if s.side_color == BLACK else 0  # tans to 0,1
                one_feature = np.hstack((one_feature, one_piece))
        return one_feature


if __name__ == '__main__':
    from cnn.model import ReversiModel
    reversi_model = ReversiModel()
    print(reversi_model.model.predict(np.random.random((1, BOARD_SIZE, BOARD_SIZE, 2))))
    uctalg = UCTAlg(reversi_model)
    uctalg.run(time_limit=4)
    print()
    # del reversi_model
    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)
