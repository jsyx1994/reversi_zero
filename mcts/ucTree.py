# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""This defines the structure of a UCT"""

import copy
import random

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
        self.psa = 0

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
        # if self.parent is not None:
        #     return self.parent
        # else:
        #     pass    # This the root node

    @property
    def my_board(self):
        return self.state
        # if self.state is not None:
        #     return self.state
        # else:
        #     pass    # print("Not assigned a state")

    @property
    def my_children(self):
        return self.children


