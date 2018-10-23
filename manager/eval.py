# -*- coding: utf-8 -*-
# !/usr/bin/env python3

from game.common import *
from manager.config import eval_rounds
import shutil
from cnn.config import model_challenger_path, model_defender_path
from manager.config import eval_log_path
import datetime
import time
from manager.selfplay import SelfPlay
IDLE_TIME = 600


def update_generation():
    model_lock.acquire()
    try:
        shutil.copy(src=model_challenger_path, dst=model_defender_path)
    except Exception as e:
        model_lock.release()
        print('update model failed...', e)
    else:
        print('model updated')
        ts = datetime.datetime.now()
        with open(eval_log_path, 'a') as f:
            f.write(str(ts) + ' model updated\n')
            f.close()
    finally:
        model_lock.release()


def eval(rounds):
    challenger_wins = 0
    p1, p2 = 'defender', 'challenger'
    p1, p2 = p2, p1
    eval_play = SelfPlay(player1=p1, player2=p2)
    for t in range(rounds):
        winner = eval_play.play_one_game()
        # print('winner is %s' % ('Black' if winner == BLACK else ('White' if winner == WHITE else 'TIE')))
        if (winner is BLACK) and (p1 is 'challenger'):
            challenger_wins += 1
            print('round {}: challenger win, total wins: {}'.format(t+1, challenger_wins))
        elif (winner is WHITE) and (p2 is 'challenger'):
            challenger_wins += 1
            print('round {}: challenger win, total wins: {}'.format(t+1, challenger_wins))
        else:
            print('round {}: challenger failed'.format(t))
        p1, p2 = p2, p1

    winning_rate = challenger_wins/rounds
    print('winning rate: {}'.format(winning_rate))
    if winning_rate > .55:
        update_generation()

def eval4ever(rounds=eval_rounds):
    while 1:
        eval(rounds=rounds)
        print('Have a {} seconds rest...'.format(IDLE_TIME))
        time.sleep(IDLE_TIME)


if __name__ == '__main__':
    eval(10)
