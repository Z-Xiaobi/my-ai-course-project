# -*- coding='utf-8' -*-
# @Time    : 11/6/21 15:10
# @Author  : Xiaobi Zhang
# @FileName: q-learning_4x4_board.py

import math
import numpy as np

############################
## Initialization
############################

inputs = input()
inputs = inputs.split(' ')

actions = ['up', 'right', 'down', 'left']
start = [0, 1]  # start position
epsilon = 0.5
discount = 0.1
lr = 0.3
liv_rwd = -0.1
goal_rwd = 100
fbd_rwd = -100


def init_state(init_info):
    return int(init_info[0]), int(init_info[1]), int(init_info[2]), int(init_info[3])


def get_index(square_id):
    """convert id to board coordinate"""
    idx1 = math.ceil(square_id / 4) - 1
    if square_id % 4 == 0:
        idx2 = 3
    else:
        idx2 = square_id % 4 - 1
    return idx1, idx2


goal1, goal2, forbidden, wall = init_state(inputs[:4])
g1_row, g1_col = get_index(goal1)
g2_row, g2_col = get_index(goal2)
forb_row, forb_col = get_index(forbidden)
wall_row, wall_col = get_index(wall)

#
# print("goal1 ", g1_row, g1_col)
# print("goal2 ", g2_row, g2_col)
# print("forbidden ", forb_row, forb_col)
# print("wall ", wall_row, wall_col)

############################
## Q-learning on Game Board
############################

# 4 x 4 board and 4 actions
q_values = np.zeros((4, 4, 4))

def init_rewards():
    rwds = np.full((4, 4), liv_rwd)
    rwds[g1_row, g1_col] = 100
    rwds[g2_row, g2_col] = 100
    rwds[forb_row, forb_col] = -100
    return rwds
# 4 x 4 board
rewards = init_rewards()


import random
def get_next_action_idx(agent_row, agent_col, epsilon):
    """Epsilon Greedy Algorothm to generate next action"""
    if np.random.rand() < epsilon:
        # np.random.seed(1)
        random.seed(1)
        return np.random.randint(4)
    else:
        return np.argmax(q_values[agent_row, agent_col])


def get_next_position(agent_row, agent_col, action_idx):
    """wall: not move, count reward; outside of board: not move, no count reward"""
    next_agent_row, next_agent_col = agent_row, agent_col
    # up
    if action_idx == 0:
         next_agent_row += 1
    # right
    elif action_idx == 1:
        next_agent_col +=1
    # down
    elif action_idx == 2:
        next_agent_row -= 1
    # left
    elif action_idx == 3:
        next_agent_col -= 1

    # check fringe and wall
    if next_agent_row == wall_row and next_agent_col == wall_col:
        return agent_row, agent_col
    if next_agent_row not in range(0, 4) or next_agent_col not in range(0, 4):
        return agent_row, agent_col

    return next_agent_row, next_agent_col


def is_terminate(row, col):
    if row == g1_row and col == g1_col:
        return True
    if row == g2_row and col == g2_col:
        return True
    if row == forb_row and col == forb_col:
        return True
    return False

# print(is_terminate(g1_row, g1_col))
# print(is_terminate(g2_row, g2_col))
# print(is_terminate(forb_row, forb_col))

def q_learning(num_episode):
    # print("Q-Learning Start on ", str(num_episode), " Episodes")
    for episode in range(num_episode):
        curr_agent_row, curr_agent_col = start[0], start[1]
        while not is_terminate(curr_agent_row, curr_agent_col):
            action_idx = get_next_action_idx(curr_agent_row, curr_agent_col, epsilon)
            next_agent_row, next_agent_col = get_next_position(curr_agent_row, curr_agent_col, action_idx)
            curr_next_rwd = rewards[next_agent_row, next_agent_col]

            # update q value
            curr_q_value = q_values[curr_agent_row, curr_agent_col, action_idx]
            temp = curr_next_rwd + discount * np.max(q_values[next_agent_row, next_agent_col])
            next_q_value = (1 - lr) * curr_q_value + lr * temp
            q_values[curr_agent_row, curr_agent_col, action_idx] = next_q_value
            curr_agent_row, curr_agent_col = next_agent_row, next_agent_col
    # print("Q-Learning Finish.")

q_learning(num_episode=10000)


############################
## Process Output
############################

def square_opt_action(square_row, square_col):
    if square_row == wall_row and square_col == wall_col:
        return 'wall-square'
    if square_row == g1_row and square_col == g1_col:
        return 'goal'
    if square_row == g2_row and square_col == g2_col:
        return 'goal'
    if square_row == forb_row and square_col == forb_col:
        return 'forbid'
    square_q_value = q_values[square_row][square_col]
    action_idx = int(np.argmax(square_q_value))
    return actions[action_idx]


# print("q values:", q_values[::-1])
# print("rewards:", rewards[::-1])

if len(inputs) > 4:
    if inputs[4] == 'p':
        for square_idx in range(1, 17):
            row, col = get_index(square_idx)
            action = square_opt_action(row, col)
            print(square_idx, action)
    elif inputs[4] == 'q':
        if len(inputs) > 5:
            square_idx = int(inputs[5])
            square_idx_row, square_idx_col = get_index(square_idx)
            q_values = q_values[square_idx_row][square_idx_col]
            for action_idx in range(4):
                print(actions[action_idx], round(q_values[action_idx], 2))
        else:
            raise ValueError("Missing parameter for  \'q\'")
    else:
        raise ValueError("Invalid parameter: ", inputs[5])
