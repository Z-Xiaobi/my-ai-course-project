# -*- coding='utf-8' -*-
# @Time    : 9/20/21 18:00
# @Author  : Xiaobi Zhang
# @FileName: modified_pancake.py
# Burned pancake problem
# Revised pancake problem that also consider the two sides of pancake, one side is burnt, another is not
# Goal is to place all the pancakes non-burnt side up in the order of small size to big size
# from top to the plate


# Input:
# Pancakes
# 1. id number of pancake, the smallest pancake has id 1
# 2. whether if the burnt side is down (“w”) or not (“b”)
# 3. number of pancakes is 4
# Algorithms
# BFS: "b" or A*:"a"
# Example :
# 2b1w4b3w-a

from copy import deepcopy
import operator
import sys

###########################################################
# Problem State Space
###########################################################

## state [[id,dr],[id,dr],[id,dr],[id,dr]]
## id: id of this piece of pancake
## dr: the burned side direction, down "w"; up "b"


## Specify the goal state
burned_pancake_goal_state = [['1','w'], ['2','w'], ['3','w'], ['4','w']]

## Check whether the state is goal state
def burned_pancake_test_goal_state(state):
    return operator.eq(state, burned_pancake_goal_state)

## Specify the initial state
def set_burned_pancake_intial_state(pancakes):
    """
    Receive a string and convert it to a state
    :param pancakes: a string of pancake information
                    eg: 2b1w4b3w
    :return: a burned pancake problem state
    """

    pancakes = list(pancakes)
    # split length
    slen = 2
    burned_pancake_intial_state = [pancakes[i:i+slen] for i in range(0,len(pancakes), slen)]
    return burned_pancake_intial_state



def is_valid_state(state):
    """
    sequence should be #+'w'or'b'
    and # should in the range of pancake number
    :param state: a burned pancake problem state
    :return: True for valid or False for not valid
    """
    num_cake = len(state)  # number of pancakes
    for s in state:

        if str(s[0]) < '0' or str(s[0]) > str(num_cake):
            return False

        if s[1] != 'w' and s[1] != 'b':
            return False

    return True

def to_tie_breaking_string(state):
    """replace “w” with 1 and “b” with 0 to obtain an eight-digit number."""
    tb_str = ""
    if is_valid_state(state):
        for s in state:
            if s[1] == 'w':
                tb_str += str(s[0]) + '1'
            elif s[1] == 'b':
                tb_str += str(s[0]) + '0'

    return tb_str


def flip_ith_pancake(state, i):
    """
    flip all the upper pancakes from the ith position (include ith pancake)
    :param state: a burned pancake problem state
    :param i: flip position, integer
    :return: a flipped pancake state
    """
    new_state = []

    for idx in range(i, -1, -1):
        flip_side = 'w' if state[idx][1] == 'b' else 'b'
        s = [state[idx][0], flip_side]
        new_state.append(s)

    for idx in range(i+1, len(state)):
        new_state.append(state[idx])

    return new_state


def burned_pancakes_possible_actions(state):
    """
    Actions : choose which position to flip
    :param state: a burned pancake problem state
    :return: list of possible following actions of current input state
            actions (index of lowest position pancake will be flipped)
    """
    actions = [] # possible actions
    num_cake = len(state)  # number of pancakes
    for i in range(num_cake):
        actions.append(i)
    return actions


## Successor State
## action is one of the actions from 'flip_burned_pankcakes_possible_actions'
def burned_pancakes_successor_state(action, state):
    new_state = flip_ith_pancake(state, action)
    return new_state



## Problem tuple
def burned_pancakes_problem(pancakes):
    """

    :param pancakes: string in format: #C#C#C#C
    :return: a tuple of burned pancakes problem in the format:
    (
        initial_state,       # initial state for search
        poss_act_func,       # function operating on a state to return list of possible actions
        successor_func,      # function operating on an action and a state to return successor state
        goal_test_func       # Boolean function operating on a state to test if it is a goal
    )

    """
    burned_pancake_initial_state = set_burned_pancake_intial_state(pancakes)
    return (burned_pancake_initial_state,
            burned_pancakes_possible_actions,
            burned_pancakes_successor_state,
            burned_pancake_test_goal_state
            )


###########################################################
# Tree Data Structure
###########################################################
## each node takes the form:
## [parent_node, list_of_child_nodes, [action_path, state, cost, heuristic_value]]

def new_node():
    return [0, [], [[], [], 0, 0]]

def node_add_child(node, child):
    node[1].append(child)
    child[0] = node
    return node

def node_add_children(node, children):
    for child in children:
        node_add_child(node, child)
    return node

def node_get_children(node):
    return node[1]

def node_set_parent(node, parent):
    node[0] = parent
    parent[1].append(node)
    return node

def node_get_parent(node):
    return node[0]


def node_set_path(node, path):
    node[2][0] = path
    return node

def node_get_path(node):
    return node[2][0]

def node_set_state(node, state):
    node[2][1] = state
    return node

def node_get_state(node):
    return node[2][1]

def node_set_cost(node, cost):
    node[2][2] = cost
    return node

def node_get_cost(node):
    return node[2][2]

def node_set_heuristic(node, heuristic):
    node[2][3] = heuristic
    return node

def node_get_heuristic(node):
    return node[2][3]

def node_get_path_length(node):
    return len(node_get_path(node))


def showlist(list):
    for item in list:
        print(item)

def node_satisfies_goal(node, goal):
    if goal in node_get_state(node):
        return True
    return False

'''
def node_state_occurs_in_ancestor(node):
    state = node_get_state(node)
    parent = node_get_parent(node)
    return node_state_occurs_in_upward_path(parent, state)


### Subordinate function for node_state_occurs_in_ancestor(node)
def node_state_occurs_in_upward_path(node, state):
    while True:
        if node == 0:
            return False
        if node_get_state(node) == state:
            return True
        node = node_get_parent(node)
'''

def node_get_depth(node):
    depth = 0
    while True:
        if node == 0:
            return depth
        node = node_get_parent(node)
        depth = depth + 1


###########################################################
# Search on Tree
###########################################################

## PROBLEM is a problem specification tuple of the form:
##     (
##       initial_state,       # initial state for search
##       poss_act_func,       # function operating on a state to return list of possible actions
##       successor_func,      # function operating on an action and a state to return successor state
##       goal_test_func       # Boolean function operating on a state to test if it is a goal
##     )



def get_initial_node_queue(initial_state):
    """Queue that only contains root node with initial state"""
    return [node_set_state(new_node(), initial_state)]


def possible_action_successor_pairs(state, poss_actions, successor_fun):
    """
    Get the pair of possible actions and successor function
    :param state:
    :param poss_actions: function to tell possible actions (burned_pancakes_possible_actions)
    :param successor_fun: successor function
    :return: a list for each pair of possible actions  and successor function
    """
    poss_acts = poss_actions(state)
    # return a list of tuples for one action and the next state it produces
    return [(action, successor_fun(action, state)) for action in poss_acts]


'''
def heuristic_A_star(state):
    """Heuristic function of A*: number of pancakes out of order, only consider id"""
    i = 1
    count = 0

    for s in state:
        # if str(s[0]) != i or s[1] != 'w': # consider id and burnt side order
        if str(s[0]) != str(i):
            count += 1
        i += 1
    return count
'''
def heuristic_A_star(state):
    """Heuristic function of A*: the ID of the largest pancake that is still out of place"""
    i = 1 # iterator of id
    ids = [] # the id of pancake that out of order

    for s in state:
        if str(s[0]) != str(i):
            ids.append(s[0])
        i += 1


    heuristic = lambda id_list: int(max(id_list)) if len(id_list) > 0 else 0
    return heuristic(ids)

def cost_A_star(node):
    parent = node[0]
    # print(parent[2])
    cost_of_parent = parent[2][2]
    action = node[2][0]
    return cost_of_parent + int(action[-1]) + 1


def node_expand(node, poss_actions, successor_fun):
    """
    Add children for current node
    :param node: current node that need to expand
    :param poss_actions: possible actions
    :param successor_fun: successor function
    :return: current node
    """
    ## each node takes the form:
    ## [parent_node, list_of_child_nodes, [action_path, state, cost, heuristic_value]]
    state = node_get_state(node)
    path = node_get_path(node) # path of actions
    # pairs for action and the next state it produces (action, nextstate)
    act_suc_pairs = possible_action_successor_pairs(state, poss_actions,
                                                successor_fun)
    for pair in act_suc_pairs:
        action = pair[0]
        result_state = pair[1]
        # new_node(): [0, [], [[],[],0,0]]
        child = new_node()
        # create a empty node, and set the current node as parent
        # node_set_parent( node, parent )
        node_set_parent(child, node)
        # add one action to path
        # node[2][0] = path
        node_set_path(child, path + [action])
        # set the next state produced by action for the child node
        node_set_state(child, result_state)

    # return the current node rather than its children
    return node

def add_nodes_A_star(new_nodes, node_queue, cost_func, heuristic_func):
    """
    expend following nodes in A*
    :param new_nodes: a list of nodes that needs to consider which to expand (children)
    :param node_queue: a list of current node fringe
    :param cost_func: cost function
    :param heuristic_func: heuristic function
    :return: new node fringe
    """

    for n_node in new_nodes:
        new_h = heuristic_func(node_get_state(n_node))
        new_g = cost_func(n_node)
        node_set_heuristic(n_node, new_h)
        node_set_cost(n_node, new_g)

        new_f_Astar = new_h + new_g  # Astar ranking based on f = heuristic + cost

        inserted = False
        # print("len(node_queue): ", len(node_queue))
        for i in range(len(node_queue)):
            # f of nodes in node queue
            fi_Astar = node_get_cost(node_queue[i]) + node_get_heuristic(node_queue[i])

            # if (new_f_Astar < fi_Astar):
            #     node_queue.insert(i, n_node)
            #     inserted = True
            #     break

            if (new_f_Astar < fi_Astar):
                node_queue.insert(i, n_node)
                inserted = True
                break
            elif (new_f_Astar == fi_Astar):  # tie-breaking
                new_str = to_tie_breaking_string(n_node[2][1])
                i_str = to_tie_breaking_string(node_queue[i][2][1])
                if (new_str < i_str):
                    node_queue.insert(i, n_node)
                    inserted = True
                    break


        if not inserted:
            node_queue.append(n_node)
    return node_queue


def add_to_node_queue(strategy, node_queue, new_nodes):
    """add new nodes to node queue with specified strategy"""
    # BFS
    if (strategy == 'b'):
        return node_queue + new_nodes
    # A*
    if (strategy == 'a'):
        heuristic_func = heuristic_A_star
        # cost_func = node_get_depth  # cost from root
        cost_func = cost_A_star
        return add_nodes_A_star(new_nodes,
                                node_queue,
                                cost_func,
                                heuristic_func)
    print("ERROR: unknown strategy: [" + strategy + "]\n You should use \'a\' or \'b\'")
    return None


## [parent_node, list_of_child_nodes, [action_path, state, cost, heuristic_value]]
def print_action_list(act_list, act_nodes, strategy):
    """output steps of problem solution based in strategy"""
    for i in range(len(act_list)):
        flip_pos = act_list[i] # flip position
        step = ""
        cake_idx = 0
        node = act_nodes[i]
        state = node[2][1]
        for s in state:
            step += ''.join(s)
            if cake_idx == flip_pos:
                step += '|'
            cake_idx += 1

        if strategy == 'a':
            cost = node[2][2]
            heuristic = node[2][3]
            step += ' g:'+str(cost) + ', h:'+str(heuristic)

        print(step)

    goal = ""
    for s in act_nodes[-1][2][1]:
        goal += ''.join(s)
    if strategy == 'a':
        cost = act_nodes[-1][2][2]
        heuristic = act_nodes[-1][2][3]
        goal += ' g:' + str(cost) + ', h:' + str(heuristic)
    print(goal) # goal


## PROBLEM is a problem specification tuple of the form:
##     (
##       initial_state,       # initial state for search
##       poss_act_func,       # function operating on a state to return list of possible actions
##       successor_func,      # function operating on an action and a state to return successor state
##       goal_test_func       # Boolean function operating on a state to test if it is a goal
##     )

def search(problem, strategy, max_nodes=20000):

    initial_state = problem[0]
    poss_act_func = problem[1]
    successor_func = problem[2]
    goal_test_func = problem[3]

    node_queue = get_initial_node_queue(initial_state)
    initial_node = node_queue[0]
    if len(node_queue) == 0:
        return False

    for i in range(max_nodes):
        # get initial state from node_queue
        first_node = node_queue.pop(0)  # take 1st node from queue
        ## [parent_node, list_of_child_nodes, [action_path, state, cost, heuristic_value]]
        if goal_test_func(node_get_state(first_node)):
            action_path = node_get_path(first_node)

            # get all passed states
            # ignore the root
            action_nodes = []
            node = first_node


            while node_get_parent(node) != 0:
                action_nodes.append(node)
                node = node_get_parent(node)
            action_nodes.append(initial_node)
            action_nodes.reverse()
            print_action_list(action_path, action_nodes, strategy)

            return action_path

        # children: node[1] --- list_of_child_nodes
        children = node_get_children(node_expand(first_node,
                                                 poss_act_func,
                                                 successor_func
                                                 )
                                     )
        node_queue = add_to_node_queue(strategy, node_queue, children)


    return False



if __name__ == "__main__":

    # python hw1.py 1b2b3w4b-b
    inputs = sys.argv[1]
    # inputs = input()
    inputs = inputs.split('-')
    pancakes = inputs[0]
    strategy = inputs[1]

    # print(pancakes, strategy)

    problem = burned_pancakes_problem(pancakes=pancakes)
    search(problem,strategy)
    








