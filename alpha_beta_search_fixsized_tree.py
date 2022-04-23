# -*- coding='utf-8' -*-
# @Time    : 10/11/21 20:23
# @Author  : Xiaobi Zhang
# @FileName: alpha_beta_search_fixsized_tree.py
"""
Minmax Game tree with Alpha Beta Pruning
Debugging based on online tool: https://raphsilva.github.io/utilities/minimax_simulator/
"""

import math
N_INF = -math.inf # negative infinity
P_INF = math.inf # positive infinity


#################################################
##           Minimax Tree Node
#################################################
# node: [parent_node, list_of_child_nodes, state]
# state : [value, alpha, beta, type]
# type:
# leaf node --- 0
# max node --- 1
# min node --- 2

# initial max state
MAX_INIT_STATE = [N_INF, N_INF, P_INF, 1]
# initial min state
MIN_INIT_STATE = [P_INF, N_INF,P_INF, 2]
# initial leaf state
def leaf_state(value):
    return [value, value, value, 0]


class MinimaxNode:
    def __init__(self, state):
        self.value = state[0]
        self.alpha = state[1]
        self.beta = state[2]
        self.type = state[3]
        self.parent = None
        self.children = []
        self.depth = 0
        self.index = 0 # index at current level (from left to right)

    def check_valid(self):
        if self.type != 0 and self.type != 1 and self.type != 2:
            raise ValueError("MinmaxNode have no type \'"+ str(self.type) + "\'")

    def add_child(self,child):
        if child and not isinstance(child, MinimaxNode):
            raise ValueError("child should be a MinimaxNode")
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
        child.index = len(self.children) - 1

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_children(self):
        return self.children

    def set_parent(self, parent):
        if parent and not isinstance(parent, MinimaxNode):
            raise ValueError("parent should be a MinimaxNode")
        self.parent = parent
        parent.add_child(self)

    def get_parent(self):
        return self.parent

    def set_state(self, state):
        self.value = state[0]
        self.alpha = state[1]
        self.beta = state[2]

    def set_value(self, value):
        self.value = value

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_beta(self, beta):
        self.beta = beta

    def get_value(self):
        value = self.value
        return value

    def get_alpha(self):
        alpha = self.alpha
        return alpha

    def get_beta(self):
        beta = self.beta
        return beta

    def get_depth(self):
        return self.depth





#################################################
##           Minimax Agent
#################################################
class MinimaxAgent:

    def __init__(self, leaves):
        # leaf index from 0 to 11
        self.leaves = leaves # size 12
        self.leaves_dic = {} # dictionary to store index for each leaf
        self.prune_value = []
        self.prune_idx = []
        self.depth = 4
        self.tree = self._new_tree()

    def _new_tree(self):
        # index for current leaf that need to be added into tree
        l_idx = 0
        # depth 0
        root = MinimaxNode(MAX_INIT_STATE)
        # depth 1
        root.add_children([
            MinimaxNode(MIN_INIT_STATE),
            MinimaxNode(MIN_INIT_STATE),
            MinimaxNode(MIN_INIT_STATE),
        ])
        # depth 2
        for min_node in root.get_children():
            min_node.add_children([
                MinimaxNode(MAX_INIT_STATE),
                MinimaxNode(MAX_INIT_STATE),
            ])
            # depth 3
            for max_node in min_node.get_children():
                leaf1 = MinimaxNode(leaf_state(self.leaves[l_idx]))
                leaf2 = MinimaxNode(leaf_state(self.leaves[l_idx+1]))
                max_node.add_children([leaf1, leaf2])
                self.leaves_dic[leaf1] = str(l_idx)
                self.leaves_dic[leaf2] = str(l_idx + 1)
                l_idx += 2
        return root

    def _print_tree_node_by_level(self, mmnode, level):
        """
        Recursively print nodes in each level, traverse in BFS order
        :type mmnode: MinimaxNode
        """
        if mmnode is None:
            return

        if mmnode.type is 0:
            print(' ' * 15 * level + '->', mmnode.value, "level",
                  mmnode.depth, " idx", mmnode.index, self.leaves_dic[mmnode])
        elif mmnode.type is 1:
            print(' ' * 15 * level + '->', "MAX [",
                  mmnode.value, ",", mmnode.alpha, ",", mmnode.beta, "]",
                  "level", mmnode.depth, "idx", mmnode.index)
        elif mmnode.type is 2:
            print(' ' * 15 * level + '->', "MIN [",
                  mmnode.value, ",", mmnode.alpha, ",", mmnode.beta, "]",
                  "level", mmnode.depth, " idx", mmnode.index)

        children = mmnode.get_children()
        for child in children:
            self._print_tree_node_by_level(child, level+1)

    def print_tree(self):
        """Visualize minimax tree"""
        if self.tree is None:
            return
        self._print_tree_node_by_level(self.tree, 0)

    def get_leaves_values(self, mmnode):
        """
        recursively find the leaves and append values
        :type mmnode: MinimaxNode
        """
        prune_values = []
        # leaf node
        if mmnode.type == 0:
            return [mmnode.value]
        else:
            for child in mmnode.get_children():
                prune_values += self.get_leaves_values(child)
        return prune_values

    def get_leaves_idxs(self, mmnode):
        """
        recursively find the leaves and append index
        :type mmnode: MinimaxNode
        :param mmnode: current minimax node
        :return: a list of leaves nodes indices from current node
        """

        prune_idxs = []

        # root node
        if mmnode.type == 0:
            return [self.leaves_dic[mmnode]]
        else:
            for child in mmnode.get_children():
                prune_idxs += self.get_leaves_idxs(child)

        return prune_idxs


    def get_value(self, mmnode, alpha, beta):
        mmnode.check_valid()
        if mmnode.type == 0:
            return mmnode.value
        elif mmnode.type == 1:
            return self.max_value(mmnode, alpha, beta)
        elif mmnode.type == 2:
            return self.min_value(mmnode, alpha, beta)
        return



    def max_value(self, mmnode, alpha, beta):
        """
        recursively search optimal value for max node
        :type mmnode: MinimaxNode
        """
        mmnode.check_valid()

        # obtain update from children nodes
        successors = mmnode.get_children()
        for i in range(len(successors)):

            # if current node obtained update from children
            # pass it to unvisited children
            if mmnode.alpha is not N_INF or mmnode.beta is not P_INF:
                alpha = mmnode.alpha
                beta = mmnode.beta

            # retrieve value from current visiting children
            succ = successors[i]
            self.get_value(succ, alpha, beta)
            value = max(mmnode.value, succ.value)

            # check whether to prune
            if beta is None: beta = mmnode.beta
            if value >= beta:
                print("prune at value = ", value, " beta = ", mmnode.beta)
                for j in range(i+1, len(successors)):
                    self.prune_value += self.get_leaves_values(successors[j])
                    self.prune_idx += self.get_leaves_idxs(successors[j])
                mmnode.value = value
                break

            mmnode.alpha = max(mmnode.alpha, value)
            mmnode.value = mmnode.alpha



    def min_value(self, mmnode, alpha, beta):
        """
        recursively search optimal value for min node
        :type mmnode: MinimaxNode
        """
        mmnode.check_valid()

        # obtain update from children nodes
        successors = mmnode.get_children()
        for i in range(len(successors)):

            # if current node obtained update from children
            # pass it to unvisited children
            if mmnode.alpha is not N_INF or mmnode.beta is not P_INF:
                alpha = mmnode.alpha
                beta = mmnode.beta

            # retrieve value from current visiting children
            succ = successors[i]
            self.get_value(succ, alpha, beta)
            value = min(mmnode.value, succ.value)

            # check whether to prune
            if alpha is None: alpha = mmnode.alpha
            if value <= alpha:
                print("prune at value = ", value, " alpha = ", mmnode.alpha)
                for j in range(i+1, len(successors)):
                    self.prune_value += self.get_leaves_values(successors[j])
                    self.prune_idx += self.get_leaves_idxs(successors[j])
                mmnode.value = value
                break

            mmnode.beta = min(mmnode.beta, value)
            mmnode.value = mmnode.beta

    def alpha_beta_prunning(self):
        self.get_value(self.tree, None, None)
        print(' '.join(self.prune_idx))


if __name__ == '__main__':
    inputs = input()
    inputs = inputs.split(' ')
    inputs = [int(i) for i in inputs]
    mmagent = MinimaxAgent(leaves=inputs[:12])
    mmagent.alpha_beta_prunning()