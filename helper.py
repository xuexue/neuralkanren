"""Helper function for working with constraint trees"""

from __future__ import print_function
from collections import defaultdict

import torch
import torch.nn as nn

import lisp


def get_leaf(cst, path, collapse_lvar=False):
    """Retrieve the leaf / constraint at a path in the cst, and
    convert it into a string.

    Arguments:

    cst  -- a constraint tree, consisting of conj / disj / pause
            internal nodes and constraint leafs. the tree 
            representing the "state" of the PBE problem
            and comes from Interaction.state
    path -- array of 0's and 1's, indicating whether to go left or right
            at a disjunction
    collapse_lvar -- whether to rename all logic variables to "_"
    """
    if cst[0] == 'pause':
        return get_leaf(cst[2], path, collapse_lvar)
    if cst[0] == 'conj':
        return get_leaf(cst[1], path, collapse_lvar)
    if cst[0] == 'disj':
        if path[0] == 0:
            return get_leaf(cst[1], path[1:], collapse_lvar)
        else:
            return get_leaf(cst[2], path[1:], collapse_lvar)
    assert len(path) == 0
    return lisp.unparse(cst, collapse_lvar)

def get_candidate_from_pause(pause):
    """Extract the candidate program at a given pause node. """
    state = pause[1][1]
    state = state[0][2]
    return state

def get_candidate(cst, path, collapse_lvar=False):
    """Retrieve the candidate at a path in the cst, and
    convert it into a string.

    Arguments:

    cst  -- a constraint tree, consisting of conj / disj / pause
            internal nodes and constraint leafs. the tree 
            representing the "state" of the PBE problem
            and comes from Interaction.state
    path -- array of 0's and 1's, indicating whether to go left or right
            at a disjunction
    collapse_lvar -- whether to rename all logic variables to "_"
    """
    if cst[0] == 'conj':
        return get_candidate(cst[1], path, collapse_lvar)
    if cst[0] == 'disj':
        if path[0] == 0:
            return get_candidate(cst[1], path[1:], collapse_lvar)
        else:
            return get_candidate(cst[2], path[1:], collapse_lvar)

    assert len(path) == 0 and cst[0] == 'pause'
    state = get_candidate_from_pause(cst)
    return lisp.unparse(state, collapse_lvar)

def get_all_constraints(cst):
    """Retrieve all the constraints in a cst. Each constraint is
    in its parsed form (not stringified)

    Arguments:

    cst  -- a constraint tree, consisting of conj / disj / pause
            internal nodes and constraint leafs. the tree 
            representing the "state" of the PBE problem
            and comes from Interaction.state
    """
    if cst[0] == 'pause':
        return get_all_constraints(cst[2])
    if cst[0] == 'disj':
        return get_all_constraints(cst[1]) + get_all_constraints(cst[2])
    if cst[0] == 'conj':
        return get_all_constraints(cst[1])
    return [cst]

def get_candidates(cst, current_path=None, candidate=None):
    """Retrieve all (path, candidate) pairs in a cst. Each path is 
    encoded as an array of 0 / 1 indicating to go left / right at a 
    disjunction. Each candidate is stringified.

    cst  -- a constraint tree, consisting of conj / disj / pause
            internal nodes and constraint leafs. the tree 
            representing the "state" of the PBE problem
            and comes from Interaction.state
    current_path (optional) -- accumulator variable for recursion
            indicating path from the root to the current cst node
    candidate (optional) -- accumulator variable for recursion
            indicating current candidate
    """
    if current_path is None:
        current_path = []

    if type(cst) == list:
        label = cst[0]
        if label == 'pause':
            candidate = get_candidate_from_pause(cst)
            candidate = lisp.unparse(candidate, True)
            return get_candidates(cst[2], current_path, candidate)
        if label == 'state':
            raise ValueError("Should not have state")
        if label == 'conj': # only go left
            return get_candidates(cst[1], current_path, candidate)
        if label == 'disj':
            return get_candidates(cst[1], current_path + [0], candidate) + \
                   get_candidates(cst[2], current_path + [1], candidate)
        return [(current_path, candidate)]

def flatten_branches(cst, path):
    """Return a list of children in the n-ary equilvaent form of the cst.
    The cst itself is a binary tree, so children of the top-level conj / disj
    may be nested.
    """
    type = cst[0]
    stack = [(path+[0], cst[1]), (path+[1], cst[2])]
    leafs = []
    while len(stack) > 0:
        p, item = stack.pop()
        if item[0] == type:
             # node type (conj/disj) same as root, so go deeper
            stack.append((p+[0], item[1]),)
            stack.append((p+[1], item[2]),)
        else:
            leafs.append((p, item))
    return leafs

class MLP(nn.Module):
    """
    Simple multi-layer perceptron
    """
    def __init__(self, dims):
        # This can be replaced with nn.Sequential
        super(MLP, self).__init__()
        self.layers = []
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            # No batch norm in this verison
            if i != len(dims) - 2:
                self.layers.append(nn.ReLU())

        for i, layer in enumerate(self.layers):
            self.add_module("l_%d" % i, layer)
    def forward(self, current):
        for layer in self.layers:
            current = layer(current)
        return current

class MergeMLP(MLP):
    """
    Simple multi-layer perceptron that concatenates inputs before
    passing it through the MLP. Used for training a merge function.
    """
    def forward(self, *inputs):
        merged_input = torch.cat(inputs, 1)
        return super(MergeMLP, self).forward(merged_input)


