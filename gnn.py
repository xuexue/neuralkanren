"""For using a Graph Neural Network (GNN) to guide miniKanren"""

import random
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable

import lisp
import helper
from helper import MLP, MergeMLP
from interact import Interaction
from gnn_grammar import (GNN_TOKENS, GNN_RELATIONS, GNN_NODES, GNN_IDS,
                         parse_split_trees)

class GNNModel(nn.Module):
    """Graph Neural Network model for guiding miniKanren, using PyTorch.
    """
    def __init__(self,
                 embedding_size=64,
                 message_size=128,
                 msg_fn_layers=2,
                 merge_fn_extra_layers=2,
                 num_passes=1,
                 edge_embedding_size=32,
                 cuda=False):
        super(GNNModel, self).__init__()

        # set hyperparameters
        self.num_passes = num_passes # number of up/down passes
        self.embedding_size = embedding_size
        self.message_size = message_size
        self.edge_embedding_size = edge_embedding_size
        # set known vocab embedding -- for now also the consts
        self.embedding = nn.Embedding(len(GNN_TOKENS), embedding_size)
        # leaf scoring function for outputting
        self.score = MLP([embedding_size, 1])
        # message functions for each class x child x direction
        self.msg_fn_keys = [k for Class in GNN_NODES
                             for k in Class.msg_fn_keys()]
        # edge embedding for each edge type
        self.edge_embedding = nn.Embedding(len(self.msg_fn_keys),
                                           self.edge_embedding_size)
        # create mapping of msg fn keys -> index
        self.msg_fn_dict = {} 
        for i, k in enumerate(self.msg_fn_keys):
            self.msg_fn_dict[k] = Variable(torch.LongTensor([i]))
            if cuda:
                self.msg_fn_dict[k] = self.msg_fn_dict[k].cuda()
        # create the message functions:
        msg_fn_shape = [self.embedding_size + self.edge_embedding_size] + \
                       [self.message_size] * (msg_fn_layers - 1) +\
                       [self.message_size]
        self.msg_fn_shared = MLP(msg_fn_shape)
        # merge function for each class
        self.merge_fn = {}
        for Class in GNN_NODES:
            if Class.nmerge > 0:
                layers = [self.message_size * i
                          for i in range(Class.nmerge, 0, -1)] + \
                         [self.message_size] * merge_fn_extra_layers

                self.merge_fn[Class.name] = MergeMLP(layers)

        self.lvar_epsilon = torch.nn.Parameter(torch.FloatTensor([-10.0]))

        # gru for each class
        self.gru = {
            Class.name : nn.GRUCell(
                input_size=self.message_size,
                hidden_size=self.embedding_size,
                bias=True)
            for Class in GNN_NODES
        }

        self.lvar_epsilon = torch.nn.Parameter(torch.FloatTensor([-10.0]))
        # add modules in msgfn, mergefn, gru manually
        for k, module in self.gru.items():
            self.add_module("gru_%s" % k, module)
        for k, module in self.merge_fn.items():
            self.add_module("merge_%s" % k, module)

        self._cuda = cuda
        if self._cuda:
            self.cuda()

    # these functions below are compatible with a modified version
    # of pytorch fold for training

    def init_const(self, ids):
        """Initialize a constant embedding by looking up the ids."""
        if self._cuda:
            ids = ids.cuda()
        emb = self.embedding(ids)
        return emb

    def init_lvar(self, lvars):
        """Initialize a logic variable embedding by looking up the ids."""
        n = lvars.size()[0]
        id = Variable(torch.LongTensor([GNN_IDS["lvar"]]))
        if self._cuda:
            id = id.cuda()
        emb = self.embedding(id).repeat(n, 1)
        return emb

    def get_message(self, key, emb):
        """Get messages -- this might no longer work with pytorch fold
        after simplification...
        """
        k_index = self.msg_fn_dict[key]
        edge_emb = self.edge_embedding(k_index)
        edge_emb = edge_emb.repeat(emb.size(0), 1)
        merged_input = torch.cat([emb, edge_emb], 1) # for batching
        return self.msg_fn_shared(merged_input )

    def get_merge(self, key, *msgs):
        """Merge messages together

        key -- the node type
        msgs -- messages (tensors) from other nodes
        """
        return self.merge_fn[key](*msgs)

    def get_gru(self, key, msg, old):
        """Apply GRU to combine messages

        key -- the node type
        msg -- message (tensors) to be added to the node
        old -- current embedding of the node
        """
        return self.gru[key](msg, old)

    def get_merge_cat(self, nn, *msgs): # n = len(msgs)
        """Merge messages together by concatenation

        nn -- length of messages
        msgs -- messages (tensors) from other nodes
        """
        messages = torch.stack(msgs, 0)
        merged = torch.mean(messages, 0)
        return merged

    def get_logits(self, n, *embs): # n = len(embs)
        """Score the embeddings to produce logits

        n    -- number of embeddings
        embs -- embeddings (tensors) to score
        """
        embs_cat = torch.stack(embs, 0)
        leaf_logit = self.score(embs_cat)
        leaf_logit = leaf_logit.transpose(1,0).squeeze(-1)
        return leaf_logit

    def get_cat(self, n, *embs): # n = len(embs)
        """Concatenate embeddings

        n    -- number of embeddings
        embs -- embeddings (tensors) to score
        """
        return torch.stack(embs, 1).squeeze(-1)

    def get_combine_min(self, n, *logits): # n = len(embs)
        """Apply min pooling to logits

        n      -- number of embeddings
        logits -- logit scores to pool together
        """
        logits = torch.cat(logits, 1)
        return torch.min(logits, dim=1, keepdim=True)[0]

    def get_combine_max(self, n, *logits): # n = len(embs)
        """Apply max pooling to logits

        n      -- number of embeddings
        logits -- logit scores to pool together
        """
        logits = torch.cat(logits, 1)
        return torch.max(logits, dim=1, keepdim=True)[0]

    def get_combine_mean(self, n, *logits): # n = len(embs)
        """Apply average pooling to logits

        n      -- number of embeddings
        logits -- logit scores to pool together
        """
        logits = torch.cat(logits, 1)
        return torch.mean(logits, dim=1, keepdim=True)


def gnn_forward(asts, model, num_passes=1, test_acc=None):
    """
    Forward pass for Graph Neural Network, set up so that pytorch
    fold can be used for dynamic batching
    """
    # reset
    for ast, acc in asts:
        ast.reset(model)
        ast.annotate()
    # first upward pass
    for ast, acc in asts: # the different conj in the disj
        for leaf in acc['constraints']: # constraint in conj
            leaf.up_first(model)
    for passes in range(num_passes):
        # downward
        for ast, acc in asts: # the different conj in the disj
            for leaf in acc['constraints']: # constraint in conj
                leaf.down(model)

            # update logic variables
            for lvar in acc['lvar'].values():
                lvar.actually_update(model)
            for leaf in acc['constraints']: # constraint in conj
                leaf.up(model)
    # read out the logit
    out = []
    for ast, acc in asts: # conj
        leaf_logit = ast.logit(model)
        out.append(leaf_logit)
    out = model.get_cat(len(out), *out)
    return out


def test_forward():
    """Simple example using interaction, parsing, and sample call to GNN.forward()
    where we still take the ground truth path at each step."""

    problem = "(q-transform/hint (quote (lambda (cdr (cdr (var ()))))) (quote ((() y . 1) (#f y () . #t) (#f b () b . y) (x #f (#f . #f) . #t) (a #f y x s . a))))"
    step = 0
    model = GNNModel() # small model, randomly initialized

    print("Starting problem:", problem)
    with Interaction(lisp.parse(problem)) as env:
        signal = None
        while signal != "solved":
            # parse & score
            acc = {'constraints': []}

            parsed_subtree = parse_split_trees(env.state)
            out = gnn_forward(parsed_subtree, model)
            prob = F.softmax(out, 1)
            print(prob)

            # ignore the score and take the ground-truth step
            signal = env.follow_path(env.good_path)
            step += 1
            print('Step', step, 'Signal:', signal)
    print("Completed.")


if __name__ == '__main__':
    test_forward()
