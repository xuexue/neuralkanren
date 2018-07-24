"""For using a Recurrent Neural Network (RNN) to guide miniKanren"""
from __future__ import print_function

import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from collections import namedtuple

import lisp
import helper
from helper import MLP
from interact import Interaction
from rnn_grammar import parse_tree, RNN_TOKENS, RNN_RELATIONS, FlatConj, FlatDisj

############################## RNN MODEL ####################################

class RNNModel(nn.Module):
    """Recurrent Neural Network model for guiding miniKanren, using PyTorch.
    """
    def __init__(self,
                 embedding_size=128,
                 num_layers=2,
                 bidirectional=True,
                 cuda=False):
        """Create a new RNNModel object based on the specifications

        embedding_size -- size of each RNN embedding
        num_layers -- number of RNN layers
        bidirectional -- whether the RNN is bidirectional
        cuda -- whether to use GPU
        """
        super(RNNModel, self).__init__()
        self._num_layers = num_layers
        self._embedding_size = embedding_size
        self._hidden_size = embedding_size
        if bidirectional:
            self._hidden_size //= 2

        # create an embedding for the tokens
        self.embedding = nn.Embedding(len(RNN_TOKENS), embedding_size)

        # create a separate LSTM model for each relation type
        self.lstms = {}
        for relation_type in RNN_RELATIONS:
            lstm = nn.LSTM(embedding_size,
                           self._hidden_size,
                           self._num_layers,
                           bidirectional=bidirectional)
            self.lstms[relation_type] = lstm
            self.add_module("lstm_%s" % relation_type, lstm)

        # create a scoring MLP
        self.score = MLP([embedding_size, 1])

        # check CUDA
        self._cuda = cuda
        if self._cuda:
            self.cuda()

    def torch_leafseq(self, leafseq):
        """Convert an array of tokens represented by the indices in
        RNN_TOKEN_IDS into a PyTorch tensor."""
        seq = Variable(torch.Tensor(leafseq).long())
        seq = seq.unsqueeze(0)
        if self._cuda:
            seq = seq.cuda()
        return seq

    def forward(self, choices):
        """Score a minibatch of parsed constraint tree states.

        choices -- list of choices, type FlatConj | FlatDisj | Constraint
        """

        # We will do some batching manually, to batch together constraints that
        # use the same relation, and thus the same RNN. First, we will look at
        # each constraint in each parsed tree in `all_parsed`.

        keys = {}      # key_tuple ->
                       #   index of constraint in result
        inputs = {}    # relation ->
                       #   list of (constraint token length, embedding)
        memo = {}      # unparsed constraint str ->
                       #   index of constraint in result

        # to_add is a queue of nodes to be added, each element of this queue
        # will be a pair (key, node), where key is the tuple with
        #   (batch id, leaf_id)
        to_add = []
        for n_lf, lf in enumerate(choices):
            to_add.append(((n_lf,), lf),)

        while to_add:
            key_tuple, node = to_add.pop()
            # if a node is conj / disj, their children need to be added
            if type(node) in (FlatConj, FlatDisj):
                for n_lf, lf in enumerate(node.constraints):
                    to_add.append((key_tuple + (n_lf,), lf),)
            # if a node is an actual relation / constraint...
            else:
                lc = node
                if lc.unparsed in memo:
                    # Sometimes, the same constraint will appear multiple
                    # times. We should only do forward pass once.
                    keys[key_tuple] = memo[lc.unparsed]
                else:
                    if lc.type not in inputs:
                        inputs[lc.type] = []
                    # Get the embedding of each token in the constraint
                    emb = self.embedding(self.torch_leafseq(lc.seq))
                    inputs[lc.type].append((lc.len, emb))
                    keys[key_tuple] = len(inputs[lc.type]) - 1
                    memo[lc.unparsed] = keys[key_tuple]

        # Do forward pass for each relation type

        outputs = {}   # relation -> PyTorch tensor of scores
                       #   in the same order as `inputs`
        for rnn_type in inputs:
            # need to have seq length in descending order
            sorting = [(l,old_i) for old_i, (l, seq) in enumerate(inputs[rnn_type])]
            sorting.sort(reverse=True)
            seqlen = [l for l, old_id in sorting]
            max_len, max_i = sorting[0]
            # pad the sequence
            paddedseq = []
            for l, old_i in sorting:
                seq = inputs[rnn_type][old_i][1].transpose(0,1)
                if l == max_len:
                    padseq = seq
                else:
                    padseq = torch.cat([seq, Variable(torch.zeros(max_len-l,1,self._embedding_size))])
                    if self._cuda:
                        padseq = padseq.cuda()
                paddedseq.append(padseq)
            # construct the padded seq
            seqem = torch.cat(paddedseq,1)
            packedem = pack_padded_sequence(seqem, seqlen)
            # LSTM fwd pass
            packout, _ = self.lstms[rnn_type](packedem)
            out, _ = pad_packed_sequence(packout)
            # extract & get LSTM output
            prescore = out[[l-1 for l in seqlen],
                           range(len(seqlen)),
                           :]
            # re-order prior to scoring
            reverse_index = list(range(len(sorting)))
            for n_i, (_, old_i) in enumerate(sorting):
                reverse_index[old_i] = n_i
            prescore = prescore[reverse_index, :]
            outputs[rnn_type] = self.score(prescore)

        # Combine along conjunctions / disjunctions

        def get_combined(oo, key_tuple, node):
            """Helper function to combine scores along conj/disj

            oo         -- outputs dictionary
            key_tuples -- tuple of from (batch id, constraint id)
            node       -- curent node
            """
            if type(node) not in (FlatConj, FlatDisj):
                # lookup score
                j = keys[key_tuple]
                x = oo[node.type][j]
                return x
            result = []
            for n_lf, lf in enumerate(node.constraints):
                r = get_combined(oo, key_tuple + (n_lf,), lf)
                result.append(r)

            if type(node) == FlatDisj:
                return torch.max(torch.stack(result), 0)[0]

            if type(node) == FlatConj:
                return torch.mean(torch.stack(result), 0)
                #return torch.min(torch.stack(result), 0)[0]

        # Construct output tensors based on outputs


        logits = []
        for n_lf, lf in enumerate(choices):
            logits.append(get_combined(outputs, (n_lf,), lf))
        return torch.stack(logits, 1)

def test_forward():
    """Simple example using interaction, parsing, and sample call to RNN.forward()
    where we still take the ground truth path at each step."""

    problem = "(q-transform/hint (quote (lambda (cdr (cdr (var ()))))) (quote ((() y . 1) (#f y () . #t) (#f b () b . y) (x #f (#f . #f) . #t) (a #f y x s . a))))"
    step = 0
    model = RNNModel(embedding_size=32, num_layers=1) # small model

    print("Starting problem:", problem)
    with Interaction(lisp.parse(problem)) as env:
        signal = None
        while signal != "solved":
            # parse & score
            choices = parse_tree(env.state)
            out = model.forward(choices)
            prob = F.softmax(out, 1)
            print(prob)
            # ignore the score and take the ground-truth step
            signal = env.follow_path(env.good_path)
            step += 1
            print('Step', step, 'Signal:', signal)
    print("Completed.")


if __name__ == '__main__':
    test_forward()

