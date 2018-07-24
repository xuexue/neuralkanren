from collections import namedtuple

import lisp
import helper
from interact import Interaction

FlatConj = namedtuple("FlatConj", ["candidate", "constraints",
                                   "path", "unparsed"])
FlatDisj = namedtuple("FlatDisj", ["candidate", "constraints",
                                   "path", "unparsed"])
Constraint = namedtuple("Constraint", ["unparsed",
                                       "type", # rename to relation!
                                       "seq",
                                       "len",
                                       "candidate",
                                       "path"])
RNN_TOKENS = ['(', ')', '.', "quote", '#t', '#f', 'list', 'cons', 'car',
              'cdr', 'lambda', 'app', 'closure', 'var',
              's', '1', '0', 'x', 'y', 'a', 'b', 'if', 'null?',
              'lvar', 'evalo', 'eval-listo', 'lookupo',
              '==', 'not-falseo', 'not-nullo']
RNN_RELATIONS = ['==', 'lookupo', 'evalo', 'eval-listo']
RNN_TOKEN_IDS = {token: i for (i, token) in enumerate(RNN_TOKENS)}

def get_token_ids(tokens):
    """Convert a list of tokens to a list of integer IDs of those tokens."""
    tok_seq = []
    for tok in tokens:
        tok = 'lvar' if tok.startswith("_") else tok
        tok_seq.append(RNN_TOKEN_IDS[tok])
    return tok_seq

def parse_tree(cst):
    """Recursive function that converts a constraint tree into a parsed form
    consisting of FlatConj, FlatDisj, and Constraint objects.
    Returns a list of choices (FlatConj | FlatDisj | Constraint objects)
    representing the possible paths to expand.

    cst  -- a constraint tree, consisting of conj / disj / pause
            internal nodes and constraint leafs. the tree
            representing the "state" of the PBE problem
            and comes from Interaction.state
    """
    choices = []

    def _parse_tree(cst, candidate, path):
        # get the label of the top-most element of the CST
        # this should be either 'pause', 'conj', 'disj', or a relation
        assert type(cst) == list
        label = cst[0]
        assert label != 'state'

        # root node in cst is a PAUSE
        if label == 'pause':
            # obtain the candidate partial program from the "pause" node
            candidate = lisp.unparse(helper.get_candidate_from_pause(cst),
                                    collapse_lvar=True)
            obj = _parse_tree(cst[2], candidate, path)
            choices.append(obj)
            return obj

        # root node in cst is a CONJ
        if label == 'conj':
            # obtain all children of conj nodes
            leafs = [_parse_tree(leaf, candidate, path)
                    for p, leaf in helper.flatten_branches(cst, path)]
            return FlatConj(constraints=leafs,
                            candidate=candidate,
                            path=path,
                            unparsed = "("+" && ".join(l.unparsed for l in leafs)+")")
        if label == 'disj':
            # obtain all children of disj nodes
            leafs = [_parse_tree(leaf, candidate, p)
                    for p, leaf in helper.flatten_branches(cst, path)]
            return FlatDisj(constraints=leafs,
                            candidate=candidate,
                            path=path,
                            unparsed = "("+" || ".join(l.unparsed for l in leafs)+")")

        # root node is a RELATION; leaf node
        leafstr = lisp.unparse(cst, True)
        leaf_tokens = lisp.tokenize(leafstr)
        leaf_type = leaf_tokens[1]
        tok_seq = get_token_ids(leaf_tokens)
        return Constraint(unparsed=leafstr,
                        candidate=candidate,
                        type=leaf_type,
                        seq=tok_seq,
                        len=len(tok_seq),
                        path=path[:])

    _parse_tree(cst, None, [])
    return choices

def test_parse_tree():
    """Simple example usage of Interaction + Parsing, where we take the ground
    truth path at each step."""

    problem = "(q-transform/hint (quote (lambda (cdr (cdr (var ()))))) (quote ((() y . 1) (#f y () . #t) (#f b () b . y) (x #f (#f . #f) . #t) (a #f y x s . a))))"
    step = 0

    print("Starting problem:", problem)
    with Interaction(lisp.parse(problem)) as env:
        signal = None
        while signal != "solved":
            # parse, then ignore
            choices = parse_tree(env.state)

            signal = env.follow_path(env.good_path)
            step += 1
            print('Step', step, 'Signal:', signal)
    print("Completed.")

if __name__ == '__main__':
    test_parse_tree()
