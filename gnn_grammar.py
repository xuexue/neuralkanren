"""GNN Gramamr"""

import six
import torch
from torch.autograd import Variable

import lisp

import helper
from interact import Interaction


##############################################################################
# NODES
##############################################################################

# CONSTANTS
UP = "up"
DOWN = "down"
LEFT = "left"
MIDDLE = "middle"
RIGHT = "right"

GNN_TOKENS = ['()', "quote", '#t', '#f', 'list', 'cons', 'car',
              'cdr', 'lambda', 'app', 'closure', 'var',
              's', '1', '0', 'x', 'y', 'a', 'b', 'if', 'null?',
              'lvar',  # rest are internal
              'pair', 'conj', 'disj', '==',
              'evalo', 'eval-listo', 'lookupo', 'not-falseo', 'not-nullo']
GNN_IDS = {token: i for (i, token) in enumerate(GNN_TOKENS)}

class Node(object): # abstract
    """
    Super class nodes for parsed GNN constraint tree
    """
    name = None # used as a key for finding functions
    nmerge = 0  # number of child messages to merge during up pass
    def __init__(self, path=None):
        self.path = path
        self.children = {}
        self.embedding = []
        self.has_lvar = None
        self.up_done = False

    def set_embedding(self, label, model):
        """
        Initialize embedding using some label
        """
        label_id = Variable(torch.LongTensor([GNN_IDS[label]]))
        emb = model.init_const(label_id)
        self.embedding = [emb]

    def set_zero_embedding(self, model):
        """
        Initialize embedding to zero (??)
        """
        self.set_embedding(self.name, model)
        #init = model.init_zero()
        #self.embedding = [init]


    def annotate(self):
        """
        Annotate whether LVAR are required
        """
        self.has_lvar = False
        for child in self.children.values():
            child.annotate()
            if child.has_lvar:
                self.has_lvar= True

    def down(self, model):
        """
        Perform downward pass (parent -> children) of messages along
        the induced graph neural network.
        """
        for pos, child in sorted(self.children.items()):
            if child.has_lvar:
                msg = model.get_message(
                        (self.name, pos, DOWN),
                        self.embedding[-1])
                child.queue_update(msg, model)
                child.down(model)
    def reset(self, model):
        if len(self.embedding) == 0:
            self.reset_embedding(model)
        elif self.up_done:
            if len(self.embedding) > 2:
                self.embedding = self.embedding[:2]
        else:
            self.embedding = self.embedding[:1]

        for child in self.children.values():
            child.reset(model)

    def up_first(self, model):
        """
        Perform first upward pass (child -> parent) of messages along
        the induced graph neural network.
        """
        if self.up_done:
            # For performance gains, if we cache certain part of the constraint
            # graph across steps
            for child in self.children.values():
                assert(child.up_done)
                #child.up_first()
            return
        msgs = []
        for pos, child in sorted(self.children.items()):
            child.up_first(model)
            if child.embedding == []:
                import pdb; pdb.set_trace()

            assert(len(child.embedding) >= 1) # TOOD: remove assert?
            msg = model.get_message(
                    (self.name, pos, UP),
                    child.embedding[-1])
            msgs.append(msg)
        merged_message = model.get_merge(self.name, *msgs)

        self.queue_update(merged_message, model)
        self.up_done = True

    def up(self, model):
        """
        Perform upward pass (child -> parent) of messages along
        the induced graph neural network, just to the root of atomic
        structures
        """
        if not self.children:
            return

        msgs = []
        for pos, child in sorted(self.children.items()):
            if child.has_lvar:
                child.up(model)
            msg = model.get_message(
                    (self.name, pos, UP),
                    child.embedding[-1])
            msgs.append(msg)
        merged_message = model.get_merge(self.name, *msgs)
        self.queue_update(merged_message, model)

    def update(self, message, model):
        """
        Perform GRU update of the current node embedding
        """
        if len(self.embedding) > 0:
            new_emb = model.get_gru(self.name, message, self.embedding[-1])
            self.embedding.append(new_emb)
        else:
            #assert (not model.gru_forward)
            self.embedding.append(message)

    def queue_update(self, message, model):
        """
        Queue update. Some child classes will overwrite this because
        they won't actually update.
        """
        self.update(message, model)
    # for MLKanrenModel
    @classmethod
    def msg_fn_keys(self):
        """
        Keys for all message functions used to communicate with
        children. Since edge types between parent-child is dependent
        entirely on the parent type, parent class will contain all
        the message function nn.Modules.
        """
        return []

    def logit(self, model):
        """Compute logit. This is for leafs"""
        return model.get_logits(1, self.embedding[-1])

class AsymNode(Node):
    nmerge = 2
    def __init__(self, left, right, path=None):
        super(AsymNode, self).__init__(path)
        self.children = {
            LEFT: left,
            RIGHT: right
        }

    @classmethod
    def msg_fn_keys(self):
        keys = []
        for child in [LEFT, RIGHT]:
            for dir in [UP, DOWN]:
                keys.append((self.name, child, dir),)
        return keys

    def reset_embedding(self, model):
        self.set_zero_embedding(model)

class SymNode(Node): # abstract
    """
    Abstract class for nodes that are symmetric -- same edge type
    (and hence message function) for both children
    """
    nmerge = 2
    def __init__(self, left, right, path=None):
        super(SymNode, self).__init__(path)
        # dep
        self.left = left
        self.right = right
        # actually used
        self.children = {
            LEFT: left,
            RIGHT: right
        }

    def reset_embedding(self, model):
        self.set_zero_embedding(model)

    def up_first(self, model):
        """
        Perform first upward pass (child -> parent) of messages along
        the induced graph neural network.
        """
        if self.up_done:
            for child in self.children.values():
                assert(child.up_done)
                child.up_first(model)
            return
        msgs = []
        for k, child in sorted(self.children.items()):
            child.up_first(model)
            assert(len(child.embedding) >= 1)
            msg = model.get_message(
                    (self.name, UP),
                    child.embedding[-1])
            msgs.append(msg)
        merged_message = model.get_merge(self.name, *msgs)
        self.queue_update(merged_message, model)
        self.up_done = True

    def up(self, model):
        """
        Normal upward pass
        """
        if not self.children:
            return
        msgs = []
        for k, child in sorted(self.children.items()):
            if child.has_lvar:
                child.up(model)
            msg = model.get_message(
                    (self.name, UP),
                    child.embedding[-1])
            msgs.append(msg)
        merged_message = model.get_merge(self.name, *msgs)
        self.queue_update(merged_message, model)


    def down(self, model):
        message = model.get_message(
                (self.name, DOWN),
                self.embedding[-1])
        if self.left.has_lvar:
            self.left.queue_update(message, model)
            self.left.down(model)
        if self.right.has_lvar:
            self.right.queue_update(message, model)
            self.right.down(model)

    def __str__(self):
        return "(%s %s %s)" % (self.name, self.left, self.right)

    def setLeft(self, l):
        self.left = l
        self.children[LEFT] = l

    def setRight(self, r):
        self.right = r
        self.children[RIGHT] = r

    @classmethod
    def msg_fn_keys(self):
        return [(self.name, UP), (self.name, DOWN)]

class SymConNode(SymNode): # abstrct constraint node
    """
    Abstract class for nodes that are symmetric and keep unparsed
    """
    def __init__(self, left, right, path, unparsed, state):
        super(SymConNode, self).__init__(left, right, path)
        self.unparsed = unparsed
        self.state = state

class TripletNode(Node): # abstract constraint node
    """
    Abstract class for asymmetric node with three children
    """
    nmerge = 3
    def __init__(self, left, middle, right, path, unparsed, state):
        super(TripletNode, self).__init__(path)
        self.unparsed = unparsed
        self.state = state
        self.children = {
            LEFT: left,
            MIDDLE: middle,
            RIGHT: right
        }

    def __str__(self):
        lst = tuple([self.name] + [
                str(self.children[k]) for k in [LEFT, MIDDLE, RIGHT]])
        return "(%s %s %s %s)" % lst

    @classmethod
    def msg_fn_keys(self):
        keys = []
        for child in [LEFT, MIDDLE, RIGHT]:
            for dir in [UP, DOWN]:
                keys.append((self.name, child, dir),)
        return keys

    def reset_embedding(self, model):
        self.set_zero_embedding(model)


##############################################################################
# Leaf nodes -- constants & logic variables
##############################################################################

class LVar(Node):
    name = "lvar"
    def __init__(self, label):
        super(LVar, self).__init__()
        self.label = label
        self.has_lvar = True
        self.up_done = True

        self.embedding = []
        # incoming messages
        self.incoming_messages = []

    def reset_embedding(self, model):
        #self.set_embedding("lvar", model)
        init_embedding = model.init_lvar(Variable(torch.Tensor(1))) # for fold
        self.embedding = [init_embedding]

    def annotate(self):
        self.has_lvar = True

    def queue_update(self, message, model):
        # queues updates, don't actually update
        self.incoming_messages.append(message)

    def actually_update(self, model):
        # aggregate using MEAN!
        merged = model.get_merge_cat(
                len(self.incoming_messages), # key
                *self.incoming_messages)
        self.update(merged, model)
        self.incoming_messages = []

    def __str__(self):
        return self.label

class Constant(Node):
    name = "const"

    def __init__(self, label):
        super(Constant, self).__init__()
        self.label = label
        self.has_lvar = False
        self.up_done = True
        self.label = label
        # emulate upward pass
        if label not in GNN_IDS:
            raise ValueError("Unknown token: %s" % label)

    def reset_embedding(self, model):
        self.set_embedding(self.label, model)

    def annotate(self):
        self.has_lvar = False

    def queue_update(self, message, model):
        pass # DO NOT UPDATE

    def __str__(self):
        return str(self.label)

##############################################################################
# Nodes specific to the LISP language grammar
##############################################################################

class Pair(AsymNode):
    name = "pair"
    def __str__(self):
        return "(%s . %s)" % (self.children[LEFT], self.children[RIGHT])

class Conj(SymNode):
    nmerge = 0 # hack
    name = "conj"

class Disj(SymNode):
    nmerge = 0 # hack
    name = "disj"

class Unify(SymConNode):
    name = "=="

class Lookupo(TripletNode):
    name = "lookupo"

class EvalExpo(TripletNode):
    name = "evalo"

class EvalListo(TripletNode):
    name = "eval-listo"

#######################################
# N-ary version of conjunctions & disjunctions
#######################################
class NAryNode(SymNode):
    nmerge = -1
    def __init__(self, children, path=None):
        super(SymNode, self).__init__(path)
        self.children = {i: child for i, child in enumerate(children)}

    def reset_embedding(self, model):
        self.set_zero_embedding(model)

    def __str__(self):
        return "(%s %s)" % (self.name, ' '.join(
            str(self.children[i]) for i in range(len(self.children))))

    def setLeft(self, l):
        raise ValueError()

    def setRight(self, r):
        raise ValueError()

    def down(self, model):
        message = model.get_message(
                (self.name, DOWN),
                self.embedding[-1])
        for child in self.children.values():
            if child.has_lvar:
                child.queue_update(message, model)
                child.down(model)

class NAryConj(NAryNode):
    name = "conj"
    def logit(self, model):
        logits = [child.logit(model) for child in self.children.values()]
        return model.get_combine_min(len(logits), *logits)

class NAryDisj(NAryNode):
    name = "disj"
    def logit(self, model):
        logits = [child.logit(model) for child in self.children.values()]
        return model.get_combine_max(len(logits), *logits)

###############################################################################
# STUFF
###############################################################################
GNN_NODES = [Constant, LVar, Conj, Disj, NAryConj, NAryDisj,
             Unify, Pair, Lookupo, EvalExpo, EvalListo]
GNN_NODE_CLASS = {Cls.name: Cls for Cls in GNN_NODES}
GNN_RELATIONS = [Unify, Lookupo, EvalExpo, EvalListo]

# SETTINGS:
MEMOIZE = True

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def has_lvar(unparsed):
    return "_." in unparsed

def get_cst_node_type(cst):
    """Return the constraint tree node type of cst."""
    if type(cst) == list:
        if len(cst) == 0:
            return 'const'
        if type(cst[0]) in (str, six.text_type):
            if cst[0] in GNN_NODE_CLASS:
                return cst[0]
            if cst[0] in ('pause', 'state'):
                return cst[0]
        #print cst[0]
        #import pdb; pdb.set_trace()
        if len(cst) > 0:
            return 'pair'
        return 'const'

    if type(cst) in (str, six.text_type) and cst.startswith('_.'):
        return 'lvar'
    return 'const'

def parse_tree(cst, acc=None, prev=None, candidate=None, path=None):
    """Recursive function that converts a constraint tree into a parsed form
    consisting of subclasses of gnn_nodes.Node

    cst  -- a constraint tree, consisting of conj / disj / pause
            internal nodes and constraint leafs. the tree
            representing the "state" of the PBE problem
            and comes from Interaction.state
    acc  -- an accumulator that is passed by value
    prev -- a previous instance of parsed_tree() accumulator from the previous
            step of the same problem
    candidate -- candidate partial program related to this part of the subtree
    path -- array of 0's and 1's, indicating whether to go left or right
            at a disjunction
    """
    # set up initial values that are objects
    path = path or []
    acc = acc or { "constraints": [], "memo": {}, "lvar": {}}
    if "constraints" not in acc:
        acc["constraints"] = [] # repository of constraints
    if "memo" not in acc:
        acc["memo"] = {} # memoized parsed subtree
    if "lvar" not in acc:
        acc["lvar"] = {} # all logic variables
    prev = prev or {}

    node_type = get_cst_node_type(cst)

    # get the pause & state out of the way
    if node_type == 'pause':
        candidate = lisp.unparse(helper.get_candidate_from_pause(cst), True)
        return parse_tree(cst[2], acc, prev, candidate, path)
    if node_type == 'state':
        raise ValueError("Should never be here!")

    # logical constructs
    if node_type == 'conj':
        children = []
        for subpath, subcst in helper.flatten_branches(cst, path):
            # for conj, subpath is ignored!
            child, acc = parse_tree(subcst, acc, prev, candidate, path)
            children.append(child)
        return NAryConj(children, path), acc
        #left, acc = parse_tree(cst[1], acc, prev, candidate, path)
        #right, acc = parse_tree(cst[2], acc, prev, candidate, path)
        #return Conj(left, right, path), acc
    if node_type == 'disj':
        children = []
        for subpath, subcst in helper.flatten_branches(cst, path):
            child, acc = parse_tree(subcst, acc, prev, candidate, subpath)
            children.append(child)
        return NAryDisj(children, path), acc
        #left, acc = parse_tree(cst[1], acc, prev, candidate, path+[0])
        #right, acc = parse_tree(cst[2], acc, prev, candidate, path+[1])
        #return Disj(left, right, path), acc

    # if using MEMOIZE, check if this (sub-)tree has already been parsed, and
    # its parsed version stored
    unparsed = lisp.unparse(cst)
    if MEMOIZE and unparsed in acc["memo"]:
        return acc["memo"][unparsed], acc

    # logic variables: add to repository of logic variables if not exist
    if node_type == 'lvar':
        if cst not in acc['lvar']:
            new_node = prev.get('lvar', {}).get(cst)
            new_node = new_node or LVar(cst)
            acc['lvar'][cst] = new_node
        return acc["lvar"] [cst], acc

    # relations
    if node_type in ('evalo', 'lookupo', '==', 'eval-listo',
                     'not-falseo', 'not-nullo'):
        # a relations can have multiple children in its tree
        child_asts = []
        for child_cst in cst[1:]:
            child_ast, acc = parse_tree(child_cst, acc, prev, candidate, path)
            child_asts.append(child_ast)
        NodeClass = GNN_NODE_CLASS[node_type]
        new_node_args = child_asts + [path, unparsed, candidate]
        new_node = NodeClass(*new_node_args)
        acc["constraints"].append(new_node)
        return new_node, acc

    # pairs
    if node_type == 'pair':
        if len(cst) == 3 and cst[1] == '.':
            # first
            left, acc = parse_tree(cst[0], acc, prev, candidate, path)
            # second
            right, acc = parse_tree(cst[2], acc, prev, candidate, path)
        else:
            # first
            left, acc = parse_tree(cst[0], acc, prev, candidate, path)
            # rest
            right, acc = parse_tree(cst[1:], acc, prev, candidate, path)
        new_node = Pair(left, right)
        if MEMOIZE and not has_lvar(unparsed):
            acc["memo"][unparsed] = new_node
        return new_node, acc

    # language constructs (inside the relations leafs)
    if node_type == 'const':
        new_node = Constant(unparsed)
        if MEMOIZE and not has_lvar(unparsed):
            acc["memo"][unparsed] = new_node
        return new_node, acc

    raise ValueError("Should not be here?")


def parse_split_trees(cst):
    """Split the CST into multiple trees, one per candidate.
    This function returns a list of tree objects, subclasses of gnn_nodes.Node

    cst  -- a constraint tree, consisting of conj / disj / pause
            internal nodes and constraint leafs. the tree
            representing the "state" of the PBE problem
            and comes from Interaction.state
    """
    shared_memo = {}
    parsed = []
    for path, subtree in helper.flatten_branches(cst, []):
        ast, acc = parse_tree(subtree, {"memo": shared_memo}, path=path)
        parsed.append((ast, acc))
    return parsed

def test_parse_tree():
    """Simple example usage of Interaction + Parsing, where we take the ground
    truth path at each step."""

    problem = "(q-transform/hint (quote (lambda (cdr (cdr (var ()))))) (quote ((() y . 1) (#f y () . #t) (#f b () b . y) (x #f (#f . #f) . #t) (a #f y x s . a))))"
    step = 0

    print("Starting problem:", problem)
    with Interaction(lisp.parse(problem)) as env:
        signal = None
        while signal != "solved":
            parsed_subtree = parse_split_trees(env.state)
            print(len(parsed_subtree))
            signal = env.follow_path(env.good_path)
            step += 1
            print('Step', step, 'Signal:', signal)
    print("Completed.")

if __name__ == '__main__':
    test_parse_tree()
