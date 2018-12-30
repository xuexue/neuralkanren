# Neural Guided Constraint Logic Programming for Program Synthesis

This repository contains code that goes with the paper:
Neural Guided Constraint Logic Programming for Program Synthesis 
https://papers.nips.cc/paper/7445-neural-guided-constraint-logic-programming-for-program-synthesis

This repository contains an implementation of miniKanren where the
constraint trees are represented transparently. We add scaffolding code
to show how to drive miniKanren using an external agent in Python.
We provide Recurrent Neural Network (RNN) and Graph Neural Network (GNN)
agents as examples. The implementations of the RNN and GNN are consistent
with the models described in the paper [0].


## Dependencies

* Chez Scheme: https://github.com/cisco/ChezScheme
* Python3+
* PyTorch 0.3+: https://pytorch.org/


## Transparent miniKanren

The following files contain an implementation of miniKanren where the
constraint trees are represented transparently, and a python interface for interacting
with a miniKanren process.


| File                                   | Description                                   |
| ---------------------------------------| ----------------------------------------------|
| [mk.scm](mk.scm)                       | transparent implemenation of minikanren       |
| [evalo.scm](evalo.scm)                 | definition of evalo, a relational interpreter |
| [query.scm](query.scm)                 | build queries annotated with ground truth     |
| [query-outputs.scm](query-outputs.scm) | compute ground truth outputs for queries      |
| [interact.scm](interact.scm)           | interaction process for python to talk to     |
| [interact.py](interact.py)             | python interface for scheme interaction       |
| [lisp.py](lisp.py)                     | helper for parsing lisp in python             |


## Neural Network Model

The following files contain neural network agents that can drive miniKanren's search.

| File                             | Description                         |
| ---------------------------------| ------------------------------------|
| [helper.py](helper.py)           | helper for working with constraints |
| [rnn_grammar.py](rnn_grammar.py) | parsing constraints for RNN model   |
| [rnn.py](rnn.py)                 | forward pass for RNN model          |
| [gnn_grammar.py](gnn_grammar.py) | parsing constraints for GNN model   |
| [gnn.py](gnn.py)                 | forward pass for GNN model          |


## Data Files

The following files contain the test problems used in the paper.

| File                                                         | Description                         |
| -------------------------------------------------------------| ------------------------------------|
| [data/test_problems.txt](data/test_problems.txt)             | held out tree manipulation problems |
| [data/repeat_test.txt](data/repeat_test.txt)                 | generalization: repeat(N)           |
| [data/drop_last_test.txt](data/drop_last_test.txt)           | generalization: dropLast(N)         |
| [data/bring_to_front_test.txt](data/bring_to_front_test.txt) | generalization: bringToFront(N)     |


## References

[0] [NeurIPS 2018](https://papers.nips.cc/paper/7445-neural-guided-constraint-logic-programming-for-program-synthesis) Neural Guided Constraint Logic Programming for Program Synthesis

[1] [ICLR 2018 Workshop](https://openreview.net/forum?id=HJIHtIJvz): with slightly less detail (4 pages)

[2] [Lisa Zhang's Master's Thesis](http://lisazhang.ca/msc_thesis.pdf): with slightly more detail (23 pages)

[3] More information about [miniKanren](http://minikanren.org)
