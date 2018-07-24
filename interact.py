from __future__ import print_function
import random
from subprocess import Popen, PIPE

import lisp

class Interaction(object):
    """Interaction object that communicates with scheme to solve a
    miniKaren Programming by Example query. Should use with 

        with Interaction(query) as env:
            ...

    See example_interaction_gt() for example usage.
    """

    INTERACT_SCRIPT = 'interact.scm'

    def __init__(self, query):
        """Create new interaction object.

        query -- A parsed lisp expression that evaluates to a PBE
                 problem. Normally a call to q-transform/hint,
                 which creates a PBE problem based using a
                 ground-truth program.
        """
        self.query = query
        self.proc = None # scheme process
        self.state = None # constraint tree
        self.good_path = None # ground truth path

    def __enter__(self):
        """Start a scheme process, and send the query to the process. """
        self.proc = Popen(['scheme', '--script', self.INTERACT_SCRIPT],
                          stdin=PIPE, stdout=PIPE)
        self._send(self.query)
        self._read_state()
        return self

    def __exit__(self, *args):
        """Stop the scheme process. """
        self.proc.stdin.close()

    def _read(self):
        """Helper frunction to read from the scheme process. """
        txt = self.proc.stdout.readline().rstrip().decode('utf-8')
        if not txt:
            return None
        return lisp.parse(txt)

    def _send(self, datum):
        """Helper frunction to write to the scheme process.

        Arguments:

        datum -- the content of the message to be relayed, as a parsed
                 lisp data structure.
        """
        self.proc.stdin.write((lisp.unparse(datum) + '\n').encode('utf-8'))
        self.proc.stdin.flush()

    def _good_path(self):
        """Populate self.good_path by ineracting with the scheme process. """
        if self.state is None:
            self.good_path = None
        else:
            self._send('good-path')
            self.good_path = self._read()

    def _read_state(self):
        """Populate self.state and self.good_path by ineracting with
        the scheme process.
        """
        self.state = self._read()
        self._good_path()

    def follow_path(self, path):
        """Communicate to the scheme process to expand the candidate at a
        given path.

        Arguments:

        path -- array of 0's and 1's, indicating whether to go left or right
                at a disjunction
        """
        self._send(path)
        feedback = self._read()
        self._read_state()
        return feedback

    def steps_remaining(self):
        """Returns the number of steps remaining to solve the PBE problem if
        ground-truth steps are taken. Communicates with scheme proces.
        """
        self._send('steps-remaining')
        return self._read()

    def jump_to_steps_remaining(self, n):
        """Fast-forward the solving process so that exactly n steps remains
        to solve the PBE problem if ground-truth steps are taken.
        Communicates with scheme proces.
        """
        self._send(['jump-to-steps-remaining', n])
        self._read_state()


def example_interaction_gt():
    """Simple example usage of Interaction, where we take the ground truth
    path at each step."""

    problem = "(q-transform/hint (quote (lambda (cdr (cdr (var ()))))) (quote ((() y . 1) (#f y () . #t) (#f b () b . y) (x #f (#f . #f) . #t) (a #f y x s . a))))"
    step = 0

    print("Starting problem:", problem)
    with Interaction(lisp.parse(problem)) as env:
        signal = None
        while signal != "solved":
            # env.state is ignored
            signal = env.follow_path(env.good_path)
            step += 1
            print('Step', step, 'Signal:', signal)
    print("Completed.")


def example_interaction_with_policy(policy="random"):
    """Example usage of Interaction that uses a random policy
    or user interaction to decide which candidate to expand.
    """
    assert policy in ("random", "user")

    def random_policy(state):
        """Choose a random candidate to expand"""
        from helper import get_candidates
        lfs = get_candidates(state)
        path = random.choice(lfs)[0]
        return path

    def user_policy(state):
        """Use user input to choose a candidate to expand"""
        from helper import get_candidates
        from helper import get_candidate
        lfs = get_candidates(state)
        while True:
            print("Choices: ")
            for idx, (lf_path, lf_prog) in enumerate(lfs):
                print(idx, lf_prog)
            index = input("Choices: ")
            try:
                path = lfs[int(index)][0]
            except:
                print("ERROR. TRY AGAIN.")
            return path

    problem = "(q-transform/hint (quote (lambda (cdr (cdr (var ()))))) (quote ((() y . 1) (#f y () . #t) (#f b () b . y) (x #f (#f . #f) . #t) (a #f y x s . a))))"
    print("Starting problem:", problem)
    print("Policy:", policy)
    step = 0
    max_steps = 50
    policy_fn = {"random": random_policy,
                 "user": user_policy}[policy]


    with Interaction(lisp.parse(problem)) as env:
        signal = None
        while signal != "solved" and step < max_steps:
            path = policy_fn(env.state)
            is_good_path = (path == env.good_path)
            signal = env.follow_path(path)
            step += 1
            print('Step:', step, 'GT Path?:', is_good_path, 'Signal:', signal)
    print("Completed.")


if __name__ == "__main__":
    example_interaction_gt()
    example_interaction_with_policy("user")
