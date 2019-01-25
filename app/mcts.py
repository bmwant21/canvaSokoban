import math


class Node(object):
    def __init__(self):
        self.parent = None
        self.children = []

        self.state = None  # game state at this node
        # Stats for the node
        self.w_i = 0
        self.n_i = 0
        self.t = 0

    @property
    def uct(self):
        n_i = self.n_i  # number of simulations after the i-th move; total number of visits
        if n_i == 0:
            return math.inf

        c = math.sqrt(2.0)  # exploration parameter
        w_i = self.w_i  # number of wins after the i-th move; total simulation reward
        t = self.t  # total number of simulations for the parent node
        score = w_i / n_i + c * math.sqrt(math.log(t) / n_i)
        return score


class MCTS(object):

    DEPTH = 5  # how much iteration
    def __init__(self):
        pass

    def selection(self, node):
        # Select promising node
        promising_node = max(node.children, key=lambda x: x.uct)


    def expansion(self):
        """Append all possible states from the leaf node"""
        # get all possible states
        # pick a random node and simulate a random playout from it

    def simulation(self):
        """Pick a child node arbitrarily. Continue until terminal state or resource limit"""

    def backpropagation(self):
        """Update nodes. Travel upwards to the root and increment visit score.
        Update win score if needed"""


    def move(self):
        pass
