import math
import random

from app.game import Game
from app.field import DIRECTIONS


class Node(object):
    def __init__(self, state, parent=None, children=None):
        self.state = state  # game state at this node
        self.parent = parent
        self.children = children or []

        # Stats for the node
        self.w_i = 0
        self.n_i = 0

    def add_child(self, node):
        self.children.append(node)

    @property
    def t(self):
        """Parent node's total number of simulations"""
        return self.parent.n_i if self.parent else 1

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

    def __str__(self):
        return f'Node ({self.w_i}/{self.n_i})'


class GameState(object):
    def __init__(self, game):
        self.game = game

    def play(self, policy):
        game = self.game
        current_pos = game._pos
        while not game.finished:
            available_moves = []
            for move in DIRECTIONS.values():
                new_pos = current_pos + move
                if game.can_move(new_pos):
                    available_moves.append(move)
            selected_move = policy.choose_action(available_moves)

            game.step(selected_move)

            # Terminal state we've reached a finish point
            if game.success:
                return True

        return False

    def get_all_states(self):
        """Return all available states that can be reached from given state"""
        game = self.game
        current_pos = game._pos
        # breakpoint()
        states = []
        for move in DIRECTIONS.values():
            new_pos = current_pos + move
            game_copy = game.copy()
            if game_copy.can_move(new_pos):
                game_copy.step(new_pos)
                states.append(GameState(game=game_copy))
        return states


class RandomPolicy(object):
    def choose_action(self, available_actions):
        return random.choice(available_actions)


class MCTS(object):

    DEPTH = 5  # how many iterations
    TIME_LIMIT = 5  # 5 seconds to make a search

    def __init__(self, root):
        self.root = root
        self.policy = RandomPolicy()

    def selection(self, node):
        # Select promising node, search most prominent down the tree until leaf node found
        promising_node = node
        while promising_node.children:
            promising_node = max(node.children, key=lambda x: x.uct)
        print('Selected most promising node', promising_node)
        return promising_node

    def expansion(self, node):
        """Append all(some) possible states from the leaf node"""
        # get all possible states
        # one node per simulation is the most memory-efficient
        for state in node.state.get_all_states():
            new_node = Node(state=state, parent=node)
            node.add_child(new_node)
            print('Adding node', new_node)

    def simulation(self, node):
        """Pick a child node arbitrarily. Continue until terminal state or resource limit.
        No new nodes are created at this phrase"""
        # pick a random node
        selected_node = random.choice(node.children)
        # and do a random (defined by some policy) play out from it
        result = selected_node.state.play(self.policy)
        return result, selected_node

    def backpropagation(self, node, result):
        """Update nodes. Travel upwards to the root and increment visit score.
        Update win score if needed"""
        current_node = node
        while current_node is not None:
            if result is True:
                current_node.w_i += 1
            current_node.n_i += 1
            current_node = current_node.parent
        print(current_node)

    def search(self):
        # while some time limit
        node = self.selection(self.root)
        self.expansion(node)
        result, selected_node = self.simulation(node)
        self.backpropagation(selected_node, result)

        breakpoint()
        print('Best move is', node.state.game._pos)

    def move(self):
        """Change root after a move being made"""


if __name__ == '__main__':
    game = Game.create_game_debug()
    print(game)
    game_state = GameState(game=game)
    root = Node(state=game_state)
    t = MCTS(root=root)
    t.search()
