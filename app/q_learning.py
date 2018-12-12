"""
https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe
"""
import random

import click
import numpy as np

from app.game import Game, pos_to_state, state_to_pos
from app.field import Position, DIRECTIONS


def choose_an_action(space_size):
    return random.randint(0, space_size-1)


def perform_action(state, action, game: Game):
    current_pos = state_to_pos(state, game)  # make a method on Game?
    move = list(DIRECTIONS.values())[action]
    new_pos = current_pos + move
    # cannot move, stay in the same position, receiving nothing
    if game.field._is_border(new_pos.x, new_pos.y):
        return 0, state

    dist_passed = current_pos.steps_to(game._start) + 1
    moves_left = game._moves_left - dist_passed
    if new_pos.steps_to(game._end) > moves_left:
        reward = -1
    else:
        reward = game.field[new_pos.y][new_pos.x]  # game.get_reward(pos) method?
    new_state = pos_to_state(new_pos, game)  # method on Game?
    return reward, new_state


def q_learning(game: Game):
    states_space_size = (game.field.leny - 2)*(game.field.lenx - 2)
    actions_space_size = len(DIRECTIONS)
    QSA = np.zeros(shape=(states_space_size, actions_space_size))
    max_iterations = 80
    gamma = 1  # discount factor
    alpha = 0.9  # learning rate
    eps = 0.99  # exploitation rate
    s = 0  # initial state
    for i in range(max_iterations):
        # explore the world?
        a = choose_an_action(actions_space_size)
        # or not?
        if random.random() > eps:
            # todo (misha): decrease epsilon on which criteria
            a = np.argmax(QSA[s])

        r, s_ = perform_action(s, a, game)
        qsa = QSA[s][a]
        qsa_ = np.argmax(QSA[s_])
        QSA[s][a] = qsa + alpha*(r + gamma*qsa_ - qsa)

        # change state
        s = s_

        # I want to converge here instead of max iterations
    print(QSA)
    return QSA


def print_policy(policy, game):
    """
    ↖ ↑ ↗
    ← · →
    ↙ ↓ ↘
    """
    directions = (
        '↑',  # up
        '→',  # right
        '↓',  # down
        '←',  # left
    )

    def get_direction_index(state):
        return np.argmax(policy[state])

    rows = game.field.leny-2
    for row in range(rows):
        for column in range(game.field.lenx-2):
            state = row*rows + column
            pos = Position(x=column+1, y=row+1)
            index = get_direction_index(state)
            text = '{} '.format(directions[index])
            if pos == game._start:
                click.secho(text, fg='green', nl=False)
            elif pos == game._end:
                click.secho(text, fg='red', nl=False)
            else:
                click.echo(text, nl=False)
        print()


def main():
    game = Game.create_game_debug()
    policy = q_learning(game)
    print_policy(policy, game)


if __name__ == '__main__':
    main()
