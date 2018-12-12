"""
https://pdfs.semanticscholar.org/99b2/fd28dcab3657c5f1271a05223f4740e4b65c.pdf
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


def get_u_r(state: int, rsa):
    return np.max(rsa[state])


def print_policy(rsa, game):
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
        return np.argmax(rsa[state])

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


def r_learning(game: Game):
    states_space_size = 16
    actions_space_size = 4
    rho = 0
    alpha = 0.9  # learning rate for rho value
    rsa = np.zeros(shape=(states_space_size, actions_space_size))
    beta = 0.9  # learning rate for rsa
    max_iterations = 100
    s = 0  # initial state; is starting state better?
    for i in range(max_iterations):
        a = choose_an_action(actions_space_size)  # random action selection
        r_imm, s_ = perform_action(s, a, game)
        urs = get_u_r(s, rsa)
        urs_ = get_u_r(s_, rsa)
        if random.random() < beta:
            rsa[s][a] = r_imm - rho + urs_

        # action agrees with a policy?
        if random.random() < alpha and rsa[s][a] == urs:
            rho = r_imm + urs_ - urs

        # change state
        s = s_
    print(rsa)
    return rsa


def main():
    game = Game.create_game_debug()
    policy = r_learning(game)
    print_policy(policy, game)


if __name__ == '__main__':
    main()
