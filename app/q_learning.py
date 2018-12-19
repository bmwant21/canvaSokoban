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


def convert_reward(reward) -> float:
    """
    r'(s) = -d/(r(s)+c)
    """
    d = 1
    c = 2
    return -d / (reward + c)


def perform_action(state, action, game: Game):
    current_pos = state_to_pos(state, game)  # make a method on Game?
    move = list(DIRECTIONS.values())[action]
    new_pos = current_pos + move
    done = False
    # cannot move, stay in the same position, receiving nothing
    if game.field._is_border(new_pos.x, new_pos.y):
        return 0, state, done

    dist_passed = current_pos.steps_to(game._start) + 1
    moves_left = game._moves_left - dist_passed
    reward = game.field[new_pos.y][new_pos.x]  # game.get_reward(pos) method?
    if new_pos.steps_to(game._end) > moves_left:
        r = -1
        done = True
    else:
        r = convert_reward(reward)

    if new_pos == game._end:
        r = reward
        done = True

    new_state = pos_to_state(new_pos, game)  # method on Game?
    return r, new_state, done


def q_learning(game: Game):
    total_episodes = 500
    max_steps = game._moves_left
    states_space_size = (game.field.leny - 2)*(game.field.lenx - 2)
    actions_space_size = len(DIRECTIONS)
    QSA = np.zeros(shape=(states_space_size, actions_space_size))
    # strive for long term high reward
    gamma = 1  # discount factor
    # In fully deterministic environment a learning rate of 1 is optimal
    alpha = 1  # learning rate
    # Exploration rate for e-greedy action selection
    min_eps = 0.01
    eps = 1.0
    max_eps = 1.0
    decay_rate = 0.001
    for episode in range(total_episodes):
        # reset state
        s = pos_to_state(game._start, game)
        for step in range(max_steps):
            # explore the world, choose an action randomly
            a = choose_an_action(actions_space_size)
            # or follow existing policy as learned more
            if random.random() > eps:
                a = np.argmax(QSA[s])

            r, s_, done = perform_action(s, a, game)
            qsa = QSA[s][a]
            qsa_ = np.max(QSA[s_])
            # QSA[s][a] = qsa + alpha*(r + gamma*qsa_ - qsa)
            QSA[s][a] = r + qsa_

            # change state
            s = s_
            if done is True:
                break
        eps = min_eps + (max_eps - min_eps)*np.exp(-decay_rate*episode)
    return QSA


def get_policy(q):
    def get_action_index(state):
        return np.argmax(q[state])

    policy = [get_action_index(state) for state, _ in enumerate(q)]
    return policy


def get_path(game):
    QSA = q_learning(game)
    policy = get_policy(QSA)
    path = []
    current_pos = game._start
    while True:
        path.append(current_pos)
        if current_pos == game._end:
            break
        state = pos_to_state(current_pos, game)
        action = policy[state]
        current_pos += list(DIRECTIONS.values())[action]
    return path


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
    rows = game.field.leny-2
    for row in range(rows):
        for column in range(game.field.lenx-2):
            state = row*rows + column
            action = policy[state]
            pos = Position(x=column+1, y=row+1)
            text = '{} '.format(directions[action])
            if pos == game._start:
                click.secho(text, fg='green', nl=False)
            elif pos == game._end:
                click.secho(text, fg='red', nl=False)
            else:
                click.echo(text, nl=False)
        print()


def main():
    game = Game.create_game_debug()
    print(game)
    Q = q_learning(game)
    policy = get_policy(Q)
    print_policy(policy, game)


if __name__ == '__main__':
    main()
