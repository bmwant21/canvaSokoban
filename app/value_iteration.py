import click
import numpy as np

from app.game import Game, pos_to_state, state_to_pos
from app.field import Position, DIRECTIONS


def normalize_reward(reward, game) -> float:
    """
    Return value from [0, 1] for a given reward.
    """
    diff = game.max_value - game.min_value
    return reward / diff


def get_probability_for(pos: Position, game: Game) -> float:
    probabilities = {
        0: 1/4,
        1: 1/3,
        2: 1/2,
    }
    score = 0
    if pos.x == 1 or pos.x == game.field.lenx - 2:
        score += 1
    if pos.y == 1 or pos.y == game.field.leny - 2:
        score += 1
    return probabilities[score]


def get_available_states_from(state: int, action: int, game: Game) -> list:
    states = []
    current_pos = state_to_pos(state, game)
    move = list(DIRECTIONS.values())[action]
    pos_from = current_pos + move
    if game.field._is_border(pos_from.x, pos_from.y):
        return states

    unavailable = 0
    probability = get_probability_for(pos_from, game)
    # Available positions from current state when taking action
    for p in DIRECTIONS.values():
        new_pos = pos_from + p
        if game.field._is_border(new_pos.x, new_pos.y):
            unavailable += 1
            continue
        distance_passed = pos_from.steps_to(game._start) + 1
        # print('Distance passed to', new_pos, distance_passed)
        moves_left = game._moves_left - distance_passed
        # print('Moves left', moves_left)
        # print('Steps to end', new_pos.steps_to(game._end))
        if new_pos.steps_to(game._end) > moves_left:
            # negative reward if cannot finish the game from this state
            r = -1
        else:
            r = normalize_reward(game.field[new_pos.y][new_pos.x], game)
        s_ = pos_to_state(new_pos, game)
        states.append((probability, s_, r))

    return states


def get_action_effects(state: int, action: int, game: Game):
    move = list(DIRECTIONS.values())[action]
    current_pos = state_to_pos(state, game)
    new_pos = current_pos + move
    if game.field._is_border(new_pos.x, new_pos.y):
        # return [(0, state, -1)] ?
        return []

    p = get_probability_for(current_pos, game)
    s_ = pos_to_state(new_pos, game)
    # yes, but give -1 reward
    # we may be going back, so make sure steps are counter properly
    dist_passed = current_pos.steps_to(game._start) + 1
    moves_left = game._moves_left - dist_passed
    if new_pos.steps_to(game._end) > moves_left:
        r = -1
    else:
        # r = normalize_reward(game.field[new_pos.y][new_pos.x], game)
        r = game.field[new_pos.y][new_pos.x]

    if new_pos == game._end:
        r = 100
    return [(p, s_, r)]


def _display_value_func(v):
    """
    if i % display_freq == 0:
    print('25 first elements of actual value function. An array of number of possible states (%d) elements:\n%s'
          % (states_space_size, v[:25]))
    :return:
    """
    # print(v)


def value_iteration(states_space_size, game, gamma=1.0):
    # value function, represents a VALUE for each state
    v = np.zeros(states_space_size)
    max_iterations = 100
    display_freq = 1  # max_iterations // 10
    eps = 1e-10
    last_dif = float('inf')

    print('Starting training loop...')
    for i in range(max_iterations):
        _display_value_func(v)
        prev_v = np.copy(v)  # last value function
        for s in range(states_space_size):  # for each STATE s
            q_sa = []
            for a in range(len(DIRECTIONS)):  # for each ACTION a
                next_states_rewards = []
                # print('Going', list(DIRECTIONS.keys())[a], 'from state', s)
                # iterate the states you can go from determined state-action pair (s,a)
                for next_sr in get_action_effects(s, a, game):
                    # print(next_sr)
                    # (probability, next_state, reward) of the states you can go from (s,a)
                    p, s_, r = next_sr
                    # reward if we choose this action
                    next_states_rewards.append((p*(r + prev_v[s_])))
                # store the sum of rewards for each pair (s,a)
                q_sa.append(np.sum(next_states_rewards))
            # choose the max reward of (s,a) pairs and put it on the actual value function for STATE s
            print(q_sa)
            v[s] = max(q_sa)
        # break
        # check convergence
        # if np.abs(np.abs(np.sum(prev_v - v)) - last_dif) < eps:
        #     print('Value-iteration converged at iteration %d' % (i+1))
        #     break
        last_dif = np.abs(np.sum(prev_v - v))
    return v


def extract_policy(game, v, states_space_size, gamma=1.0):
    """
    Extract the policy given a value-function
    """

    policy = np.zeros(states_space_size) #Policy : array of 0s with as many elements as possible states
    for s in range(states_space_size):
        q_sa = np.zeros(len(DIRECTIONS)) # q_sa: array of 0s with as many elements as possible actions
        for a in range(len(DIRECTIONS)):
            for next_sr in get_available_states_from(s, a, game): #Iterate the states you can go from determined state-action pair
                #next_sr is a tuple of (probability, next_state, reward)
                p, s_, r = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)

    return policy


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
            index = row*rows + column
            pos = Position(x=column+1, y=row+1)
            text = '{} '.format(directions[policy[index]])
            if pos == game._start:
                click.secho(text, fg='green', nl=False)
            elif pos == game._end:
                click.secho(text, fg='red', nl=False)
            else:
                click.echo(text, nl=False)
        print()


def main():
    # create random game
    game_instance = Game.create_game_debug()
    play_size_x = game_instance.field.lenx - 2
    play_size_y = game_instance.field.leny - 2
    states_space_size = play_size_x * play_size_y
    v = value_iteration(
        states_space_size=states_space_size,
        game=game_instance,
    )
    print(game_instance)
    rows = game_instance.field.leny-2
    for row in range(rows):
        for column in range(game_instance.field.lenx-2):
            index = row*rows + column
            print(v[index], end=' ')
        print()
    policy = extract_policy(game_instance, v, states_space_size)
    # import pdb; pdb.set_trace()
    print_policy(policy.astype('int'), game=game_instance)


if __name__ == '__main__':
    main()
