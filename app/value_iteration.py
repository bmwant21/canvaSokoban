import click
import numpy as np

from app.game import Game
from app.field import Position, DIRECTIONS


def normalize_reward(reward, game):
    """
    Return value from [0, 1] for a given reward.
    """

def get_available_states_from(state: int, action: int, game: Game) -> list:
    states = []
    xsize = game.field.lenx - 2  # inner field
    ysize = game.field.leny - 2  # inner field
    xpos = state % xsize + 1  # +border
    ypos = state // ysize + 1  # +border
    current_pos = Position(x=xpos, y=ypos)
    move = list(DIRECTIONS.values())[action]
    pos_from = current_pos + move
    if game.field._is_border(pos_from.x, pos_from.y):
        return states

    pairs = []
    unavailable = 0
    total = len(DIRECTIONS)
    r_total = 0
    # Available positions from current state when taking action
    for p in DIRECTIONS.values():
        new_pos = pos_from + p
        if game.field._is_border(new_pos.x, new_pos.y):
            unavailable += 1
            continue
        if new_pos.steps_to(game._end) > game._moves_left:
            # negative reward if cannot finish the game from this state
            r = -1
        else:
            r = normalize_reward(game.field[new_pos.y][new_pos.x], game)
        s_ = (new_pos.y-1)*ysize + (new_pos.x-1)
        pairs.append((s_, r))

    # probability is always the same for all available directions
    probability = 1/(total - unavailable)
    for s_, r in pairs:
        if r == -1:
            p = 0
        elif r == 0 or r_total == 0:
            p = 0.01
        else:
            p = r/r_total
        states.append((0.25, s_, r))
    return states


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
    max_iterations = 1000
    display_freq = 1 # max_iterations // 10
    eps = 1e-20
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
                for next_sr in get_available_states_from(s, a, game):
                    # print(next_sr)
                    # (probability, next_state, reward) of the states you can go from (s,a)
                    p, s_, r = next_sr
                    # reward from one-step-ahead state
                    next_states_rewards.append((p*(r + prev_v[s_])))
                # store the sum of rewards for each pair (s,a)
                q_sa.append(np.sum(next_states_rewards))
            # choose the max reward of (s,a) pairs and put it on the actual value function for STATE s
            v[s] = max(q_sa)

        # check convergence
        if np.abs(np.abs(np.sum(prev_v - v)) - last_dif) < eps:
            print('Value-iteration converged at iteration %d' % (i+1))
            break
        last_dif = np.abs(np.sum(prev_v - v))
    return v


def extract_policy(game, v, states_space_size, gamma=1.0):
    '''
    Extract the policy given a value-function
    '''

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
