"""
Solve a game using dynamic programming
"""
import queue

import click
from app.game import Game
from app.field import Position, DIRECTIONS


DIRECTION_ARROWS = (
    '↑',  # up
    '→',  # right
    '↓',  # down
    '←',  # left
)


def dynamic_programming(game):
    q = queue.Queue()
    # todo (misha): no need to use it, we can count steps based on position
    path_map = dict()  # hold steps to find way back
    scores_map = dict()
    pos_start = game._start
    score_start = game.field[pos_start.y][pos_start.x]
    q.put((game._start, 0))
    scores_map[pos_start] = score_start
    while not q.empty():
        pos, steps = q.get()
        score = scores_map[pos]
        if pos not in path_map:
            path_map[pos] = steps

        # we cannot move further from this position
        if steps + 1 > game._moves_left:
            continue

        for i, move in enumerate(DIRECTIONS.values()):
            new_pos = pos + move
            # do not consider moves hitting the border
            if game.field._is_border(new_pos.x, new_pos.y):
                continue

            new_score = score + game.field[new_pos.y][new_pos.x]
            if new_pos not in scores_map or new_score > scores_map[new_pos]:
                scores_map[new_pos] = new_score
            q.put((new_pos, steps+1))

    return scores_map, path_map


def print_path(path_map, policy, game):
    def get_direction(pos):
        maxv = 0
        index = 0
        steps = path_map[pos]
        for i, move in enumerate(DIRECTIONS.values()):
            new_pos = pos + move
            if game.field._is_border(new_pos.x, new_pos.y):
                continue

            value = policy.get(new_pos, 0)
            steps_ = path_map.get(new_pos, steps)
            if steps_ > steps and value > maxv:
                maxv = value
                index = i
        return index

    rows = game.field.leny - 2
    print('Path from {start} to {end} to maximize reward:'.format(
        start=game._start,
        end=game._end,
    ))

    for irow in range(rows):
        for icol in range(game.field.lenx-2):
            pos = Position(x=icol+1, y=irow+1)
            if pos in path_map:
                a = get_direction(pos)
                text = '{} '.format(DIRECTION_ARROWS[a])
            else:
                text = 'X '

            if pos == game._start:
                click.secho(text, fg='green', nl=False)
            elif pos == game._end:
                click.secho(text, fg='red', nl=False)
            else:
                click.echo(text, nl=False)
        print()


def print_policy(policy, game):
    """
    ↖ ↑ ↗
    ← · →
    ↙ ↓ ↘
    """

    def get_direction(pos):
        maxv = 0
        index = 0
        for i, move in enumerate(DIRECTIONS.values()):
            new_pos = pos + move
            if game.field._is_border(new_pos.x, new_pos.y):
                continue

            value = policy.get(new_pos, 0)
            if value > maxv:
                maxv = value
                index = i
        return index

    print('Policy for each state to maximize cumulative reward '
          'within given steps:')
    rows = game.field.leny - 2
    for irow in range(rows):
        for icol in range(game.field.lenx-2):
            pos = Position(x=icol+1, y=irow+1)
            if pos in policy:
                # lookup path backwards
                a = get_direction(pos)
                ch = DIRECTION_ARROWS[a]
                # print(policy[pos], end=' ')
                print(ch, end=' ')
            else:
                print('X', end=' ')
        print()


def main():
    game = Game.create_game_debug()
    scores_map, path_map = dynamic_programming(game)
    max_score = scores_map[game._end]
    print('Max score for the game is:', max_score)
    print_policy(scores_map, game)
    print_path(path_map, scores_map, game)


if __name__ == '__main__':
    main()
