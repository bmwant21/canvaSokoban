from app.game import Game
from app.value_iteration import get_available_states_from
from app.field import DIRECTIONS


def get_action_index(action_name):
    index = list(DIRECTIONS.keys()).index(action_name)
    assert 0 <= index < 4
    return index


def test_available_states():
    game = Game.create_game_debug()
    # new state if border
    # top left corner, going down
    action = get_action_index('down')
    states_1 = get_available_states_from(
        state=0,
        action=action,
        game=game,
    )
    assert len(states_1) == 3

    # new state is top left corner
    # below top left corner, going up
    action = get_action_index('up')
    states_2 = get_available_states_from(
        state=4,
        action=action,
        game=game,
    )
    assert len(states_2) == 2

    # new state is impossible
    # top left corner, going up
    action = get_action_index('up')
    states_3 = get_available_states_from(
        state=0,
        action=action,
        game=game,
    )
    assert not states_3

    # new state is in the middle of the board
    # bottom cell from top left corner, going right
    action = get_action_index('right')
    states_4 = get_available_states_from(
        state=4,
        action=action,
        game=game,
    )
    assert len(states_4) == 4


def test_probabilities():
    game = Game.create_game_debug()
    # two options available; 0.5 probability
    # below top right corner, going up
    action = get_action_index('up')
    states_1 = get_available_states_from(
        state=7,
        action=action,
        game=game,
    )
    for p, *_ in states_1:
        assert p == 1/2

    # three options available; 0.(3) probability
    # top right corner, going left
    action = get_action_index('left')
    states_2 = get_available_states_from(
        state=3,
        action=action,
        game=game,
    )
    for p, *_ in states_2:
        assert p == 1/3

    # four options available; 0.25 probability
    # left to top right corner, going down
    action = get_action_index('down')
    states_3 = get_available_states_from(
        state=2,
        action=action,
        game=game,
    )
    for p, *_ in states_3:
        assert p == 1/4


def test_rewards():
    pass
