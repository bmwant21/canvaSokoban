import aiohttp_jinja2
from aiohttp import web

from app.game import Game
from app.q_learning import get_path


@aiohttp_jinja2.template('index.html')
async def index(request):
    pass


async def get_board_data(request):
    # game = Game.create_game_debug()
    game = Game.create_game_v2()
    path = get_path(game)
    data = {
        'field': game.field._field,
        'player': game._start.to_dict(),
        'start': game._start.to_dict(),
        'end': game._end.to_dict(),
        'moves': game._moves_left,
        'path': [p.to_dict() for p in path],
    }

    return web.json_response(data)
