import aiohttp_jinja2
from aiohttp import web

from app.game import Game


@aiohttp_jinja2.template('index.html')
async def index(request):
    pass


async def get_board_data(request):
    game = Game.create_game()
    data = {
        'field': game.field._field,
        'player': game._start.to_dict(),
        'start': game._start.to_dict(),
        'end': game._end.to_dict(),
    }

    return web.json_response(data)
