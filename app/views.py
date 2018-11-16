import aiohttp_jinja2
from aiohttp import web


@aiohttp_jinja2.template('index.html')
async def index(request):
    pass



async def get_board_data(request):
    field = [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  3,  7,  3,  9, 11, 22, 12, -1],
        [-1, 33, -1, -1, -1,  0,  0, 19, -1],
        [-1, 33, -1, -1, -1,  4,  0, 13, -1],
        [-1, 22, -1, -1, -1, -1,  0, 11, -1],
        [-1, 22, -1, -1, -1, -1, -1, 17, -1],
        [-1, 31, 21, 27, 18, 14, 13, 14, -1],
        [-1,  5, 10, 11, 10, 10, 11, 20, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]
    data = {
        'field': field,
        'player': {
            'x': 1,
            'y': 3,
        },
        'start': [2, 2],
        'end': [7, 7]
    }

    return web.json_response(data)
