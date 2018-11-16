import os
from functools import partial

import jinja2
import aiohttp
import aiohttp_jinja2
from aiohttp import web

import config
from app.routes import setup_routes, setup_static_routes


def run():
    app = web.Application()
    setup_routes(app)
    setup_static_routes(app)
    aiohttp_jinja2.setup(
        app, loader=jinja2.FileSystemLoader(str(config.TEMPLATES_DIR)))

    uprint = partial(print, flush=True)

    uprint('Running aiohttp {}'.format(aiohttp.__version__))
    web.run_app(app, print=uprint, port=config.RUN_PORT)


if __name__ == '__main__':
    run()
