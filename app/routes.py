import config
from . import views


def setup_routes(app):
    router = app.router
    router.add_get('/', views.index)
    router.add_get('/load_data', views.get_board_data)


def setup_static_routes(app):
    app.router.add_static('/static/',
                          path=config.PROJECT_ROOT / 'static',
                          name='static')
    node_modules_path = config.PROJECT_ROOT / 'node_modules'
    if node_modules_path.exists():
        app.router.add_static('/node_modules/',
                              path=node_modules_path,
                              name='node_modules')
