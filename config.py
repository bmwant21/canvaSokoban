import os
from pathlib import Path


PROJECT_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
TEMPLATES_DIR = PROJECT_ROOT / 'templates'

DEFAULT_DATE_FORMAT = '%d/%m/%y'
DEFAULT_TIME_FORMAT = '%H:%M'
DEFAULT_DATETIME_FORMAT = '{} {}'.format(DEFAULT_TIME_FORMAT,
                                         DEFAULT_DATE_FORMAT)


# DEVELOPING
RUN_PORT = 8080
DEBUG = False

# Override values from settings_local.py
try:
    import settings_local
    for key, value in settings_local.__dict__.items():
        if key.isupper() and key in globals():
            globals()[key] = value
except ImportError:
    pass

# Override values from environment
for key, value in globals().copy().items():
    if key.isupper() and key in os.environ:
        globals()[key] = os.environ[key]
