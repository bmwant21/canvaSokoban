import logging

FORMAT = '[%(name)s] %(levelname)s:%(message)s'
FORMATTER = logging.Formatter(fmt=FORMAT)


def get_logger(name='default', level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt=FORMATTER)
        logger.addHandler(handler)

    return logger


logger = get_logger()
