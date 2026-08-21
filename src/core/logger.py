from __future__ import annotations

import logging, coloredlogs


def setup_logger(name: str="MAIN",
                 colored: bool=True,
                 level: int=2,
                 is_debugging: bool=True,
                 is_warning: bool=True) -> LoggerWrapper:

    """
    this function sets up a logger

    Parameters
    ----------
    name : str
        name of the logger. Default="MAIN"
    colored : bool
        use colored logs. Default=True
    level : int
        the level that is currently used.
        Default=0
    is_debugging : bool
        use debugging mode. Default=True
    is_warning : bool
        use warning mode. Default=True

    Returns
    -------
    logger : object
        logger object
    """

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)

    # create a custom formatter
    if colored:
        formatter = coloredlogs.ColoredFormatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # create a colored stream handler with the custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # add the handler to the logger and disable propagation
        _logger.addHandler(handler)

    _logger.propagate = False

    # wrapper class
    class LoggerWrapper:
        def __init__(self, logger,
                     level: int,
                     is_debugging: bool=False,
                     is_warning: bool=False):
            self.logger = logger
            self.level = level
            self.is_debugging = is_debugging
            self.is_warning = is_warning

            # self.logger.info(self)

        def __repr__(self):

            return f"LoggerWrapper(name={self.logger.name}," + \
                   f"level={self.level}, " + \
                   f"debugging={self.is_debugging}, " + \
                   f"warning={self.is_warning})"

        def __call__(self, msg: str="", level: int=1):
            if level <= self.level:
                self.logger.info(msg)

        def info(self, msg: str="", level: int=1):
            self(msg, level)

        def warning(self, msg: str=""):
            if self.is_warning:
                self.logger.warning(msg)

        def error(self, msg: str=""):
            if self.is_warning:
                self.logger.error(msg)

        def debug(self, msg):
            if self.is_debugging:
                self.logger.debug(msg)

        def set_debugging(self, is_debugging: bool):
            self.is_debugging = is_debugging

        def set_warning(self, is_warning: bool):
            self.is_warning = is_warning

        def set_level(self, level: int):
            self.level = level

    return LoggerWrapper(logger=_logger, level=level,
                         is_debugging=is_debugging,
                         is_warning=is_warning)


logger = setup_logger()
