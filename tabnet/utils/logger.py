""" System logger initialization """

import os 
import logging 
from .utils import mkdir


_SUPPORTED_LEVEL = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def init_logger(logger_dir, logger_name, level):
    """
    Logger initialization.

    Arguments:
        logger_dir (str): Directory to save same type of logs.
        logger_name (str): Log filename.
        level (str): Logger level.

    Returns:
        logger
        
    """
    mkdir(logger_dir)

    log_filename = os.path.join(logger_dir, logger_name + '.log')

    # initial handler
    fh = logging.FileHandler(log_filename, 'a', encoding='utf-8')
    sh = logging.StreamHandler()

    # set level
    level_ = _get_level(level)
    fh.setLevel(level_)
    sh.setLevel(level_)

    # set format
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    # initial logger
    logger = logging.getLogger()
    logger.setLevel(level_)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


def _get_level(level):
    if level not in _SUPPORTED_LEVEL:
        raise NotImplementedError('Not supported level.')

    return _SUPPORTED_LEVEL.get(level)


def show_message(msg, logger=None, level='DEBUG'):
        """
        Show message if logger is not None.

        Arguments:
            msg (str):
                Estimator message.

            logger (logging.Logger, or None):
                A Python logger object.

            level (str):
                Logger level.

        Returns:
            None

        """
        if logger is not None:
            if level == 'DEBUG':
                logger.debug(msg)

            elif level == 'INFO':
                logger.info(msg)

            elif level == 'WARNING':
                logger.warning(msg)

            elif level == 'ERROR':
                logger.error(msg)

            elif level == 'CRITICAL':
                logger.critical(msg)

            else:
                raise ValueError('Invalid level.')

        return 
