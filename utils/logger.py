import logging
from sys import stdout

# adding trace functionality for logger
_trace_installed = False
_warning_installed = False

def install_trace_logger():
    global _trace_installed
    if _trace_installed:
        return 
    level = logging.TRACE = logging.DEBUG - 5

    def log_logger(self, message, *args, **kwargs):
        if self.isEnabledFor(level):
            kwargs.setdefault('stacklevel', 2)
            self._log(level, message, args, **kwargs)
    logging.getLoggerClass().trace = log_logger

    def log_root(msg, *args, **kwargs):
        kwargs.setdefault('stacklevel', 2)
        # Use logging.log() to log to the root logger at the custom level
        logging.log(level, msg, *args, **kwargs)
    logging.addLevelName(level, "TRACE")
    logging.trace = log_root
    _trace_installed = True

def install_warning_logger():
    global _warning_installed
    if _warning_installed:
        return 
    level = logging.WARNING = logging.INFO + 5

    def log_logger(self, message, *args, **kwargs):
        if self.isEnabledFor(level):
            kwargs.setdefault('stacklevel', 2)
            self._log(level, message, args, **kwargs)
    logging.getLoggerClass().warning = log_logger

    def log_root(msg, *args, **kwargs):
        kwargs.setdefault('stacklevel', 2)
        logging.log(level, msg, *args, **kwargs)
    logging.addLevelName(level, "WARNING")
    logging.warning = log_root
    _warning_installed = True

install_trace_logger()
install_warning_logger()

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    purple = "\x1b[35;20m"
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    cyan = "\x1b[36;20m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.TRACE   : yellow   + "[%(name)s]" + reset + " -- [" + yellow   + "%(levelname)s" + reset + "]: %(message)s",
        logging.DEBUG   : yellow   + "[%(name)s]" + reset + " -- [" + grey     + "%(levelname)s" + reset + "]: %(message)s",
        logging.INFO    : yellow   + "[%(name)s]" + reset + " -- [" + green    + "%(levelname)s" + reset + "]: %(message)s",
        logging.WARNING : yellow   + "[%(name)s]" + reset + " -- [" + purple   + "%(levelname)s" + reset + "]: %(message)s" + purple+ " (%(filename)s)"+reset,
        logging.ERROR   : red      + "%(asctime)s -- [%(name)s]" + reset + " -- [" + red + "%(levelname)s" + reset + "]: %(message)s \n _______ %(pathname)s %(filename)s:%(lineno)d",
        logging.CRITICAL: bold_red + "%(asctime)s -- [%(name)s]" + reset + " -- [" + red + "%(levelname)s" + reset + "]: %(message)s \n _______ %(pathname)s %(filename)s:%(lineno)d"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class loglevels:
    TRACE    = logging.TRACE
    DEBUG    = logging.DEBUG
    INFO     = logging.INFO
    WARNING  = logging.WARNING
    ERROR    = logging.ERROR   
    CRITICAL = logging.CRITICAL

def setup_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler(stdout)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
    
    # Attach the loglevels class for convenience.
    logger.loglevels = loglevels
    return logger