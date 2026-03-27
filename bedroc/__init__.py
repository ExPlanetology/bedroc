# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Package level variables and initialises the package logger"""

import logging

__version__: str = "0.2.0"

# Create the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def complex_formatter() -> logging.Formatter:
    """Complex formatter"""
    fmt: str = "[%(asctime)s - %(name)-30s - %(lineno)03d - %(levelname)-9s - %(funcName)s()]"
    fmt += " - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def simple_formatter() -> logging.Formatter:
    """Simple formatter for logging

    Returns:
        Formatter for logging
    """
    fmt: str = "[%(asctime)s - %(name)-30s - %(levelname)-9s] - %(message)s"
    datefmt: str = "%H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def debug_logger() -> logging.Logger:
    """Sets up debug logging to the console.

    Returns:
        A logger
    """
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    package_logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return package_logger


def debug_file_logger() -> logging.Logger:
    """Sets up info logging to the console and debug logging to a file.

    Returns:
        A logger
    """
    # Console logger
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    package_logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    package_logger.addHandler(console_handler)
    # File logger
    file_handler: logging.Handler = logging.FileHandler(f"{__package__}.log")
    file_formatter: logging.Formatter = complex_formatter()
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    package_logger.addHandler(file_handler)

    return package_logger
