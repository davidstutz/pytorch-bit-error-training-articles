#!/usr/bin/env python
"""
Some I/O utilities.
"""

import os
import re
import json
import numpy as np
import zipfile
import importlib
import pickle
import gc
import socket
import functools
import platform
from .log import log, LogLevel

# See https://github.com/h5py/h5py/issues/961
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def makedir(dir):
    """
    Creates directory if it does not exist.

    :param dir: directory path
    :type dir: str
    """

    if dir and not os.path.exists(dir):
        os.makedirs(dir)


def remove(filepath):
    """
    Remove a file.

    :param filepath: path to file
    :type filepath: str
    """

    if os.path.isfile(filepath) and os.path.exists(filepath):
        os.unlink(filepath)


def to_float(value):
    """
    Convert given value to float if possible.

    :param value: input value
    :type value: mixed
    :return: float value
    :rtype: float
    """

    try:
        return float(value)
    except ValueError:
        assert False, 'value %s cannot be converted to float' % str(value)


def to_int(value):
    """
    Convert given value to int if possible.

    :param value: input value
    :type value: mixed
    :return: int value
    :rtype: int
    """

    try:
        return int(value)
    except ValueError:
        assert False, 'value %s cannot be converted to float' % str(value)


def get_class(module_name, class_name):
    """
    See https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa.

    :param module_name: module holding class
    :type module_name: str
    :param class_name: class name
    :type class_name: str
    :return: class or False
    """
    # load the module, will raise ImportError if module cannot be loaded
    try:
        m = importlib.import_module(module_name)
    except ImportError as e:
        log('%s' % e, LogLevel.ERROR)
        return False
    # get the class, will raise AttributeError if class cannot be found
    try:
        c = getattr(m, class_name)
    except AttributeError as e:
        log('%s' % e, LogLevel.ERROR)
        return False
    return c


def hostname():
    """
    Get hostname.

    :return: hostname
    :rtype: str
    """

    return socket.gethostname()


def pid():
    """
    PID.

    :return: PID
    :rtype: int
    """

    return os.getpid()


def partial(f, *args, **kwargs):
    """
    Create partial while preserving __name__ and __doc__.

    :param f: function
    :type f: callable
    :param args: arguments
    :type args: dict
    :param kwargs: keyword arguments
    :type kwargs: dict
    :return: partial
    :rtype: callable
    """
    p = functools.partial(f, *args, **kwargs)
    functools.update_wrapper(p, f)
    return p


def partial_class(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


def append_or_extend(array, mixed):
    """
    Append or extend a list.

    :param array: list to append or extend
    :type array: list
    :param mixed: item or list
    :type mixed: mixed
    :return: list
    :rtype: list
    """

    if isinstance(mixed, list):
        return array.extend(mixed)
    else:
        return array.append(mixed)


def one_or_all(mixed):
    """
    Evaluate truth value of single bool or list of bools.

    :param mixed: bool or list
    :type mixed: bool or [bool]
    :return: truth value
    :rtype: bool
    """

    if isinstance(mixed, bool):
        return mixed
    if isinstance(mixed, list):
        return all(mixed)


def display():
    """
    Get the availabel display.

    :return: display, empty if none
    :rtype: str
    """

    if 'DISPLAY' in os.environ:
        return os.environ['DISPLAY']

    return None