#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_helpers.py
# @Author: Song bing yan
# @Date  : 2020/10/27
# @Des   : additional functions

import logging
import os
def logger_fn(name, input_file, level=logging.INFO):
    """
    set the logger
    :param name: the name of the logger
    :param input_file: the logger file path
    :param level: the logger level
    :return: the logger
    """
    py_logger = logging.getLogger(name)
    py_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(input_file, mode = 'a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    py_logger.addHandler(fh)
    return py_logger

