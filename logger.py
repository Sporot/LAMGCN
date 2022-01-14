# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:松饼
@file:logging_test.py
@time:2021/10/22
@des:
"""

import logging

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)

def set_logger(filename):
    logger = logging.getLogger('test')
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

if __name__ == '__main__':
    filename = 'logs/new_exp/test/logger.log'
    logger = set_logger(filename)
    logger.info('i am cool')


