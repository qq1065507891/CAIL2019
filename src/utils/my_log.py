#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :my_log.py
@Description  :
@Time         :2023/02/23 10:39:16
@Author       :Mike Zeng
@Version      :1.0
'''


import os
import logging

from logging import handlers

from src.config.config import config


class MyLog(object):
    def __init__(self, config):
        self.path = config.log_path
        if not os.path.exists(config.log_folder_path):
            os.mkdir(config.log_folder_path)

    def create_logger(self):
        level_relations = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'crit': logging.CRITICAL
        } 
        logger = logging.getLogger(self.path)
        fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        # 设置日志格式
        format_str = logging.Formatter(fmt)
        # 设置日志级别
        logger.setLevel(level_relations.get('info'))
        # 往屏幕上输出+
        
        sh = logging.StreamHandler()
        # 设置屏幕上显示的格式
        sh.setFormatter(format_str)
        # 往文件里写入#指定间隔时间自动生成文件的处理器 
        th = handlers.TimedRotatingFileHandler(
            filename=self.path, when='D', backupCount=3,
            encoding='utf-8')
        # 设置文件里写入的格式
        th.setFormatter(format_str)

        # 把对象加到logger里
        logger.addHandler(sh)  # 把对象加到logger里
        logger.addHandler(th)
        return logger

log = MyLog(config).create_logger()
