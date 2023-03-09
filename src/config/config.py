#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :config.py
@Description  :
@Time         :2023/02/23 10:39:27
@Author       :Mike Zeng
@Version      :1.0
'''

import os


class Config(object):
    curPath = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.split(os.path.split(curPath)[0])[0]

    data_path = os.path.join(root_path, 'data')
    train_path = os.path.join(root_path, 'data/train')
    dev_path = os.path.join(root_path, 'data/test')

    model_path = os.path.join(root_path, 'model')

    log_folder_path = os.path.join(root_path, 'log')
    log_path = os.path.join(log_folder_path, 'log.txt')

    out_path = os.path.join(data_path, 'out')

    label_map_path = os.path.join(out_path, 'label_list.pkl')

    MODEL_NAME = 'ernie-3.0-base-zh'

    max_len = 360
    drop_out = 0.3

    num = 20
    epochs = 30
    batch_size = 32
    hidden_size = 1024

    learning_rate = 5e-5
    require_improvement = 5000

config = Config()
