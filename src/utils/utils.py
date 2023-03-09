#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :utils.py
@Description  :
@Time         :2023/02/23 10:39:05
@Author       :Mike Zeng
@Version      :1.0
'''

import os
import pickle
import codecs
import time

import paddle
import random
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta

from src.utils.my_log import log


def ensure_dir(path):
    """
    确定文件夹是否存在
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)


def make_seed(num):
    """
    设置随机种子
    :param num:
    :return:
    """
    random.seed(num)
    paddle.seed(num)
    np.random.seed(num)


def save_pkl(path, obj, obj_name, use_bert=False):
    """
    将文件保存为二进制文件
    :param path:
    :param obj:
    :param obj_name:
    :param use_bert:
    :return:
    """
    log.info(f'{obj_name} save in {path}, use_bert {use_bert}')
    with codecs.open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path, obj_name):
    """
    读取二进制文件
    :param path:
    :param obj_name:
    :return:
    """
    log.info(f'load {obj_name} in {path}')
    with codecs.open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_time_idf(start_time):
    """
    计算用时多少
    :param start_time:
    :return:
    """
    end_time = time.time()
    idf = end_time - start_time
    return timedelta(seconds=int(round(idf)))


def training_curve(loss, acc, val_loss=None, val_acc=None):
    """
    loss和acc的趋势图
    :param loss:
    :param acc:
    :param val_loss:
    :param val_acc:
    :return:
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(loss, color='r', label='Training Loss')
    if val_loss is not None:
        ax[0].plot(val_loss, color='g', lable='Validation Loss')
    ax[0].legend(loc='best', shadow=True)
    ax[0].grid(True)

    ax[1].plot(acc, color='r', label='Training Accuracy')
    if val_acc is not None:
        ax[1].plot(val_acc, color='g', lable='Validation Accuracy')
    ax[1].legend(loc='best', shadow=True)
    ax[0].grid(True)
    plt.show()
