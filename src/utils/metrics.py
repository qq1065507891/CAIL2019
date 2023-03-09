# -*- coding: utf-8 -*-
# @Time    : 2023/2/28 23:02
# @Author  : Mike Zeng
# @Site    : 
# @File    : metrics.py
# @Software: PyCharm 
# @Comment :

import numpy as np

from paddle.metric import Metric
from sklearn.metrics import f1_score, accuracy_score


class MultiLabelReport(Metric):
    """
    AUC and F1Score for multi-label text classification task.
    """
    def __init__(self, name='MultiLabelReport', average='micro'):
        super(MultiLabelReport, self).__init__()
        self.average = average
        self._name = name
        self.reset()

    def f1_score(self, y_prob):
        '''
        Returns the f1 score by searching the best threshhold
        '''
        best_score = 0
        for threshold in [i * 0.01 for i in range(100)]:
            self.y_pred = y_prob > threshold
            score = f1_score(y_pred=self.y_pred, y_true=self.y_true, average=self.average)
            if score > best_score:
                best_score = score
        return best_score

    def acc(self, y_prob):
        """
        Returns the acc by searching the best threshhold
        """
        best_score = 0
        for threshold in [i * 0.01 for i in range(100)]:
            self.y_pred = y_prob > threshold
            score = accuracy_score(y_pred=self.y_pred, y_true=self.y_true)
            if score > best_score:
                best_score = score
        return best_score

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.y_prob = None
        self.y_true = None

    def update(self, probs, labels):
        if self.y_prob is not None:
            self.y_prob = np.append(self.y_prob, probs.numpy(), axis=0)
        else:
            self.y_prob = probs.numpy()
        if self.y_true is not None:
            self.y_true = np.append(self.y_true, labels.numpy(), axis=0)
        else:
            self.y_true = labels.numpy()

    def accumulate(self):
        f1 = self.f1_score(y_prob=self.y_prob)
        acc = self.acc(y_prob=self.y_prob)
        return f1, acc

    def name(self):
        """
        Returns metric name
        """
        return self._name