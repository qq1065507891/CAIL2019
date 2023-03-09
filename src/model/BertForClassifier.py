#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :BertForClassifier.py
@Description  :
@Time         :2023/02/23 10:39:39
@Author       :Mike Zeng
@Version      :1.0
'''

import paddle.nn as nn

from paddlenlp.transformers import ErnieModel


class BertClassifier(nn.Layer):
    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.config = config

        self.ernie = ErnieModel.from_pretrained(config.MODEL_NAME)

        self.drop_out = nn.Dropout(config.drop_out, name='dropout')

        self.fc = nn.Linear(self.ernie.config['hidden_size'], config.hidden_size, name='fc')

        self.classification = nn.Linear(config.hidden_size, self.config.num, name='classifier')

    def forward(self, batch):
        """
        前向转播
        :param batch: input_ids, token_type_ids
        """
        input_ids, token_type_ids = batch
        out_put = self.ernie(input_ids=input_ids, token_type_ids=token_type_ids)[1]

        dropout = self.drop_out(out_put)

        fc = self.fc(dropout)

        dropout = self.drop_out(fc)

        classifier = self.classification(dropout)

        return classifier