#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :process.py
@Description  :
@Time         :2023/02/22 15:48:04
@Author       :Mike Zeng
@Version      :1.0
'''


import os
import codecs
import paddle

from tqdm import tqdm
from paddle.io import TensorDataset

from src.utils.utils import log, load_pkl, save_pkl, ensure_dir


class InputExamples(object):
    """
    保存文本的实例
    """
    def __init__(self, guid, text_a, text_b=None, label=None) -> None:
        """
        :param guid: 文章id
        :param text_a: 文本
        :param text_b: 第二段文本, defaults to None
        :param label: 标签, defaults to None
        """
       
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label =label


class InputFeatures(object):
    """
    保存bert输入数据实例
    :param object: _description_
    :return: _description_
    """
    def __init__(self, input_ids, token_type_ids, seq_length=None, attention_mask=None, label_ids=None):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.seq_length = seq_length
        self.label_ids = label_ids


class Process(object):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        ensure_dir(self.config.out_path)

    def read_file(self, path):
        """
        读取数据
        :param path: 文件所在位置
        :return: list
        """
        text = []
        for p in os.listdir(path):
            file_path = os.path.join(path, p)
            with codecs.open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f.readlines()):
                    content, label = line.strip().split('\t')
                    text.append(InputExamples(guid=i, text_a=content, label=label.split(',')))
        return text

    def get_label_map(self):
        """
        获取标签就map
        :return: dict
        """
        label_map = {
                0: "婚后有子女",
                1: "限制行为能力子女抚养",
                2: "有夫妻共同财产",
                3: "支付抚养费",
                4: "不动产分割",
                5: "婚后分居",
                6: "二次起诉离婚",
                7: "按月给付抚养费",
                8: "准予离婚",
                9: "有夫妻共同债务",
                10: "婚前个人财产",
                11: "法定离婚",
                12: "不履行家庭义务",
                13: "存在非婚生子",
                14: "适当帮助",
                15: "不履行离婚协议",
                16: "损害赔偿",
                17: "感情不和分居满二年",
                18: "子女随非抚养权人生活",
                19: "婚后个人财产"
            }
        save_pkl(self.config.label_map_path, label_map, 'label', 'True')
        return label_map
    
    def get_examples(self, name):
        """
        获取数据
        :param name: 
        """
        label_map = self.get_label_map()
        if name == 'train':
            examples = self.read_file(self.config.train_path)
            data = self._create_examples(examples, label_map, name)
        else:
            examples = self.read_file(self.config.train_path)
            data = self._create_examples(examples, label_map, name)
        return data

    def _create_examples(self, examples, label_map=None, name='train'):
        """
        :param examples: 
        :param label_map: 
        """
        data = self.process_data(examples, label_map, name)
        return data
    
    def process_data(self, examples, label_map, name):
        """
        预处理数据
        :param examples: 读取的文本
        :param label_map: 获取的label_map
        :param name: 处理的类型(train, test, dev)
        """
        feature_dir = os.path.join(self.config.out_path, '{}_{}.pkl'.format(name, self.config.max_len))
        if os.path.exists(feature_dir):
            features = load_pkl(feature_dir, name)
        else:
            features = self.convert_examples_to_features(examples, label_map)
            save_pkl(feature_dir, features, name, use_bert='True')

        log.info(f" Num examples = {len(features)}")

        input_ids, token_type_ids, labels = [], [], []

        for f in tqdm(features):
            input_ids.append(f.input_ids)
            token_type_ids.append(f.token_type_ids)
            labels.append(f.label_ids)

        all_input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64)
        all_token_type_ids = paddle.to_tensor(token_type_ids, dtype=paddle.int64)
        all_labels = paddle.to_tensor(labels, dtype=paddle.float32)

        data = TensorDataset([all_input_ids, all_token_type_ids, all_labels])
        return data

    def convert_examples_to_features(self, examples, label_map=None):
        """
        将数据转换为bert的输入形式
        :param examples: 文本数据
        :param label_map:  defaults to None
        """
        log.info(f'#examples {len(examples)}')

        features = []
        for i, example in enumerate(tqdm(examples)):
            encoding = self.tokenizer.encode(text=example.text_a, max_seq_len=self.config.max_len,
                                             pad_to_max_seq_len=True, return_length=True)
            
            input_ids = encoding['input_ids']
            token_type_ids = encoding['token_type_ids']
            if label_map:
                label = [0 for _ in range(len(label_map))]
                for x in example.label:
                    label[int(x)] = 1
            else:
                label = None

            assert len(input_ids) == self.config.max_len 
            assert len(token_type_ids) == self.config.max_len

            if i < 2:
                log.info("*** Example ***")
                log.info("guid: %s" % (example.guid))
                log.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                log.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                log.info(f"label: {example.label} (id = {[str(x) for x in label]}")
            
            features.append(
                InputFeatures(input_ids=input_ids, token_type_ids=token_type_ids, label_ids=label)
            )
        return features
