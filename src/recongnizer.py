#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :recongnizer.py
@Description  :
@Time         :2023/02/23 10:38:48
@Author       :Mike Zeng
@Version      :1.0
'''

import os
import paddle
import time
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import ErnieTokenizer, LinearDecayWithWarmup
from paddle.io import DataLoader, TensorDataset
from paddle.optimizer import AdamW

from src.model.BertForClassifier import BertClassifier
from src.utils.process import Process
from src.utils.utils import log, make_seed, get_time_idf, load_pkl
from src.utils.metrics import MultiLabelReport


class Recongizer(object):
    def __init__(self, config) -> None:
        self.config = config
        log.info('*******************构建模型********************')
        self.tokenizer = ErnieTokenizer.from_pretrained(self.config.MODEL_NAME)

        self.model = BertClassifier(self.config)
        self.config.model_path = os.path.join(config.model_path, 'bert_gru_ner.pdparams')

        if os.path.exists(self.config.model_path):
            log.info('*******************加载模型********************')
            state_dict = paddle.load(config.model_path)
            self.model.set_state_dict(state_dict)

    def train(self):
        make_seed(1001)

        log.info('**********数据预处理************')
        
        start_time = time.time()

        process = Process(self.config, self.tokenizer)

        train_featurces = process.get_examples('train')

        dev_featurces = process.get_examples('dev')

        train_loader = DataLoader(
            train_featurces,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )

        dev_loader = DataLoader(
            dev_featurces,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0
        )

        end_time = get_time_idf(start_time)

        log.info(f'*********数据预处理完成， 用时{end_time}**********')
        num_training_steps = len(train_loader) * self.config.epochs

        self.model.train()
        # 学习率
        lr_scheduler = LinearDecayWithWarmup(self.config.learning_rate, num_training_steps, 0.0)

        # 衰减的参数
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        # 梯度剪切
        clip = nn.ClipGradByGlobalNorm(clip_norm=1.0)

        # 优化器
        optimizer = AdamW(
            learning_rate=lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=0.0,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=clip
        )

        # 交叉熵损失
        criterion = nn.loss.BCEWithLogitsLoss()
        # 评估的时候采用准确率指标
        metric = MultiLabelReport()

        total_batch = 0  # 记录进行多少batch
        dev_best_loss = float('inf')  # 记录上次最好的验证集loss
        last_improve = 0  # 记录上次提升的batch
        flag = False  # 停止位的标志, 是否很久没提升

        log.info("***** Running training *****")

        start_time = time.time()
    
        for epoch in range(self.config.epochs): 
            log.info(f'Epoch  [{epoch + 1}]/[{self.config.epochs}]')

            for i, batch in enumerate(train_loader):
                *x, y = batch

                output = self.model(x)
                loss = criterion(output, y)

                probs = F.sigmoid(output)
                metric.update(probs, y)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                # 每训练50次输出在训练集和验证集上的效果
                if total_batch % 100 == 0:
                    # correct = metric.compute(output, y)
                    # metric.update(correct)
                    # acc = metric.accumulate()
                    train_f1, train_acc = metric.accumulate()
                    
                    dev_f1, dev_acc, dev_loss = self.evaluate(dev_loader, metric)
                    
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        paddle.save(self.model.state_dict(), self.config.model_path)
                        improve = '+'
                        last_improve = total_batch
                    else:
                        improve = "-"
                        
                    time_idf = get_time_idf(start_time)

                    msg = f'Iter:{total_batch}, Train Loss:{loss.item():.5f},  ' \
                          f'Train f1_score:{train_f1:.3f}, Train acc:{train_acc:.3f},  ' \
                          f'Val Loss:{dev_loss:.5f}, Val f1_scorce:{dev_f1:.3f}, Val acc:{dev_acc:.3f},  ' \
                          f'Time:{str(time_idf)}  {improve}'

                    log.info(msg)
                    self.model.train()
                    metric.reset()

                total_batch = total_batch + 1
                # 在验证集上loss超过1000batch没有下降, 结束训练 
                if total_batch - last_improve > self.config.require_improvement:
                    log.info('在验证集上loss超过10000次训练没有下降, 结束训练')
                    flag = True
                    break

            if flag:
                break

    @paddle.no_grad()
    def evaluate(self, loader, metric):
        """
        验证集的正确率和loss
        :param loader: 验证集
        :param metric: 计算准确率函数
        """
        self.model.eval()
        loss_total = []
        loss_fn = nn.BCEWithLogitsLoss()
        metric.reset()
        
        for batch in loader:
            *x, labels = batch
            outputs = self.model(x)
            loss = loss_fn(outputs, labels)
            loss_total.append(loss)
            probs = F.sigmoid(outputs)
            metric.update(probs, labels)

        f1, acc = metric.accumulate()

        #     correct = metric.compute(outputs, labels)
        #     metric.update(correct)
        #
        # acc = metric.accumulate()

        return f1, acc, np.mean(loss_total)

    @paddle.no_grad()
    def predict(self, inputs, threshold=0.5):
        """
        模型预测
        :param inputs: 输入文本or list or dict
        :return:
        """
        self.model.eval()
        if isinstance(inputs, list):
            inputs = inputs
        elif isinstance(inputs, dict):
            inputs = [value for key, value in inputs.items()]
        elif isinstance(inputs, str):
            inputs = [inputs]
        else:
            raise ValueError

        label_map = load_pkl(self.config.label_map_path, 'label_map')
        label_dict = {value: key for key, value in label_map.items()}

        outputs = {}

        for i, line in enumerate(inputs):
            labels = ''
            encoding = self.tokenizer.encode(text=line, max_seq_len=self.config.max_len,
                                             pad_to_max_seq_len=True)

            input_ids = paddle.to_tensor([encoding['input_ids']])
            token_type_ids = paddle.to_tensor([encoding['token_type_ids']])
            x = [input_ids, token_type_ids]
            out_put = self.model(x)

            out_put = F.sigmoid(out_put).squeeze(0)

            out_put = [1 if x > threshold else 0 for x in out_put]
            out_put = [i for i, x in enumerate(out_put) if x == 1]

            for out in out_put:
                label = label_dict[out]
                labels = label + '\t'

            outputs[i] = [{'inputs': line, 'label': labels.strip()}]
        return outputs
