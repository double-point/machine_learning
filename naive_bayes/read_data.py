# encoding:utf-8
# FileName: read_data
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/4/10 17:34
# Description: 读取所有文件
import os

import jieba
import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def read_label(filepath):
    """
    读取对应的标签文件
    @param filepath:
    @return:
    """
    data = pd.read_table(filepath, header=None, names=['number', 'label'], encoding='utf-8')

    return data


def read_stop_words(filepath):
    """
    读取对应的停用词
    @param filepath:
    @return:
    """
    data = pd.read_table(filepath, header=None, names={'stop_words'}, encoding='utf-8')
    return data


def read_data(dirpath, label):
    dic_data = {}
    for dirname_child in os.listdir(dirpath):
        # 匹配每个文件夹对应的新闻类型
        type = label.loc[label.number == dirname_child, 'label'].values[0]
        # 遍历文件夹
        dirpath_child = os.path.join(dirpath, dirname_child)
        for filename in os.listdir(dirpath_child):
            filepath = os.path.join(dirpath_child, filename)
            print(filepath)
            # 按行读取数据
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            """用字典存储数据"""
            dic_data[content] = type

    return dic_data


if __name__ == '__main__':
    """加载标签数据"""
    filepath_label = r'dataset\class_list.txt'
    df_label = read_label(filepath_label)

    """加载停用词"""
    filepath_stop_words = r'dataset\stopwords.txt'
    df_stop_words = read_stop_words(filepath_stop_words)

    """加载训练数据"""
    dirpath_dataset = r'dataset\train_data'
    df_dataset = read_data(dirpath_dataset, df_label)
