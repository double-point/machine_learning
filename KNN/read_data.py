# encoding:utf-8
# FileName: read_data
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/5/11 16:51
# Description: 读取数据：数字识别的训练集和测试集数据
import os

import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def read_data(dirpath):
    """
    读取数据并拼接成数组
    @param dirpath:
    @return:
    """
    # 遍历文件
    filenames = os.listdir(dirpath)
    # 数据保存在二维数组中，标签保存在一维数组中
    data_arr = np.zeros((len(filenames), 1024))
    data_label = []

    for i in range(len(filenames)):
        filename = filenames[i]
        # 读取每个文件的内容
        filepath = os.path.join(dirpath, filename)
        data_arr[i, :] = concat_info(filepath)
        data_label.append(filename[:1])

    return data_arr, data_label, filenames


def concat_info(filepath):
    """
    将32*32拼接成1*1024数组
    @param data:
    @return:
    """
    # 创建1x1024零向量
    data_arr = np.zeros((1, 1024))
    with open(filepath) as file:
        # 按行读取
        for i in range(32):
            # 读一行数据
            line_str = file.readline()
            # 每一行的前32个元素依次添加到data_arr中
            for j in range(32):
                data_arr[0, 32 * i + j] = int(line_str[j])
    return data_arr


if __name__ == '__main__':
    """加载训练数据"""
    dirpath_train = r'dataset\trainingDigits'
    data_train, label_train, train_filenames = read_data(dirpath_train)
    print(data_train)
    print(label_train)
    exit()
    """加载测试数据"""
    dirpath_test = r'dataset\testDigits'
    data_test, label_test, test_filenames = read_data(dirpath_test)
