# encoding:utf-8
# FileName: read_data
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/4/26 11:01
# Description: 加载数据

import pandas as pd
import numpy as np

# 显示所有列
from sklearn.datasets import load_files

pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def read_data(filepath):
    data = pd.read_csv(filepath, encoding='gbk')
    print(data)

    return data


if __name__ == '__main__':
    """加载数据"""
    filepath = r'dataset\heros.csv'
    df_data = read_data(filepath)
