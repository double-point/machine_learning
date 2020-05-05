# encoding:utf-8
# FileName: read_data
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/4/16 19:08
# Description: 

import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def read_data(filepath):
    """
    读取数据
    @param filepath:
    @return:
    """
    df_data = pd.read_csv(filepath)

    return df_data


if __name__ == '__main__':
    filepath = 'data.csv'
    df_data = read_data(filepath)