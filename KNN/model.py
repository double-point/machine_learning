# encoding:utf-8
# FileName: model
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/5/11 23:09
# Description: 通过sklearn 调用knn模型
import os
import time

import pandas as pd
import numpy as np

# 显示所有列
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from KNN.read_data import read_data

pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def get_best_parm(data_train, label_train):
    """
    通过网格搜索确定最优的k值
    @param data_train:
    @param label_train:
    @return:
    """
    # 设置需要搜索的K值，'n_neighbors'是sklearn中KNN的参数
    parameters = {'n_neighbors': [arr for arr in range(1, 20)]}
    # 注意：这里暂时不用指定参数
    knn = KNeighborsClassifier()
    # 通过GridSearchCV来搜索最好的K值
    clf = GridSearchCV(knn, parameters, cv=5)
    clf.fit(data_train, label_train)
    # 输出最好的参数以及对应的准确率
    print("最终最佳准确率：%.2f" % clf.best_score_, "最终的最佳K值", clf.best_params_)


def model(data_train, label_train, data_test, label_test, test_filenames):
    """
    建立knn模型并进行预测
    @param data_train:
    @param label_train:
    @param data_test:
    @param label_test:
    @param test_filenames:
    @return:
    """
    knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
    knn.fit(data_train, label_train)
    predict_y = knn.predict(data_test)
    print("KNN准确率: %.4lf" % accuracy_score(label_test, predict_y))

    # # 检查判断出错的图片
    # for index, label_y, predict_y in zip(range(len(test_filenames)), label_test, predict_y):
    #     if label_y != predict_y:
    #         print(test_filenames[index], label_y, predict_y)


def pre_knn_model():
    """
    方便其他算法调用
    @return:
    """
    os.chdir(os.path.join(os.path.dirname(os.getcwd()), 'KNN'))
    """加载训练数据"""
    dirpath_train = r'dataset\trainingDigits'
    data_train, label_train, train_filenames = read_data(dirpath_train)
    """加载测试数据"""
    dirpath_test = r'dataset\testDigits'
    data_test, label_test, test_filenames = read_data(dirpath_test)

    """获取最优k值"""
    # get_best_parm(data_train, label_train)

    return data_train, label_train, data_test, label_test, test_filenames


if __name__ == '__main__':
    start = time.time()
    """建模预测"""
    data_train, label_train, data_test, label_test, test_filenames = pre_knn_model()
    model(data_train, label_train, data_test, label_test, test_filenames)
    # 运行时间
    end = time.time()
    print('Running time: %s Seconds' % (end - start))