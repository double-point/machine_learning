# encoding:utf-8
# FileName: model
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/4/16 19:11
# Description: svm模型

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 显示所有列
from sklearn import preprocessing, svm, metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from svm.read_data import read_data
from tools import sns_set

pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def feature_data(df_data):
    # print(df_data)
    """1. 数据分布"""
    # 569条数据、32个字段。32个字段中1个object类型，一个int型id，剩下的都是float 类型。
    # 数据中不存在缺失值
    # object类型可能是类别型数据，即最终的预测类型，需要进行处理，先记下
    # 可以发现：其中mean代表平均值、se代表标准差、worst代表最坏值（这里具体指肿瘤的特征最大值）
    # 仔细想一下，一个特征的三个不同维度的数据表示，我们完全可以用一个维度去代表（考虑用mean？）
    print(df_data.info())
    # 查看连续型数据分布
    print(df_data.describe())

    """2. 类别特征向量化"""
    le = preprocessing.LabelEncoder()
    le.fit(df_data['diagnosis'])
    df_data['diagnosis'] = le.transform(df_data['diagnosis'])

    """3. 提取特征"""
    # 提取所有mean 字段和label字段
    df_data_X = df_data.filter(regex='_mean')
    df_data_y = df_data['diagnosis']

    """4. 选择最优的k个特征"""
    # 选择最优的特征
    selector = SelectKBest(f_classif, k=len(df_data_X.columns))
    selector.fit(df_data_X, df_data_y)
    scores = -np.log10(selector.pvalues_)
    indices = np.argsort(scores)[::-1]
    print("Features importance :")
    for feature in range(len(scores)):
        print("%0.1f %s" % (scores[indices[feature]], df_data_X.columns[indices[feature]]))

    """通过画图进行特征选择"""
    sns = sns_set()
    sns.heatmap(df_data[df_data_X.columns].corr(), linewidths=0.1, vmax=1.0, square=True,
                cmap=sns.color_palette('RdBu', n_colors=256),
                linecolor='white', annot=True)
    plt.title('the feature of corr')
    plt.show()

    """分析"""
    # 存在强相关的特征，可以选取其中任一个
    # 即：radius_mean、perimeter_mean和area_mean强相关，我们选取得分最高的perimeter_mean 即可
    df_data_X = df_data_X.drop(['radius_mean', 'area_mean'], axis=1)

    """5. 进行特征归一化/缩放"""
    scaler = preprocessing.StandardScaler()
    df_data_X = scaler.fit_transform(df_data_X)

    return df_data_X, df_data_y


def model_data(data_X, data_y):
    """
    训练模型
    @param data_X:
    @param data_y:
    @return:
    """
    """1.1. 第一种模型验证方法"""
    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2)
    # 创建SVM分类器
    model = svm.LinearSVC()
    # 用训练集做训练
    model.fit(X_train, y_train)
    # 用测试集做预测
    pred_label = model.predict(X_test)
    print('准确率: ', metrics.accuracy_score(pred_label, y_test))

    """1.2. 第二种模型验证方法"""
    # 创建SVM分类器
    model = svm.LinearSVC(max_iter=3000)
    # 使用K折交叉验证 统计svm准确率
    print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, data_X, data_y, cv=10)))

    """2. 通过网格搜索寻找最优参数"""
    parameters = {
        'gamma': np.linspace(0.0001, 0.1),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }
    model = svm.SVC()
    grid_model = GridSearchCV(model, parameters, cv=10, return_train_score=True)
    grid_model.fit(X_train, y_train)
    # 用测试集做预测
    pred_label = grid_model.predict(X_test)
    print('准确率: ', metrics.accuracy_score(pred_label, y_test))
    # 输出模型的最优参数
    print(grid_model.best_params_)


if __name__ == '__main__':
    filepath = 'data.csv'
    df_data = read_data(filepath)
    # 数据特征工程
    df_data_X, df_data_y = feature_data(df_data)
    # 建立模型
    model_data(df_data_X, df_data_y)