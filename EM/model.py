# encoding:utf-8
# FileName: model
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/4/26 14:04
# Description: EM算法的模型应用

import pandas as pd
import matplotlib.pyplot as plt

# 显示所有列
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from EM.read_data import read_data
from tools import sns_set

pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def eda_data(df_ori):
    """
    数据的EDA操作
    @param df_ori:
    @return:
    """
    """
    1. 数据整体描述
    """
    df_data = df_ori.copy()
    df_data.drop('英雄', axis=1, inplace=True)
    # 次要定位存在空值
    print(df_data.info())
    print(df_data.describe())
    # 最大攻速为百分比需要替换成数值
    df_data['最大攻速'] = df_data['最大攻速'].apply(lambda str: str.replace('%', ''))
    # 次要定位数据无法填充，且已存在主要定位，直接删除该字段
    df_data.drop('次要定位', axis=1, inplace=True)

    """
    2. 数据关联度分析：通过热力图
    """
    features = df_data.columns.values.tolist()
    print(features)
    """通过画图进行特征选择"""
    sns = sns_set()
    sns.heatmap(df_data[features].corr(), linewidths=0.1, vmax=1.0, square=True, cmap=sns.color_palette('RdBu', n_colors=256),
                linecolor='white', annot=True)
    plt.title('the feature of corr')
    # plt.show()

    duplicates_features = ['生命成长', '最大法力', '法力成长', '物攻成长', '物防成长', '每5秒回血成长', '最大每5秒回血', '每5秒回蓝成长']
    features = features
    for feature in duplicates_features:
        features.remove(feature)
    print(features)
    df_data = df_data[features]

    """
    3. 数据量纲化
    """
    """通过标签编码实现特征量化"""
    for feature in ['攻击范围', '主要定位']:
        le = preprocessing.LabelEncoder()
        le.fit(df_data[feature])
        df_data[feature] = le.transform(df_data[feature])
    # df_data['攻击范围'] = df_data['攻击范围'].map({'远程': '1', '近战':''})

    """采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1"""
    stas = StandardScaler()
    df_data = stas.fit_transform(df_data)

    return df_data


def model_data(df_ori, df_data):
    """
    建模
    @param df_ori:
    @param df_data:
    @return:
    """
    print(df_ori)
    # 构造GMM聚类
    gmm = GaussianMixture(n_components=30, covariance_type='full')
    gmm.fit(df_data)

    # 训练数据
    prediction = gmm.predict(df_data)
    print(prediction)
    # 将分组结果输出到CSV文件中
    df_ori.insert(0, '分组', prediction)
    print(df_ori.sort_values('分组').head(20))

    s1 = silhouette_score(df_data, prediction, metric='euclidean')
    print(s1)


if __name__ == '__main__':
    """加载数据"""
    filepath = r'dataset\heros.csv'
    df_ori = read_data(filepath)
    # 数据EDA
    df_data = eda_data(df_ori)
    # 建模
    model_data(df_ori, df_data)