# encoding:utf-8
# FileName: data_eda
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/4/11 23:43
# Description: eda:探索性数据分析
import jieba
import pandas as pd
import numpy as np

# 显示所有列
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

from naive_bayes.read_data import read_label, read_stop_words, read_data

pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def eda_cut_words(dataset):
    """
    对数据进行分词
    @param dataset:
    @param stop_words:
    @return:
    """
    df_data = pd.DataFrame(list(dataset.items()), columns=['content', 'label'])

    """对内容进行分词"""
    # 使用jieba 分词直接切分
    df_data['cut_content'] = df_data['content'].apply(lambda str: cut_words(jieba.cut(str)))
    return df_data


def cut_words(textcut):
    context = ''
    for word in textcut:
        context += word + ' '

    return context


def model_data(data, df_stop_words):
    """

    @param data:
    @param df_stop_words:
    @return:
    """
    """使用停用词"""
    list_stop_words = df_stop_words['stop_words'].to_list()
    tfidf_vec = TfidfVectorizer(stop_words=list_stop_words, max_df=0.5)
    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(data['cut_content'], data['label'], test_size=0.2)
    # 计算权重
    X_train_features = tfidf_vec.fit_transform(X_train)
    X_test_features = tfidf_vec.transform(X_test)
    # print(tfidf_vec)
    # print(type(features))
    # print(features.shape)
    # print(tfidf_vec.vocabulary_)
    # print(tfidf_vec.stop_words)
    # print(tfidf_vec.idf_)

    """建立模型并进行训练"""
    # 使用多项式朴素贝叶斯进行预测
    clf = MultinomialNB(alpha=0.001).fit(X_train_features, y_train)
    predict_label = clf.predict(X_test_features)
    # 计算概率
    print(clf.predict_proba(X_test_features))
    print(predict_label)
    print('准确率为：', accuracy_score(y_test, predict_label))

    # 使用伯努利朴素贝叶斯进行预测
    clf = BernoulliNB(alpha=0.001).fit(X_train_features, y_train)
    predict_label = clf.predict(X_test_features)
    print('准确率为：', accuracy_score(y_test, predict_label))

    GaussianNB(priors=None)


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

    """加载停用词，"""
    df_data = eda_cut_words(df_dataset)

    """计算TF-IDF，建立模型"""
    model_data(df_data, df_stop_words)