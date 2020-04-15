# encoding:utf-8
# FileName: model
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/4/8 0:03
# Description: 

import pandas as pd
import numpy as np

# 显示所有列
from sklearn import preprocessing, tree
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def read_data(filepath):
    """
    读取数据集
    @param filepath:
    @return:
    """
    df_data = pd.read_csv(filepath, )

    return df_data


def feature_data(df_data):
    """
    特征工程
    @param df_data:
    @return:
    """
    # 对Namelen 字段进行处理
    df_data['NamelenType'] = pd.cut(df_data['Namelen'], bins=[0, 20, 30, 40, 50, 100], labels=[0, 1, 2, 3, 4])
    # 对Numbers 字段进行处理,分别为一个人、两个人、三个人和多个人
    df_data['NumbersType'] = pd.cut(df_data['Numbers'], bins=[0, 1, 2, 3, 20], labels=[0, 1, 2, 3])

    # 对年龄字段进行处理
    df_data['AgeType'] = df_data.loc[:, ['Age', 'Sex']].apply(lambda x: get_person_tag(x[0], x[1]), axis=1)

    """进行特征编码"""
    for feature in ['Sex', 'Embarked', 'CabinType', 'AgeType', ]:
        le = preprocessing.LabelEncoder()
        le.fit(df_data[feature])
        df_data[feature] = le.transform(df_data[feature])

    """进行特征归一化/缩放"""
    scaler = preprocessing.StandardScaler()
    # 对超高票价进行重新赋值
    df_data.loc[df_data['Fare'] > 300, 'Fare'] = 300
    # 对票价进行归一化
    fare_slale_parm = scaler.fit(df_data[['Fare']])
    df_data['Fare_scaled'] = scaler.fit_transform(df_data['Fare'].values.reshape(-1, 1), fare_slale_parm)

    select_features = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'Embarked', 'TitleType', 'NamelenType',
                       'NumbersType', 'CabinType', 'AgeType', 'Fare_scaled']

    return df_data[select_features]


def get_person_tag(age, sex):
    """
    针对年龄和性别对人群分类
    @param age:
    @param sex:
    @return:
    """
    # 儿童/女士/男士 标签
    child_age = 15

    if age < child_age:
        return 'child'
    else:
        if sex == 'male':
            return 'male'
        else:
            return 'female'


def model_data(df_data):
    """
    模型训练
    @param df_data:
    @return:
    """
    """分离训练集和测试集"""
    train_data = df_data[df_data['Survived'].notnull()].drop(['PassengerId'], axis=1)
    test_data = df_data[df_data['Survived'].isnull()].drop(['Survived'], axis=1)
    # 分离训练集特征和标签
    y = train_data['Survived']
    X = train_data.drop(['Survived'], axis=1)

    # 决策树模型
    clf = DecisionTreeClassifier(criterion='entropy')
    # 使用K折交叉验证 统计决策树准确率
    print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, X, y, cv=10)))

    """通过网格搜索寻找最优参数"""
    parameters = {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(2, 10),
        # 'min_impurity_decrease': np.linspace(0, 1, 10),
        'min_samples_split': range(2, 30, 2),
        'class_weight': ['balanced', None]
    }
    gird_clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, return_train_score=True)
    gird_clf.fit(X, y)

    """将生成的决策树保存"""
    # 将dot结果进行可视化 dot -Tpng titanic_tree.dot -o tree.png
    tree.export_graphviz(gird_clf.best_estimator_, feature_names=X.columns, out_file='titanic_tree.dot', filled=True)
    print(gird_clf.best_score_)
    print(gird_clf.best_params_)

    """预测并生成预测文件"""
    pred_labels = gird_clf.predict(test_data.drop(['PassengerId'], axis=1))
    # 以字典的形式来建立dataframe
    result = pd.DataFrame({'PassengerId': test_data['PassengerId'].values, 'Survived': pred_labels.astype(int)})
    # 输出到csv文件
    result.to_csv("submission_data.csv", index=False)


if __name__ == '__main__':
    df_train = read_data('train_data.csv')
    df_test = read_data('test_data.csv')
    # 合并训练集和测试集
    df_data = df_train.append(df_test, sort=False)
    df_data = feature_data(df_data)
    model_data(df_data)
