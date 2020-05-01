#### 项目实战

##### 1. 准备工作

如何创建高斯聚类呢，我们需要先了解一下高斯聚类的参数

在sklearn 中，高斯聚类可以这样创建：

```Python
# 创建高斯聚类模型
gmm = GaussianMixture(n_components=1, covariance_type='full', max_iter=100)
```

解释一下主要的参数：

- n_components：即高斯混合模型的个数，也就是我们要聚类的个数，默认值为 1。
- covariance_type：代表协方差类型。一个高斯混合模型的分布是由均值向量和协方差矩阵决定的，所以协方差的类型也代表了不同的高斯混合模型的特征。
- max_iter：代表最大迭代次数，EM 算法是由 E 步和 M 步迭代求得最终的模型参数，这里可以指定最大迭代次数，默认值为 100。

其中协方差类型covariance_type又四种取值，分别是：

- covariance_type=full，代表完全协方差，也就是元素都不为 0；
- covariance_type=tied，代表相同的完全协方差；
- covariance_type=diag，代表对角协方差，也就是对角不为 0，其余为 0；
- covariance_type=spherical，代表球面协方差，非对角为 0，对角完全相同，呈现球面的特性。
  

需要注意的是，聚类的个数往往是由业务决定的，比如对用户收入进行聚类，可以分为：高收入群体、中收入群体、低收入群体，根据用户价值进行聚类，可以分为：高价值用户、中价值用户、低价值用户、无价值用户等等

当然如果你无法确定聚类的个数，可以通过设置不同的聚类个数进而选择具有最优效果的模型

<br>

##### 2. 了解数据

本次实战我们的数据是王者荣耀的英雄属性数据，通过对69个英雄的22个属性数据，其中包括英雄的最大生命、生命成长、最大发力、最高物攻等等，通过每个英雄之间的特征，对69个英雄进行”人以群分，物以类聚“。感兴趣的同学可以尝试一下最终的结果能否应用于实际游戏中。

ok，先来看看我们本次数据的整体描述

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/EM/dataset.png)

再来看看各个英雄属性的整体情况

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/EM/EM_1.png)

一共22个英雄属性（不包括姓名），其中次要定位存在空值，且空值较多

再来看看数值型数据的整体分布情况

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/EM/EM_2.png)

数据分布没有什么异常，但是应该需要考虑进行标准化，这个后面再说

最大攻速字段应该是数值型的，我们需要对其进行处理

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/EM/EM_0.png)

另外，次要定位属性缺失值太多，而且没有有效的填充方法，直接删掉它

```python
# 最大攻速为百分比需要替换成数值
df_data['最大攻速'] = df_data['最大攻速'].apply(lambda str: str.replace('%', ''))
# 次要定位数据无法填充，且已存在主要定位，直接删除该字段
df_data.drop('次要定位', axis=1, inplace=True)
```

##### 3. 数据探索

一共只有69数据，但是却有22个属性，是否存在属性重复的情况呢？

我们知道在建模过程中，重复的属性对最终结果不会产生影响

所以我们可以通过关联度分析，看一下数据之间的关联度情况，这种方式在前面的实战种很多次都用到过。

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/EM/features_hotmap.png)

可以看到，用红框标出的，都是属性之间相互关联度特别大的，对于这些我们只需要保留其中的一种属性即可

通过筛选，我们最终确定了13个英雄属性

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/EM/EM_3.png)

再来看英雄属性：攻击范围和主要定位，是离散型特征，直接对其进行特征量化

```python
"""通过标签编码实现特征量化"""
for feature in ['攻击范围', '主要定位']:
	le = preprocessing.LabelEncoder()
	le.fit(df_data[feature])
	df_data[feature] = le.transform(df_data[feature])
```

最后就是数据的规范化，直接通过Z-Score进行数据规范

```python
"""采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1"""
stas = StandardScaler()
df_data = stas.fit_transform(df_data)
```

##### 4. 建模

选用我们前面提到的GMM进行建模

```python
# 构造GMM聚类
gmm = GaussianMixture(n_components=30, covariance_type='full')
gmm.fit(df_data)

# 训练数据
prediction = gmm.predict(df_data)
```

最终的模型聚类结果是这样的：

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/EM/EM_5.png)

<br>

##### 5. 总结

上面的图中放了前20的英雄，组号相同的英雄表示属性相近，感兴趣的同学不妨在游戏中试试？

另外，聚类算法属于无监督的学习方式，我们并没有实际的结果可以进行对比来区别模型的准确率。

这里我们试着用轮廓系数进行模型的评分

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/EM/EM_6.png)

最后得分0.246，也不是很高，说明聚类的效果不是特别好，应该还是英雄属性的原因，例如，通过主要定位就可以对英雄就行聚类，或者通过攻击范围，但是这两个属性和其他属性的结合，有时候并非是最好的，对游戏理解比较深刻的同学可以考虑一下优化方法。

