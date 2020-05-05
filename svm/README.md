

#### 实战项目

##### 1. 数据集

SVM的经典数据集：乳腺癌诊断

医疗人员采集了患者乳腺肿块经过细针穿刺 (FNA) 后的数字化图像，并且对这些数字图像进行了特征提取，这些特征可以描述图像中的细胞核呈现。通过这些特征可以将肿瘤分成良性和恶性

本次数据一共569条、32个字段，先来看一下具体数据字段吧

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/svm/data_means.jpg)

其中mean结尾的代表平均值、se结尾的代表标准差、worst结尾代表最坏值（这里具体指肿瘤的特征最大值）

所有其实主要有10个特征字段，一个id字段，一个预测类别字段

`我们的目的是通过给出的特征字段来预测肿瘤是良性还是恶性`

准备好了吗？3,2,1 开始

<br>

##### 2. 数据EDA

> EDA:Exploratory Data Analysis探索性数据分析

先来看数据分布情况

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/svm/svm_sz_02.png)

一共569条、32个字段

32个字段中1个object类型，一个int型id，剩下的都是float 类型。

`另外：数据中不存在缺失值`

大胆猜测一下，object类型可能是类别型数据，即最终的预测类型，需要进行处理，先记下

再来看连续型数据的统计数据：

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/svm/svm_sz_03.png)

好像也没啥问题（其实因为这个数据本身比较规整）

那直接开始特征工程吧

<br>

##### 3. 特征工程

首先就是将类别数据连续化

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/svm/svm_sz_04.png)

再来观察每一个特征的三个指标：均值、标准差和最大值。

`优先选择均值，最能体现该指特征的整体情况`

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/svm/svm_sz_05.png)

现在还有十个特征，我们通过热力图来看一下特征之间的关系

```python
# 热力图查看特征之间的关系
sns.heatmap(df_data[df_data_X.columns].corr(), linewidths=0.1, vmax=1.0, square=True,
			cmap=sns.color_palette('RdBu', n_colors=256),
			linecolor='white', annot=True)
plt.title('the feature of corr')
plt.show()
```

热力图是这样的：

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/svm/svm_heatmap.png)

`我们发现radius_mean、perimeter_mean和area_mean这三个特征强相关，那我们只保留一个就行了`

这里保留热力图里面得分最高的perimeter_mean

最后一步

`因为是连续数值，最好对其进行标准化`

标准化之后的数据是这样的：

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/svm/svm_sz_06.png)

<br>

##### 4. 训练模型

上面已经做好了特征工程，我们直接塞进模型看看效果怎么样

因为并不知道数据样本到底是否线性可分，所有我们都来试一下两种算法

先来看看LinearSVC 的效果

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/svm/svm_sz_07.png)

效果很好，简直好的不行

> 这个准确率就不要纠结了，后面真正做实际案例的时候再纠结准确率吧

ok，还有SVC的效果

因为SVC需要设置参数，直接通过网格搜索让机器自己找到最优参数

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/svm/svm_sz_08.png)

效果更好，小一我一时都惊呆了

可以看出，最终模型还是选择rbf高斯核函数，果然实至名归

<br>

好了，今天的项目就到这了

主要是通过数据EDA+特征工程完成了数据方面的工作，然后通过交叉验证+网格搜索确定了最优模型和最优参数 

<br><br>

#### 写在后面的话

DataWhale四月学习的最后一个算法，其实SVM自己之前就看过，也写了相应的笔记，所以总结起来就比较游刃有余。

如果看了前几篇文章的同学应该会发现现在总结还有点早，因为第四篇就是一个凑数的，那个，其实我还在写，写完了会及时更上来的。

第四篇本来是关于条件随机场的内容，结果我发现HMM自己还不是很懂，于是，第四篇就成了HMM算法了，等到HMM写完我会继续更上条件随机场这一节的。

算法学起来确实很吃力，不过一个算法+一个项目的学，应该会比单纯推算法有意思些，也会有成就感些。

**我是小一，第一步的一，我们下节见！**



对了，思维导图奉上：

![文章首发：公众号『知秋小一』](https://raw.githubusercontent.com/double-point/GraphBed/master/machine_learning/SVM/mind.png)

<br>







