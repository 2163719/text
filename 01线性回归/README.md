# linear regression——线性回归   
## 线性回归的一般形式   
![](https://github.com/2163719/ML-learning-notebooks/blob/master/LR1.png)   
## 代价函数   
采用均方误差来度量，试图使得代价函数也就是均方误差最小化。   
![](https://github.com/2163719/ML-learning-notebooks/blob/master/LR2.png)
## 线性回归的优化方法   
1. **梯度下降法**   
![](https://github.com/2163719/ML-learning-notebooks/blob/master/LR3.png)   
由于这个方法中，参数在每一个数据点上同时进行了移动，因此称为批量梯度下降法，对应的，我们可以每一次让参数只针对一个数据点进行移动，即：   
                          ![](https://github.com/2163719/ML-learning-notebooks/blob/master/LR4.png)   
这个算法就是随机梯度下降法，随机梯度下降法的好处是，当数据点很多时，运行效率更高；缺点是，因为每次只针对一个样本更新参数，未必找到最快路径达到最优值，甚至有时候会出现参数在最小值附近徘徊而不是立即收敛。但当数据量很大的时候，随机梯度下降法经常优于批梯度下降法。   
1.1 **特征缩放**   
我们面对多维特征问题的时候，要保证这些特征都具有相近的尺度，这将帮助梯度下降算法更快地收敛。解决的方法是尝试将所有特征的尺度都尽量缩放到-1到1之间。   
最简单的方法是令：![](https://github.com/2163719/ML-learning-notebooks/blob/master/LR5.png)   
对应代码如下：   
``def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())
``   
1.2 **学习率**   
梯度下降算法的每次迭代受到学习率的影响，如果学习率过小，则达到收敛所需的迭代次数会非常高；如果学习率过大，每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛。通常可以考虑这些学习率：   
                          ![](https://github.com/2163719/ML-learning-notebooks/blob/master/LR6.png)   
2. **正规方程法（最小二乘法）**   
正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：   
                          ![](https://github.com/2163719/ML-learning-notebooks/blob/master/LR7.png)   
假设我们的训练集特征矩阵为𝑋（包含了𝑥_0 = 1）并且我们的训练集结果为向量𝑦，则利用正规方程解出向量
                          ![](https://github.com/2163719/ML-learning-notebooks/blob/master/LR8.png)   
3. **牛顿法**   
4. **拟牛顿法**  
### 梯度下降法与正规方程法的比较   
![](https://github.com/2163719/ML-learning-notebooks/blob/master/LR9.png)   
### **代码及数据见LRcode.zip（单变量）**

