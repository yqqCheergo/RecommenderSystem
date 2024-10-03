本文件夹利用**ItemCF**算法编写一个推荐系统，当用户进行电影浏览时，向用户推荐和该部电影相似的电影

#### 实现思路

首先使用训练数据得到用户的偏好信息矩阵、物品的特征信息矩阵

然后计算用户对未进行评分电影的偏好分

最后选取前K个推荐给用户

#### 准备数据

数据集下载地址：https://grouplens.org/datasets/movielens/1m/

其中，`movies.dat`记录了每部电影所属类型；`ratings.dat`记录了用户对电影的评分，列分别表示UserID::MovieID::Rating::Timestamp

<img src="C:\Users\11842\AppData\Roaming\Typora\typora-user-images\image-20241001221815261.png" alt="image-20241001221815261" style="zoom:50%;" />

#### 模型实现

算法使用的是基于物品的协同过滤，即ItemCF

电影之间的相似度计算使用的是”优化后的余弦相似度“

模型训练见`ItemCFRec`类，效果评估见`ItemCFRec`类中的`precision`函数