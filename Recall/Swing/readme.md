本文件夹利用**Swing**算法编写一个推荐系统，当用户进行电影浏览时，向用户推荐和该部电影相似的电影，且若同时观看了两部电影的两个用户共同观看的电影越少，那么这两部电影的相似度越高

#### 实现思路

首先使用训练数据得到用户交互过的物品、以及物品被哪些用户交互过的信息

然后计算物品之间的相似度，以及用户对未进行评分电影的偏好分

最后选取前K个推荐给用户

#### 准备数据

数据集下载地址：https://grouplens.org/datasets/movielens/1m/

其中，`movies.dat`记录了每部电影所属类型；`ratings.dat`记录了用户对电影的评分，列分别表示UserID::MovieID::Rating::Timestamp

<img src="C:\Users\11842\AppData\Roaming\Typora\typora-user-images\image-20241001221815261.png" alt="image-20241001221815261" style="zoom:50%;" />

Swing模型使用的数据地址为`ratings_sample.dat`，为`ratings.dat`抽样后的数据（抽样了1-10的UserID，共1200条记录）

#### 模型实现

算法使用的是Swing，电影之间的相似度计算公式如下：

![img](https://pica.zhimg.com/80/v2-6f556391719e4a83ca3dfa427a2c316f_1440w.png?source=d16d100b)

模型训练见`SwingRec`类，效果评估见`SwingRec`类中的`precision`函数