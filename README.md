本仓库旨在将王树森老师《小红书推荐系统公开课》里的所有算法模型都复现一遍，工作闲暇之余抽空更新~

课程笔记：

[【上】王树森《小红书推荐系统公开课》- 课程笔记（推荐系统基础、召回、排序）](https://blog.csdn.net/qq_43629945/article/details/134109883?sharetype=blogdetail&sharerId=134109883&sharerefer=PC&sharesource=qq_43629945&spm=1011.2480.3001.8118)

[【下】王树森《小红书推荐系统公开课》- 课程笔记（特征交叉、行为序列、重排/推荐系统多样性、物品冷启动、涨指标的方法）](https://blog.csdn.net/qq_43629945/article/details/138551391?spm=1001.2014.3001.5501)

#### 召回

基于物品的协同过滤：[ItemCF](https://github.com/yqqCheergo/RecommenderSystem/tree/main/Recall/ItemCF)，[文章讲解](https://zhuanlan.zhihu.com/p/720477610)

基于用户的协同过滤：[UserCF](https://github.com/yqqCheergo/RecommenderSystem/tree/main/Recall/UserCF)，[文章讲解](https://zhuanlan.zhihu.com/p/720477610)

Swing：[Swing](https://github.com/yqqCheergo/RecommenderSystem/tree/main/Recall/Swing)，[文章讲解](https://zhuanlan.zhihu.com/p/788444439)

向量召回：

- 矩阵补充模型（及延伸出的LFM模型）：

#### 多任务学习

Shared-Bottom：[SharedBottom](https://github.com/yqqCheergo/RecommenderSystem/blob/main/MTL/SharedBottom.py)

Multi-gate Mixture-of-Experts：[MMoE](https://github.com/yqqCheergo/RecommenderSystem/blob/main/MTL/MMoE.py)

Customized Gate Control (CGC) / Progressive Layered Extraction (PLE)：[PLE](https://github.com/yqqCheergo/RecommenderSystem/blob/main/MTL/PLE.py)

Entire Space Multi-Task Model：[ESMM](https://github.com/yqqCheergo/RecommenderSystem/tree/main/MTL/ESMM)

以上4个模型对应[论文泛读](https://zhuanlan.zhihu.com/p/13078439766)，[实现代码](https://zhuanlan.zhihu.com/p/17814626067)
