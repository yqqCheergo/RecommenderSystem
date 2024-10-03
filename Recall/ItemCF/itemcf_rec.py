'''
利用ItemCF实现一个推荐系统
当用户进行电影浏览时，向用户推荐和该部电影相似的电影
'''
import json
import os.path
import random
import math

class ItemCFRec:
    def __init__(self, datafile, ratio):
        self.datafile = datafile   # 原始评分数据路径文件
        self.ratio = ratio   # 测试集和训练集的比例
        self.data = self.loadData()
        self.trainData, self.testData = self.splitData(3, 47)
        self.item_sim = self.ItemSimilarity()

    def loadData(self):   # 加载评分数据到data
        print('加载数据...')
        data = []
        for line in open(self.datafile):
            userid, itemid, record, timestamp = line.split('::')
            data.append((userid, itemid, int(record)))
        return data

    def splitData(self, k, seed, M=9):   # 拆分数据集为训练集和测试集
        print('训练集与测试集切分...')
        train, test = {}, {}
        random.seed(seed)
        for user, item, record in self.data:
            if random.randint(0, M) == k:  # [0, M]  1/9的概率为测试集
                test.setdefault(user, {})
                test[user][item] = record
            else:
                train.setdefault(user, {})
                train[user][item] = record
        return train, test

    def ItemSimilarity(self):    # 计算item之间的相似度
        print('开始计算物品之间的相似度')
        if os.path.exists('item_sim.json'):
            print('物品相似度从文件加载...')
            itemSim = json.load(open('item_sim.json', 'r'))
        else:
            itemSim = dict()
            item_user_count = dict()  # 每个物品有多少用户产生过行为
            count = dict()  # 共现矩阵

            for user, item in self.trainData.items():
                # print('user is {}'.format(user))
                for i in item.keys():
                    item_user_count.setdefault(i, 0)
                    if self.trainData[str(user)][i] > 0.0:
                        item_user_count[i] += 1
                    for j in item.keys():
                        count.setdefault(i, {}).setdefault(j, 0)
                        if (self.trainData[str(user)][i] > 0.0
                            and self.trainData[str(user)][j] > 0.0
                            and i != j):
                            count[i][j] += 1

            # 共现矩阵 -> 相似度矩阵
            for i, related_items in count.items():
                for j, cuv in related_items.items():
                    itemSim.setdefault(i, {}).setdefault(j, 0)
                    itemSim[i][j] = cuv / math.sqrt(item_user_count[i] * item_user_count[j])
            json.dump(itemSim, open('item_sim.json', 'w'))
        return itemSim

    def recommend(self, user, k=8, n_items=40):   # 为用户进行推荐
        '''
        :param user: userid
        :param k: 跟用户看过电影最相似的k部电影
        :param n_items: 为用户推荐n_items部电影，即返回结果
        :return: 字典，key为itemid，value为预测打分，共返回40部电影
        '''
        result = dict()
        u_items = self.trainData.get(user, {})   # key是itemid, value是rating/record
        for i, record in u_items.items():  # 用户看过的电影及评分
            for j, sim in sorted(self.item_sim[i].items(), key=lambda x:x[1], reverse=True)[0: k]:   # 按电影相似度从大到小排序，取前k个
                if j in u_items:   # 如果用户看过电影j
                    continue
                result.setdefault(j, 0)
                result[j] += record * sim
        return dict(sorted(result.items(), key=lambda x:x[1], reverse=True)[0: n_items])

    def precision(self, k=8, n_items=10):   # 效果评估，计算准确率
        print('开始计算准确率')
        hit = 0
        precision = 0
        for user in self.testData.keys():
            u_items = self.testData.get(user, {})
            result = self.recommend(user, k=k, n_items=n_items)
            for item, pre_rating in result.items():
                if item in u_items:
                    hit += 1
            precision += n_items
        return hit / precision   # 尝试使用不同的k (近邻电影数) 来调节准确率

if __name__ == '__main__':
    item_cf = ItemCFRec('../Dataset/ml-1m/ratings.dat', [1, 9])  # 测试集:训练集=1:9
    print('用户1进行推荐的结果如下: {}'.format(item_cf.recommend('1')))
    print('准确率为: {}'.format(item_cf.precision()))