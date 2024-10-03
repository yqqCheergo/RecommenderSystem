'''
利用UserCF实现一个推荐系统
根据被推荐用户的相似用户的喜好，为被推荐用户推荐电影
'''
import json
import os.path
import random
import math

class UserCFRec:
    def __init__(self, datafile):
        self.datafile = datafile   # 原始评分数据路径文件
        self.data = self.loadData()
        self.trainData, self.testData = self.splitData(3, 47)
        self.user_sim = self.UserSimilarity()

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
        random.seed(seed)   # 生成随机数的种子
        for user, item, record in self.data:
            if random.randint(0, M) == k:  # [0, M]  1/9的概率为测试集
                test.setdefault(user, {})
                test[user][item] = record
            else:
                train.setdefault(user, {})
                train[user][item] = record
        return train, test

    def UserSimilarity(self):    # 计算user之间的相似度
        print('开始计算用户之间的相似度')
        if os.path.exists('user_sim.json'):
            print('用户相似度从文件加载...')
            userSim = json.load(open('user_sim.json', 'r'))
        else:
            item_users = dict()  # 每个item被哪些user评价过
            for u, item in self.trainData.items():
                for i in item.keys():
                    item_users.setdefault(i, set())
                    if self.trainData[u][i] > 0:
                        item_users[i].add(u)

            # 构建倒排表
            count = dict()  # 四行四列的二维矩阵，记录两个用户都评价过的物品数量
            user_item_count = dict()  # 记录用户总共评价过几个物品
            for i, user in item_users.items():
                for u in user:
                    user_item_count.setdefault(u, 0)
                    user_item_count[u] += 1
                    count.setdefault(u, {})
                    for v in user:
                        count[u].setdefault(v, 0)
                        if u == v:
                            continue
                        count[u][v] += 1 / math.log(1 + len(user))  # 惩罚热门物品

        # 构建相似度矩阵
        userSim = dict()
        for u1, u2 in count.items():
            userSim.setdefault(u1, {})
            for v, num in u2.items():  # num就是计算相似度的分子部分
                if u1 == v:
                    continue
                userSim[u1].setdefault(v, 0.0)
                userSim[u1][v] = num / math.sqrt(user_item_count[u1] * user_item_count[v])
        json.dump(userSim, open('user_sim.json', 'w'))
        return userSim

    def recommend(self, user, k=8, n_items=40):   # 为用户进行推荐
        '''
        :param user: userid
        :param k: 跟用户兴趣最相似的k个用户
        :param n_items: 为用户推荐n_items部电影，即返回结果
        :return: 字典，key为itemid，value为预测打分，共返回40部电影
        '''
        result = dict()
        have_score_items = self.trainData.get(user, {})   # key是itemid, value是rating/record
        for u, sim in sorted(self.user_sim[user].items(), key=lambda x:x[1], reverse=True)[0: k]:   # 取与用户最相似的k个用户
            for i, record in self.trainData[u].items():
                if i in have_score_items:
                    continue
                result.setdefault(i, 0)
                result[i] = sim * record
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
        return hit / precision   # 尝试使用不同的k (近邻用户数) 来调节准确率

if __name__ == '__main__':
    user_cf = UserCFRec('../Dataset/ml-1m/ratings.dat')
    print('用户1进行推荐的结果如下: {}'.format(user_cf.recommend('1')))
    print('准确率为: {}'.format(user_cf.precision()))