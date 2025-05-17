import random
import time
import pandas as pd
import pickle
import os
import numpy as np
import math

class DataProcessing:
    def __init__(self):
        pass

    def process(self):
        print("开始转换用户数据...")
        self.process_user_data()
        print("开始转换电影数据...")
        self.process_movie_data()
        print("开始转换用户对电影的评分数据...")
        self.process_rating_data()
        print("数据转换完毕")

    def process_user_data(self, file='../Dataset/ml-1m/users.dat'):
        if not os.path.exists("data/users.csv"):
            fp = pd.read_table(file, sep="::", engine='python',
                               names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
            fp.to_csv('data/users.csv', index=False)

    def process_rating_data(self, file='../Dataset/ml-1m/ratings.dat'):
        if not os.path.exists("data/ratings.csv"):
            fp = pd.read_table(file, sep="::", engine='python',
                               names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
            fp.to_csv('data/ratings.csv', index=False)

    def process_movie_data(self, file='../Dataset/ml-1m/movies.dat'):
        if not os.path.exists("data/movies.csv"):
            fp = pd.read_table(file, sep="::", engine='python',
                               names=['MovieID', 'Title', 'Genres'])
            fp.to_csv('data/movies.csv', index=False)

    # 对用户进行有行为和无行为电影数据的标记
    def get_pos_neg_item(self, file='data/ratings.csv'):
        self.ui_scores = pd.read_csv(file)
        self.user_ids = set(self.ui_scores["UserID"].values)
        self.item_ids = set(self.ui_scores["MovieID"].values)
        self.items_dict = {user_id : self.get_one(user_id) for user_id in list(self.user_ids)}
        if not os.path.exists("data/matrix.dict"):
            self.matrix_dict = "data/matrix.dict"
            fw = open(self.matrix_dict, "wb")   # {user_id : {item_id: 0/1}}
            pickle.dump(self.items_dict, fw)
            fw.close()

    # 定义单个用户的 正向(用户有过评分的电影) 和 负向(用户无评分的电影) 数据
    def get_one(self, user_id):
        print("为用户%s准备正向和负向数据..." % user_id)
        pos_item_ids = set(self.ui_scores[self.ui_scores['UserID'] == user_id]['MovieID'])
        neg_item_ids = self.item_ids - pos_item_ids   # 差集
        neg_item_ids = list(neg_item_ids)[:len(pos_item_ids)]   # 随机选取负样本
        item_dict = {}
        for item in pos_item_ids:
            item_dict[item] = 1
        for item in neg_item_ids:
            item_dict[item] = 0
        return item_dict

class LFM:
    def __init__(self):
        self.class_count = 5   # 隐分类数量
        self.lamb = 0.01   # 正则化系数
        self.lr = 0.02   # 学习率
        self.iter_count = 5   # 迭代次数
        self.init_model()

    def init_model(self):
        file_path = "data/ratings.csv"
        pos_neg_path = "data/matrix.dict"
        self.ui_scores = pd.read_csv(file_path)
        self.user_ids = set(self.ui_scores['UserID'].values)
        self.item_ids = set(self.ui_scores['MovieID'].values)
        self.matrix_dict = pickle.load(open(pos_neg_path, 'rb'))

        # 初始化P和Q矩阵
        array_p = np.random.randn(len(self.user_ids), self.class_count)    # randn: 从标准正态分布中返回n个值
        array_q = np.random.randn(len(self.item_ids), self.class_count)
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids))   # columns指定列顺序，index指定索引
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids))

    def predict(self, user_id, item_id):
        p = np.mat(self.p.loc[user_id].values)   # 用户对每个隐类别的兴趣度 (.loc是标签索引，.iloc是位置索引)
        q = np.mat(self.q.loc[item_id].values).T   # 物品属于每个隐类别的概率
        r = (p * q).sum()    # 计算用户对物品的兴趣度预估
        # 借助sigmoid转化为是否感兴趣
        logit = 1.0 / (1 + math.exp(-r))
        return logit

    def loss(self, user_id, item_id, y, step):
        e = y - self.predict(user_id, item_id)
        # print('step: {}, user_id: {}, item_id: {}, y: {}, loss: {}'.format(step, user_id, item_id, y, e))
        return e

    # 随机梯度下降 + L2正则化防止过拟合
    def optimize(self, user_id, item_id, e):
        gradient_p = -e * self.q.loc[item_id].values
        l2_p = self.lamb * self.p.loc[user_id].values
        delta_p = self.lr * (gradient_p + l2_p)
        self.p.loc[user_id] -= delta_p

        gradient_q = -e * self.p.loc[user_id].values
        l2_q = self.lamb * self.q.loc[item_id].values
        delta_q = self.lr * (gradient_q + l2_q)
        self.q.loc[item_id] -= delta_q

    def save(self):
        f = open('model/lfm.model', 'wb')
        pickle.dump((self.p, self.q), f)   # 保存模型
        f.close()

    def train(self):
        for step in range(0, self.iter_count):
            time.sleep(30)
            for user_id, item_dict in self.matrix_dict.items():   # 返回字典中所有键值对
                print('step: {}, user_id: {}'.format(step, user_id))
                item_ids = list(item_dict.keys())
                random.shuffle(item_ids)
                for item_id in item_ids:
                    e = self.loss(user_id, item_id, item_dict[item_id], step)
                    self.optimize(user_id, item_id, e)
            self.lr *= 0.9   # 每次迭代都要降低学习率。刚开始由于离最优值较远，因此下降较快，当到达一定程度后就要减小学习率
        self.save()

    def load(self):
        f = open('model/lfm.model', 'rb')
        self.p, self.q = pickle.load(f)   # 加载模型
        f.close()

    # 计算用户未评分过的电影，并取topN返回给用户
    def rec(self, user_id, top_n=10):
        self.load()
        user_item_ids = set(self.ui_scores[self.ui_scores['UserID'] == user_id]['MovieID'])   # 用户有交互的电影id
        other_item_ids = self.item_ids - user_item_ids   # 用户未评分过的电影id
        interest_list = [self.predict(user_id, item_id) for item_id in other_item_ids]   # 0~1之间的logit
        candidates = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)   # reverse=True表示降序排列
        return candidates[:top_n]

    # 模型效果评估
    def evaluate(self):
        self.load()
        users = random.sample(self.user_ids, 10)   # 从所有用户中随机选取10个用户进行评估
        user_dict = {}
        for user in users:
            user_item_ids = set(self.ui_scores[self.ui_scores['UserID'] == user]['MovieID'])  # 用户有交互的电影id
            ae = 0.0
            for item_id in user_item_ids:
                p = np.mat(self.p.loc[user].values)
                q = np.mat(self.q.loc[item_id].values).T
                r = (p * q).sum()
                y = self.ui_scores[(self.ui_scores['UserID'] == user) & (self.ui_scores['MovieID'] == item_id)]['Rating'].values[0]
                ae += abs(r - y)   # 绝对误差AE
            user_dict[user] = ae / len(user_item_ids)
            user_dict = dict(user_dict)
            print("UserID: {}, AE: {}".format(user, user_dict[user]))
        return sum(user_dict.values()) / len(user_dict.keys())

if __name__ == "__main__":
    dp = DataProcessing()
    # dp.process()
    # dp.get_pos_neg_item()

    lfm = LFM()
    # lfm.train()
    # print(lfm.rec(6027, 10))
    print(lfm.evaluate())