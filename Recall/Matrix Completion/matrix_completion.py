import random
import time
import pandas as pd
import pickle
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

    # 存用户和电影的评分词典
    def get_user_item_rating(self, file='data/ratings.csv'):
        self.ui_scores = pd.read_csv(file)
        self.user_ids = set(self.ui_scores["UserID"].values)
        self.item_ids = set(self.ui_scores["MovieID"].values)
        self.items_dict = {user_id : self.get_one(user_id) for user_id in list(self.user_ids)}
        if not os.path.exists("data/matrix_ratings.dict"):
            self.matrix_dict = "data/matrix_ratings.dict"
            fw = open(self.matrix_dict, "wb")
            pickle.dump(self.items_dict, fw)
            fw.close()

    # 直接使用原始评分作为标签
    def get_one(self, user_id):
        pos_items = self.ui_scores[self.ui_scores['UserID'] == user_id]
        pos_dict = {row['MovieID']: row['Rating'] for _, row in pos_items.iterrows()}
        return pos_dict

class MatrixCompletionModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixCompletionModel, self).__init__()
        # 两个Embedding Layer不共享参数
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)  # [batch, emb_dim]
        item_emb = self.item_embedding(item_ids)  # [batch, emb_dim]
        ratings = torch.sum(user_emb * item_emb, dim=1)   # 内积
        return ratings   # [batch]

class MatrixCompletion:
    def __init__(self):
        self.embedding_dim = 128
        self.lamb = 0.01   # 正则化系数
        self.lr = 0.02
        self.iter_count = 5   # 迭代次数
        self.batch_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_model()

    def init_model(self):
        file_path = "data/ratings.csv"
        rating_path = "data/matrix_ratings.dict"    # {user_id: {item_id : rating}}
        self.ui_scores = pd.read_csv(file_path)
        self.user_ids = set(self.ui_scores['UserID'].values)
        self.item_ids = set(self.ui_scores['MovieID'].values)
        self.matrix_dict = pickle.load(open(rating_path, 'rb'))

        # 创建用户和物品的ID映射
        self.user_id_to_idx = {user_id : idx for idx, user_id in enumerate(self.user_ids)}
        self.item_id_to_idx = {item_id : idx for idx, item_id in enumerate(self.item_ids)}

        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        self.model = MatrixCompletionModel(self.num_users, self.num_items, self.embedding_dim).to(self.device)
        self.criterion = nn.MSELoss()    # 均方误差损失
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lamb)

    def predict(self, user_id, item_id):
        user_idx = torch.tensor(self.user_id_to_idx[user_id], dtype=torch.long).to(self.device)
        item_idx = torch.tensor(self.item_id_to_idx[item_id], dtype=torch.long).to(self.device)
        with torch.no_grad():    # 推理阶段，不需要计算梯度
            return self.model(user_idx, item_idx).item()

    def save(self):
        torch.save(self.model.state_dict(), 'model/matrix_completion.model')

    def train(self):
        for epoch in range(self.iter_count):
            print(f'开始第{epoch+1}次迭代...')
            total_loss = 0
            batch_count = 0   # 1个epoch里跑了多少次batch

            # 准备训练数据
            user_indices = []
            item_indices = []
            labels = []   # 真实评分
            for user_id, item_dict in self.matrix_dict.items():
                for item_id, label in item_dict.items():
                    user_indices.append(self.user_id_to_idx[user_id])
                    item_indices.append(self.item_id_to_idx[item_id])
                    labels.append(label)
            # 转换为tensor
            user_indices = torch.tensor(user_indices, dtype=torch.long).to(self.device)
            item_indices = torch.tensor(item_indices, dtype=torch.long).to(self.device)
            labels = torch.tensor(labels, dtype=torch.float).to(self.device)

            # 随机打乱数据
            dataset = TensorDataset(user_indices, item_indices, labels)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            for batch_users, batch_items, batch_labels in dataloader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # 前向传播
                outputs = self.model(batch_users, batch_items)
                loss = self.criterion(outputs, batch_labels)

                # 反向传播和优化
                self.optimizer.zero_grad()  # 清零梯度
                loss.backward()   # 反向传播
                self.optimizer.step()   # 更新参数

                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count
            # print('batch_count = ', batch_count)   # 1000209条评分数据 / batch_size, 并向上取整
            print(f'第{epoch+1}次迭代完成, 平均损失为: {avg_loss:.4f}')
            self.lr *= 0.9   # 学习率衰减
        self.save()

    def load(self):
        self.model.load_state_dict(torch.load('model/matrix_completion.model'))
        self.model.eval()   # 切换模型到评估模式

    def rec(self, user_id, top_n=10):
        self.load()
        user_idx = self.user_id_to_idx[user_id]
        user_item_ids = set(self.ui_scores[self.ui_scores['UserID'] == user_id]['MovieID'])  # 用户有交互的电影id
        other_item_ids = self.item_ids - user_item_ids  # 用户未评分过的电影id
        user_indices = torch.tensor([user_idx] * len(other_item_ids), dtype=torch.long).to(self.device)   # 为用户生成一个与候选物品数量相同的用户索引张量
        item_indices = torch.tensor([self.item_id_to_idx[item_id] for item_id in other_item_ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            interest_scores = self.model(user_indices, item_indices).cpu().numpy()
        candidates = sorted(zip(list(other_item_ids), interest_scores), key=lambda x: x[1], reverse=True)  # reverse=True表示降序排列
        return candidates[:top_n]

    # 模型效果评估
    def evaluate(self):
        self.load()
        users = random.sample(self.user_ids, 10)   # 从所有用户中随机选取10个用户进行评估
        user_dict = {}
        for user in users:
            user_item_ids = set(self.ui_scores[self.ui_scores['UserID'] == user]['MovieID'])   # 用户有交互的电影id
            user_indices = torch.tensor([self.user_id_to_idx[user]] * len(user_item_ids), dtype=torch.long).to(self.device)
            item_indices = torch.tensor([self.item_id_to_idx[item] for item in user_item_ids], dtype=torch.long).to(self.device)
            with torch.no_grad():
                pred_scores = self.model(user_indices, item_indices).cpu().numpy()
            # pred_scores和true_scores都是list，len为用户交互过的电影数量
            true_scores = [self.ui_scores[(self.ui_scores['UserID'] == user) & (self.ui_scores['MovieID'] == item)]['Rating'].values[0] for item in user_item_ids]
            ae = np.mean(np.abs(pred_scores - true_scores))
            user_dict[user] = ae
            print(f"UserID: {user}, AE: {ae: .4f}")
        return sum(user_dict.values()) / len(user_dict)


if __name__ == "__main__":
    dp = DataProcessing()
    # dp.process()
    # dp.get_user_item_rating()

    mc = MatrixCompletion()
    # mc.train()
    # print(mc.rec(6027, 10))
    print(mc.evaluate())