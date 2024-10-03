import math

user_score_dict = {
    'A': {'a': 3.0, 'b':4.0, 'c':0.0, 'd':3.5, 'e':0.0},
    'B': {'a': 4.0, 'b':0.0, 'c':4.5, 'd':0.0, 'e':3.5},
    'C': {'a': 0.0, 'b':3.5, 'c':0.0, 'd':0.0, 'e':3.0},
    'D': {'a': 0.0, 'b':4.0, 'c':0.0, 'd':3.5, 'e':3.0},
}

# 计算用户之间的相似度，采用的是遍历每一个用户进行计算
# user_sim = dict()
# for u1 in user_score_dict.keys():
#     user_sim.setdefault(u1, {})
#     for u2 in user_score_dict.keys():
#         if u1 == u2:
#             continue
#         # 记录用户有过评分的物品集合
#         u1_set = set([key for key in user_score_dict[u1].keys() if user_score_dict[u1][key] > 0])
#         u2_set = set([key for key in user_score_dict[u2].keys() if user_score_dict[u2][key] > 0])
#         user_sim[u1][u2] = len(u1_set & u2_set) / math.sqrt(len(u1_set) * len(u2_set))


# 计算用户之间的相似度，采用优化算法时间复杂度的方法
item_users = dict()   # 每个item被哪些user评价过
for u, item in user_score_dict.items():
    for i in item.keys():
        item_users.setdefault(i, set())
        if user_score_dict[u][i] > 0:
            item_users[i].add(u)
print('每个item被哪些user评价过:', item_users)

# 构建倒排表
C = dict()   # 四行四列的二维矩阵，记录两个用户都评价过的物品数量
N = dict()   # 记录用户总共评价过几个物品
for i, user in item_users.items():
    for u in user:
        N.setdefault(u, 0)
        N[u] += 1
        C.setdefault(u, {})
        for v in user:
            C[u].setdefault(v, 0)
            if u == v:
                continue
            C[u][v] += 1 / math.log(1 + len(user))   # 惩罚热门物品
print('倒排表C:', C)
print('倒排表N:', N)

# 构建相似度矩阵
user_sim = dict()
for u1, u2 in C.items():
    user_sim.setdefault(u1, {})
    for v, num in u2.items():   # num就是计算相似度的分子部分
        if u1 == v:
            continue
        user_sim[u1].setdefault(v, 0.0)
        user_sim[u1][v] = num / math.sqrt(N[u1] * N[v])
print('用户之间的相似度为:', user_sim)


# 预测用户对item的评分
def preUserItemScore(userA, item):
    score = 0.0
    for user in user_sim[userA].keys():
        if user != userA:
            score += user_sim[userA][user] * user_score_dict[user][item]
    return score

# 为用户推荐物品
def recommend(userA):
    # 计算userA未评分item的可能评分
    user_item_score_dict = dict()
    for item in user_score_dict[userA].keys():
        if user_score_dict[userA][item] <= 0:   # 未评分item
            user_item_score_dict[item] = preUserItemScore(userA, item)
    return user_item_score_dict

print('用户C的推荐结果为:', recommend('C'))