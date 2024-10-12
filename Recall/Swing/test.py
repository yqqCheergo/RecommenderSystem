from itertools import combinations

user_score_dict = {
    'A': {'a': 3.0, 'b':4.0, 'c':0.0, 'd':3.5, 'e':0.0},
    'B': {'a': 4.0, 'b':0.0, 'c':4.5, 'd':0.0, 'e':3.5},
    'C': {'a': 0.0, 'b':3.5, 'c':0.0, 'd':0.0, 'e':3.0},
    'D': {'a': 0.0, 'b':4.0, 'c':0.0, 'd':3.5, 'e':3.0},
}

item_user_count = dict()   # 每个物品有多少用户产生过行为
user_item_count = dict()   # 每个用户交互过哪些物品

for user, item_rating in user_score_dict.items():
    for item, rating in item_rating.items():
        user_item_count.setdefault(user, set())
        item_user_count.setdefault(item, set())
        if rating > 0.0:
            user_item_count[user].add(item)
            item_user_count[item].add(user)
print('user_item_count:', user_item_count)
print('item_user_count:', item_user_count)

# 计算物品之间的相似度
item_pairs = list(combinations(item_user_count.keys(), 2))
print('item pairs length: {}'.format(len(item_pairs)))
item_sim_dict = dict()
cnt = 0   # 计数有多少对物品pair
alpha = 0.5
for (i, j) in item_pairs:
    cnt += 1
    # print('cnt:', cnt)
    user_pairs = list(combinations(item_user_count[i] & item_user_count[j], 2))
    result = 0.0
    for (u, v) in user_pairs:
        result += 1 / (alpha + list(user_item_count[u] & user_item_count[v]).__len__())
    item_sim_dict.setdefault(i, dict())
    item_sim_dict[i][j] = result
print('item_sim_dict:', item_sim_dict)


items = item_user_count.keys()   # 定义物品列表
# 初始化新的相似度字典
new_sim_dict = {item: {other_item: 0.0 for other_item in items} for item in items}
# 填充字典
for index1, item_i in enumerate(items):
    for index2, item_j in enumerate(items):
        if index1 != index2:
            if item_j in item_sim_dict.get(item_i, {}):
                new_sim_dict[item_i][item_j] = item_sim_dict[item_i][item_j]
                new_sim_dict[item_j][item_i] = item_sim_dict[item_i][item_j]  # 确保对称
# 设置对角线为0
for item in items:
    new_sim_dict[item][item] = 0.0

print('new_sim_dict:', new_sim_dict)


# 预测用户对item的评分
def preUserItemScore(userA, item):
    score = 0.0
    for item1 in new_sim_dict[item].keys():
        if item1 != item:
            score += new_sim_dict[item][item1] * user_score_dict[userA][item1]
    return score

# 为用户推荐物品
def recommend(userA):
    # 计算userA未评分item的可能评分
    user_item_score_dict = dict()
    for item in user_score_dict[userA].keys():
        user_item_score_dict[item] = preUserItemScore(userA, item)
    return user_item_score_dict

print('用户C的推荐结果为:', recommend('C'))