import math

user_score_dict = {
    'A': {'a': 3.0, 'b':4.0, 'c':0.0, 'd':3.5, 'e':0.0},
    'B': {'a': 4.0, 'b':0.0, 'c':4.5, 'd':0.0, 'e':3.5},
    'C': {'a': 0.0, 'b':3.5, 'c':0.0, 'd':0.0, 'e':3.0},
    'D': {'a': 0.0, 'b':4.0, 'c':0.0, 'd':3.5, 'e':3.0},
}

item_user_count = dict()   # 每个物品有多少用户产生过行为
count = dict()   # 共现矩阵

for user, item in user_score_dict.items():
    for i in item.keys():
        item_user_count.setdefault(i, 0)
        if user_score_dict[user][i] > 0.0:
            item_user_count[i] += 1
        for j in item.keys():
            count.setdefault(i, {}).setdefault(j, 0)
            if (user_score_dict[user][i] > 0.0
            and user_score_dict[user][j] > 0.0
            and i != j):
                count[i][j] += 1

print('item_user_count: ', item_user_count)
print('count: ', count)

itemSim = dict()
# for i, related_items in count.items():
#     for j, cuv in related_items.items():
#         itemSim.setdefault(i, {}).setdefault(j, 0)
#         itemSim[i][j] = cuv / item_user_count[i]   # 同时喜欢物品i和j的用户数 / 喜欢物品i的用户数
#         # 该公式可以理解为：喜欢物品i的用户中，有多少比例的用户也喜欢物品j

# print('itemSim: ', itemSim)


# for i, related_items in count.items():
#     for j, cuv in related_items.items():
#         itemSim.setdefault(i, {}).setdefault(j, 0)
#         itemSim[i][j] = cuv / math.sqrt(item_user_count[i] * item_user_count[j])

for i, related_items in count.items():
    itemSim.setdefault(i, dict())
    for j, cuv in related_items.items():
        itemSim[i].setdefault(j, 0)
        itemSim[i][j] = cuv / math.sqrt(item_user_count[i] * item_user_count[j])

print('itemSim: ', itemSim)


# 预测用户对item的评分
def preUserItemScore(userA, item):
    score = 0.0
    for item1 in itemSim[item].keys():
        if item1 != item:
            score += itemSim[item][item1] * user_score_dict[userA][item1]
    return score

# 为用户推荐物品
def recommend(userA):
    # 计算userA未评分item的可能评分
    user_item_score_dict = dict()
    for item in user_score_dict[userA].keys():
        user_item_score_dict[item] = preUserItemScore(userA, item)
    return user_item_score_dict

print(recommend('C'))