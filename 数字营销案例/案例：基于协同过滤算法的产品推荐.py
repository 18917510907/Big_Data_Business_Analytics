#!/usr/bin/python
# encoding: utf-8

"""
@author:
@contact: zhangcdnuli@163.com
@file: 案例：基于协同过滤算法的产品推荐.py
@time: 2020/8/15 11:44
"""

# 准备数据。
import pandas as pd

orders = pd.read_csv("数字营销案例/data/orders.csv")
items = pd.read_csv("数字营销案例/Items_order.csv")
itemProps = pd.read_csv("数字营销案例/Items_attribute.csv", encoding='gb2312')

orders_items = pd.merge(orders, items, on="订单编号")
orders_items_props = pd.merge(orders_items, itemProps, on="标题")

result = orders_items_props[["买家会员名", "宝贝ID"]]
result["购买次数"] = 0
freq = result.groupby(["买家会员名", "宝贝ID"]).count().reset_index()

freq = freq.pivot(index="买家会员名", columns="宝贝ID", values="购买次数")
freqMatrix = freq.fillna(0).values

# 推荐算法。
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def predict(similar, base="item"):
    user_cnt = freqMatrix.shape[0]
    item_cnt = freqMatrix.shape[1]
    pred = np.zeros((user_cnt, item_cnt))
    for uid in range(user_cnt):
        for iid in range(item_cnt):
            if freqMatrix[uid, iid] == 0:
                print(uid, iid)
                pred[uid, iid] = Recommendation_s(uid, iid, similar, base=base)
    return pred


def Recommendation_s(uid, iid, similar, k=10, base="item"):
    score = 0
    weight = 0
    uid_action = freqMatrix[uid, :]  # 用户uid 对所有商品的行为评分
    iid_action = freqMatrix[:, iid]  # 物品iid 得到的所有用户评分

    if base == "item":
        iid_sim = similar[iid, :]  # 商品iid 对所有商品的相似度
        sim_indexs = np.argsort(iid_sim)[-(k + 1):-1]  # 最相似的k个物品的index（除了自己）
        iid_i_mean = np.sum(iid_action) / iid_action[iid_action != 0].size
        for j in sim_indexs:
            if uid_action[j] != 0:
                iid_j_action = freqMatrix[:, j]
                iid_j_mean = np.sum(iid_j_action) / iid_j_action[iid_j_action != 0].size
                score += iid_sim[j] * (uid_action[j] - iid_j_mean)
                weight += abs(iid_sim[j])

        if weight == 0:
            return 0
        else:
            return iid_i_mean + score / float(weight)
    else:
        uid_sim = similar[uid, :]  # 用户uid 对所有用户的相似度
        sim_indexs = np.argsort(uid_sim)[-(k + 1):-1]  # 最相似的k个用户的index（除了自己）
        uid_i_mean = np.sum(uid_action) / uid_action[uid_action != 0].size
        for j in sim_indexs:
            if iid_action[j] != 0:
                uid_j_action = freqMatrix[j, :]
                uid_j_mean = np.sum(uid_j_action) / uid_j_action[uid_j_action != 0].size
                score += uid_sim[j] * (iid_action[j] - uid_j_mean)
                weight += abs(uid_sim[j])

        if weight == 0:
            return 0
        else:
            return uid_i_mean + score / float(weight)


def get_top10(group):
    return group.sort_values("推荐指数", ascending=False)[:10]


def get_recom(prediction):
    recom_df = pd.DataFrame(prediction, columns=freq.columns, index=freq.index)
    recom_df = recom_df.stack().reset_index()
    recom_df.rename(columns={0: "推荐指数"}, inplace=True)

    grouped = recom_df.groupby("买家会员名")
    top10 = grouped.apply(get_top10)

    top10 = top10.drop(["买家会员名"], axis=1)
    top10.index = top10.index.droplevel(1)
    top10.reset_index(inplace=True)
    return top10


user_similar = 1 - pairwise_distances(freqMatrix, metric="cosine")
print(user_similar)

item_similar = 1-pairwise_distances(freqMatrix.T,metric="cosine")
print (item_similar)

user_prediction = predict(user_similar, base="user")
item_prediction = predict(item_similar, base="item")

user_recom = get_recom(user_prediction)
item_recom = get_recom(item_prediction)
print(user_recom)
print(item_recom)

user_recom.to_csv("recom_top10_UBCF.csv", index=False, encoding="utf-8")
item_recom.to_csv("recom_top10_IBCF.csv", index=False, encoding="utf-8")




