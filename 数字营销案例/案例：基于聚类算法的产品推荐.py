#!/usr/bin/python
# encoding: utf-8

"""
@author:
@contact: zhangcdnuli@163.com
@file: 案例：基于聚类算法的产品推荐.py
@time: 2020/8/15 10:55
"""

# 加载pandas库。
import pandas as pd

# 导入订单数据。
df_order = pd.read_csv("数字营销案例/data/orders.csv")
df_order = pd.read_excel("数字营销案例/data/orders.xls")
# 对数据进行探索分析。
print(df_order.head())
print(df_order.tail())
df_order.info()
# 根据业务经验先进行字段的选取，删除"买家会员名"和"买家支付宝账号"两个字段。
df_order = df_order.drop(["预约门店", "买家支付宝账号"], axis=1)
# 查看训练集中目标变量的分布信息，使用seaborn.countplot进行可视化。
# 使用shape方法查看数据集有多少个样本。
"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(7, 7))
import seaborn as sea
sea.countplot(x=df_order['宝贝种类'], ax=ax[0])
plt.show()
"""
print(df_order.shape[0])
# 过滤掉缺失值大于80%的字段。
if df_order['修改后的收货地址'].isnull().sum() > df_order.shape[0] * 0.8:
# 使用for循环将缺失值大于80%的字段删除。
    for col in df_order.columns:
        if df_order[col].isnull().sum() > df_order.shape[0] * 0.8:
            del df_order[col]
df_order.info()

# 处理后，原本46个字段，现在只剩下31个字段。
# 找到只表达一个信息的字段。
print(len(df_order.买家会员名.value_counts()))
print(len(df_order.买家实际支付积分.value_counts()))

# 这种只有一个信息的字段对于聚类而言并没有意义，使用for循环删除只表达一个信息的字段。
for col in df_order.columns:
    cate = len(df_order[col].value_counts())
    if cate <= 1:
        del df_order[col]
df_order.info()
# 观察退款金额的数值分布情况，发现0占了大部分。
print(df_order.退款金额.value_counts())

# 使用for循环将退款金额标签化，无退款为0，有退款为1。
for i in df_order.index:
    if df_order.退款金额.values[i] > 0:
        df_order.退款金额.values[i] = 1
    else:
        df_order.退款金额.values[i] = 0
print(df_order.退款金额.value_counts())

orders = df_order.loc[:, ['订单编号', '买家会员名', '买家实际支付金额', '收货地址', '宝贝种类', '宝贝总数量', '退款金额']]
print(orders.head())

# 将收货地址的省份提取出来。
print(orders.收货地址.values[0].split()[0])

# 替换掉收货地址。
orders.收货地址 = orders.收货地址.apply(lambda x: x.split()[0])
print(orders.head())
# 导入订单商品数据。
df_item = pd.read_csv("数字营销案例/Items_order.csv")
# 观察数据。
print(df_item.head())
df_item.info()
# 观察商品属性的特征，从输出结果可以看到共有439个特征。
print(df_item.商品属性.value_counts())
# 观察套餐信息的特征，从输出结果可以看到这是一个空字段。
print(df_item.套餐信息.value_counts())
# 通过观察后，确定可用的字段为'订单编号','标题','价格','购买数量'，这些字段的信息完整
items = df_item.loc[:, ['订单编号', '标题', '价格', '购买数量']]
print(items.head())
# 导入商品信息。
df_attr = pd.read_csv("数字营销案例/Items_attribute.csv", encoding='gb2312')
print(df_attr.head())
df_attr.info()
# 这个数据集包含6个字段，其中玩具类型和适用年龄存在缺失值。
# 观察玩具类型的特征，发现这个字段的特征集中在其它玩具，这个特征没有分析价值。
print(df_attr.玩具类型.value_counts())
# 观察适用年龄的特征，这个特征可以提取出玩具对应的适用儿童类型。
print(df_attr.适用年龄.value_counts())
# 提取标题和适用年龄两个字段。
attrs = df_attr.loc[:, ['标题', '适用年龄']]
print(attrs.head())
# 观察适用年龄字段的缺失值，只有4个缺失值。
print(attrs.适用年龄.isnull().value_counts())
attrs.适用年龄.fillna('missing',inplace=True)
attrs.适用年龄.isnull().value_counts()


# 自定义标签，定义2岁以下为婴儿，2岁-4岁为幼儿，5岁-7岁为学前，8岁-14岁为学生。
def addTag(x):
    tag = ''
    if '月' in x:
        tag += '婴儿|'
    if ',2岁' in x or ',3岁' in x or ',4岁' in x:
        tag += '幼儿|'
    if '5岁' in x or '6岁' in x or '7岁' in x:
        tag += '学前|'
    if '8岁' in x or '9岁' in x or '10岁' in x or '11岁' in x or '12岁' in x or '13岁' in x or '14岁' in x:
        tag += '学生|'
    if 'missing' in x:
        tag += 'missing'
    return tag


attrs['tag'] = attrs.适用年龄.astype(str).apply(addTag)
print(attrs.head())

# 提取标题和tag字段。
attrs_clean = attrs.loc[:, ['标题', 'tag']]

# 将宝贝订单表与宝贝信息表进行合并。
item_attrs = pd.merge(items, attrs_clean, on='标题', how='inner')
print(item_attrs.head())

# 删除标题字段。
del item_attrs['标题']
print(item_attrs.head())

# tag表与订单表进行合并
orders_tag = pd.merge(orders, item_attrs, on='订单编号', how='left')
print(orders_tag.head())

# 构建客户对不同年龄标签的商品的购买次数表。
test1 = orders_tag.loc[:, ['买家会员名', 'tag']]
test1['购买次数'] = 0
print(test1.head())

# 将不同年龄标签转换为字段。
test2 = test1.groupby(['买家会员名', 'tag']).count()
res_tag = test2.unstack('tag').fillna(0)
res_tag.head()

del orders_tag['tag']
del orders_tag['订单编号']
print(orders_tag.head())

res_tag.reset_index(inplace=True)
users = pd.merge(orders_tag, res_tag, on='买家会员名', how='left')
res1 = users.groupby(['买家会员名', '收货地址']).mean()
res1 = res1.fillna(0)
print(res1.head())

res1.reset_index(inplace=True)
print(res1.head())
# 使用get_dummies()方法将收货地址的省份转变成数字特征
res2 = pd.get_dummies(res1)
print(res2.head())
res2.info()

# 导入MinMaxScaler库
from sklearn.preprocessing import MinMaxScaler

data = res1.iloc[:, 2:].values
mms = MinMaxScaler()
# 数据标准化。
data_norm = mms.fit_transform(data)
print(data_norm)

# 手肘法
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# %matplotlib inline
sse = []
for k in range(1, 15):
    km = KMeans(n_clusters=k)
    km.fit(data_norm)
    sse.append(km.inertia_)
x = range(1, 15)
y = sse
plt.plot(x, y, marker='o')
plt.show()

# 轮廓系数法
from sklearn.metrics import silhouette_score

score = []
for k in range(2, 15):
    km = KMeans(n_clusters=k)
    res_km = km.fit(data_norm)
    score.append(silhouette_score(data_norm, res_km.labels_))
plt.plot(range(2, 15), score, marker='o')
#plt.imshow()
plt.show()

km = KMeans(n_clusters=8)
km.fit(data_norm)

# 把聚类的结果加入res1中。
res1['类别'] = km.labels_
print(res1.head())

cluster = res1.loc[:, ['买家会员名', '类别']]
print(cluster.head())

# 将聚类的结果写入文件。
cluster.to_csv('cluster.csv', encoding='gbk', index=False)

# 基于消费者聚类的推荐
print(orders.head())
print(df_attr.head())
orders_items = pd.merge(orders, items, on='订单编号')
print(orders_items.head())
orders_items_attrs = pd.merge(orders_items, df_attr, on='标题')
print(orders_items_attrs.head())
user_item = orders_items_attrs.loc[:, ['买家会员名', '宝贝ID']]
print(user_item.head())
user_item['购买次数'] = 0
user_item_freq = user_item.groupby(['买家会员名', '宝贝ID']).count().reset_index()
print(user_item_freq.head())
# 消费者匹配类别：消费者-物品-喜好度-消费者所属类别
print(cluster.head())
user_item_freq_cluster = pd.merge(user_item_freq, cluster, on='买家会员名')
print(user_item_freq_cluster.head())

# 对同一类别的消费者进行物品喜好度的聚合，得到同一类群中，大家对每一个物品的平均喜好度：类别-物品-平均喜好度
user_item_freq_cluster['买家会员名'] = user_item_freq_cluster['买家会员名'].apply(lambda x: str(x))
user_item_freq_cluster.info()

cluster_item_freq = user_item_freq_cluster.groupby(['类别', '宝贝ID']).mean().reset_index()
print(cluster_item_freq.head())

# 找到每一个消费者没有购买过得商品列表：消费者-未购买的商品
print(user_item_freq.head())

user_item_all = user_item_freq.pivot_table(index='买家会员名', columns='宝贝ID', values='购买次数').fillna(0)
print(user_item_all.head())

user_item_res = user_item_all.stack().reset_index()
user_item_res.rename(columns={0: '购买次数'}, inplace=True)
print(user_item_res.head())

user_item_notbuy = user_item_res[user_item_res.购买次数 == 0]
print(user_item_notbuy.购买次数.value_counts())

user_item_notbuy.drop('购买次数', axis=1, inplace=True)
print(user_item_notbuy.head())

# 消费者匹配类别，得到新的表“消费者-未购买过的商品-消费者所属类别”
user_item_notbuy_cluster = pd.merge(user_item_notbuy, cluster, on='买家会员名')
print(user_item_notbuy_cluster.head())

# 找到你没有购买过的商品，在你所属的类群中的喜好度是多少
user_item_cluster_freq = pd.merge(user_item_notbuy_cluster, cluster_item_freq, how='left', on=['宝贝ID', '类别']).fillna(0)
print(user_item_cluster_freq.head())

# 按照类别中的喜好度进行降序排序，并推荐topK个你没有购买过的，但类群中却十分流行的商品
group = user_item_cluster_freq.groupby('买家会员名')
print(group.head)


def get_topK(group, K):
    rec = group.sort_values('购买次数', ascending=False)[:K]
    return rec


topK = group.apply(get_topK, K=10)
print(topK.head())

# 删除多重索引中的第二个索引
topK.index = topK.index.droplevel(1)
print(topK.head())

topK.to_csv('rec.csv', index=False, encoding='gbk')
