# encoding: utf-8

import pandas as pd
from jieba import posseg
from wordcloud import WordCloud
from snownlp import SnowNLP
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_excel('数字营销案例/产品评论数据.xlsx')
#data = df['评论']
data = df['评论'][-df.评论.isin(['此用户没有填写评论!'])]

good = ''
bad = ''
for i in data:
    s = SnowNLP(str(i))
    text = s.sentiments
    print(i, text)
    if text >= 0.5:#根据得分区分正面评价与负面评价
        good += str(i)#合并正面评价
    else:
        bad += str(i)#合并负面评价
print('正面评价：', good)
print('负面评价：', bad)

goodwords = [w for w, f in posseg.cut(good) if f[0] != 'r' and len(w) > 1 and f[0] != 'a' and f[0] != 'd']
badwords = [w for w, f in posseg.cut(bad) if f[0] != 'r' and len(w) > 1 and f[0] != 'a' and f[0] != 'd']
c1 = Counter(goodwords)
c2 = Counter(badwords)
del c1['hellip']
del c2['hellip']
print(c1)
print(c2)

w1 = WordCloud(font_path='数字营销案例/PingFang.ttc',
               background_color='white',
               scale=5,
               width=900,height=600,
               max_font_size=200,
               min_font_size=3,
               random_state=50)
w2 = WordCloud(font_path='数字营销案例/PingFang.ttc',
               background_color='white',
               scale=5,
               width=900, height=600,
               max_font_size=200,
               min_font_size=3,
               random_state=50
               )
p1 = w1.generate_from_frequencies(dict(c1))
p2 = w2.generate_from_frequencies(dict(c2))
plt.imshow(p1)
plt.axis('off')
plt.show()
plt.imshow(p2)
plt.axis('off')
plt.show()