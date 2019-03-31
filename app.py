import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

df_power = pd.read_csv("20182019.csv")
df_power2 = pd.read_csv("20172018.csv",nrows=396)

# 把所有資料整合在 df 最後只取 "尖峰負載(MW)" "備轉容量(MW)" 來當 column
df_total = pd.concat([df_power,df_power2])
df_total.set_index('日期',inplace = True)
df = df_total.sort_index()
df = df[~df.index.duplicated()]
df = df[['尖峰負載(MW)','備轉容量(MW)']]
df_tmp = pd.read_csv('19_201_323.csv')
df_tmp.set_index('日期',inplace=True)
df_tmp.drop('備轉容量率(%)',axis=1,inplace=True)
df = pd.concat([df,df_tmp])

# 把 df 加上星期當作 column
num = 7
date_list = []
for row in df.index:
    date_list.append(num)
    num += 1
    if num == 8:
        num = 1
df['星期'] = date_list

# 把是否為假日當作一個新的 column 
holiday = [20170101,20170102,20170127,20170128,20170129,20170130,
           20170131,20170201,20170225,20170226,20170227,20170228,
           20170401,20170402,20170403,20170404,20170529,20170530,
           20171004,20171009,20171010,20180101,20180215,20180216,
           20180219,20180220,20180228,20180404,20180405,20180406,
           20180618,20180924,20181010,20181231,20190101,20190204,
           20190205,20190206,20190207,20190208,20190228,20190301]
holiday_list = []
df_dropFirst7Day = df.iloc[7:]
for row in df_dropFirst7Day.index:
    if df_dropFirst7Day.loc[row]['星期'] == 7 or df_dropFirst7Day.loc[row]['星期'] == 6:
        holiday_list.append(1)
    elif row in holiday:
        holiday_list.append(1)
    else:
        holiday_list.append(0)

df_dropFirst7Day['holiday'] = holiday_list

# 再加上前一次同個星期的尖峰負載
firstDay = int(df.iloc[0]['星期'])
firstDay = firstDay % 7
num = firstDay
scale = 0
tmp_list = []

for i in range(0,7):
    tmp_list.append([])

for row in df.index[:-7]:
    tmp_list[0].append(df['尖峰負載(MW)'].values[0 + scale])
    tmp_list[1].append(df['尖峰負載(MW)'].values[1 + scale])
    tmp_list[2].append(df['尖峰負載(MW)'].values[2 + scale])
    tmp_list[3].append(df['尖峰負載(MW)'].values[3 + scale])
    tmp_list[4].append(df['尖峰負載(MW)'].values[4 + scale])
    tmp_list[5].append(df['尖峰負載(MW)'].values[5 + scale])
    tmp_list[6].append(df['尖峰負載(MW)'].values[6 + scale])
    num += 1
    num = num % 7
    if (num == int(firstDay)) and (scale + 7) < 819 :
        scale += 7

df_dropFirst7Day['last Sun'] = tmp_list[0]
df_dropFirst7Day['last Mon'] = tmp_list[1]
df_dropFirst7Day['last Tue'] = tmp_list[2]
df_dropFirst7Day['last Wed'] = tmp_list[3]
df_dropFirst7Day['last Thu'] = tmp_list[4]
df_dropFirst7Day['last Fri'] = tmp_list[5]
df_dropFirst7Day['last Sat'] = tmp_list[6]


# 用sklearn來看看結果
df_dropFirst7Day = df_dropFirst7Day.drop(['備轉容量(MW)'],axis=1)
df_dropFirst7Day = df_dropFirst7Day.iloc[:-2]

X = df_dropFirst7Day.loc[:,df_dropFirst7Day.columns != '尖峰負載(MW)']
y = df_dropFirst7Day['尖峰負載(MW)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=False)

lm = LinearRegression()
lm.fit(X_train,y_train)
prediction = lm.predict(X_test)
plt.scatter(y_test,prediction, s = 20)
plt.xlabel('y test')
plt.ylabel('predicted y')
plt.show(block=False)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
#print(X_test)

# 運用 modle 在 4/2 - 4/8 上
df_real = pd.read_csv('realTest.csv')
predictionReal = lm.predict(df_real.drop('日期',axis=1))
date = [20190402,20190403,20190404,20190405,20190406,20190407,20190408]
result = date + list(predictionReal)
x = np.reshape(result, (2,7)).T
df_submission = pd.DataFrame(x,columns=['date','peak_load(MW)'],dtype=int)
df_submission.set_index('date')
df_submission.to_csv('submission.csv',index=False)
