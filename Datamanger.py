import matplotlib
import numpy as np
import pandas as pd

# 读取数据
train = pd.read_csv('data/pfm_train.csv')
test = pd.read_csv('data/pfm_test.csv')
print('train size:{}'.format(train.shape))  # train size:(1100, 31)
print('test size:{}'.format(test.shape))  #test size:(350, 30)
# 查看数据集中是否含有缺失值：无缺失值
train.isnull().mean()
# EmployeeNumber为员工ID，将其删除
train.drop(['EmployeeNumber'], axis = 1, inplace = True)

# 将Attrition（该字段为标签）移至最后一列，方便索引
Attrition = train['Attrition']
train.drop(['Attrition'], axis = 1, inplace = True)
train.insert(0, 'Attrition', Attrition)

from pyecharts import Bar, Line, Grid
from pyecharts import Overlap


# 通过图表分析哪些因素是主要影响员工离职的因素
def get_chatrs(train, col):
    data = train.groupby([col])['Attrition']
    data_sum = data.sum()  # 离职人数
    data_mean = data.mean()  # 离职率

    bar = Bar(col, title_pos="45%")
    bar.add('离职人数', data_sum.index, data_sum.values, mark_point=['max'],
            yaxis_formatter='人', yaxis_max=200, legend_pos="40%", legend_orient="vertical",
            legend_top="95%", bar_category_gap='25%')

    line = Line()
    line.add('离职率', data_mean.index, data_mean.values, mark_point=['max'], mark_line=['average'],
             yaxis_max=0.8)

    overlap = Overlap(width=900, height=400)
    overlap.add(bar)
    overlap.add(line, is_add_yaxis=True, yaxis_index=1)

    return overlap

##数据可视化
from pyecharts import Page

page = Page()
for col in train.columns[1:]:
    page.add(get_chatrs(train, col))
page.render('pages.html')
page
train['Attrition'].mean()

# 在分析中发现有一些字段的值是单一的,进一步验证
single_value_feature = []
for col in train.columns:
    lenght = len(train[col].unique())
    if lenght == 1:
        single_value_feature.append(col)

single_value_feature  # ['Over18', 'StandardHours']

# 删除这两个字段
train.drop(['Over18', 'StandardHours'], axis = 1, inplace = True)
train.shape  # (1100, 28)

# 对收入进行分箱
print(train['MonthlyIncome'].min())  # 1009
print(train['MonthlyIncome'].max())  # 19999
print(test['MonthlyIncome'].min())  # 1051
print(test['MonthlyIncome'].max())  # 19973

# 使用pandas的cut进行分组，分为10组
train['MonthlyIncome'] = pd.cut(train['MonthlyIncome'], bins=10)

# 将数据类型为‘object’的字段名提取出来，并使用one-hot-encode对其进行编码
col_object = []
for col in train.columns[1:]:
    if train[col].dtype == 'object':
        col_object.append(col)
col_object

train_encode = pd.get_dummies(train)

train.to_csv('trainwithoutencode.csv')
train_encode.to_csv('train.csv')

corr = train.corr()

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()
train_encode.drop(['TotalWorkingYears', 'YearsWithCurrManager'], axis = 1, inplace = True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = train_encode.iloc[:, 1:]
y = train_encode.iloc[:, 0]

# 划分训练集以及测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))  # 0.8886363636363637

pred = lr.predict(X_test)
np.mean(pred == y_test)  # 0.8863636363636364

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

#对整个train数据集的混淆矩阵
y_pred = lr.predict(X)
confmat= confusion_matrix(y_true=y,y_pred=y_pred)#输出混淆矩阵
fig,ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat,cmap=plt.cm.Blues,alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

#召回率、准确率、F1
print ('precision:%.3f' %precision_score(y_true=y,y_pred=y_pred))
print ('recall:%.3f' %recall_score(y_true=y,y_pred=y_pred))
print ('F1:%.3f' %f1_score(y_true=y,y_pred=y_pred))

# test数据集处理
test.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis = 1, inplace = True)
test_MonthlyIncome = pd.concat((pd.Series([1009, 19999]), test['MonthlyIncome']))
# 在指定位置插入与train中MonthlyIncome的max、min一致的数值，之后再删除
test['MonthlyIncome'] = pd.cut(test_MonthlyIncome, bins=10)[2:]  # 分组并去除对应的值
test_encode = pd.get_dummies(test)
test_encode.drop(['TotalWorkingYears', 'YearsWithCurrManager'], axis = 1, inplace = True)# 输出结果
sample = pd.DataFrame(lr.predict(test_encode))
sample.to_csv('sample.csv')
re = sample[0].values.tolist();
print(re)
a = ""
for i in range(0,350):
    a=a+"\n"+str(re[i])
    print(re[i])
f = open("res.txt",'a')
f.write(a)
f.close()