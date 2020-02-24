import numpy as np
import pandas as pd


# 读取数据
train = pd.read_csv('data/pfm_train.csv')
test = pd.read_csv('data/pfm_test.csv')
print('train size:{}'.format(train.shape))  # train size:(1100, 31)
print('test size:{}'.format(test.shape))  #test size:(350, 30)
print(train)
# 查看数据集中是否含有缺失值：无缺失值
train.isnull().mean()
# EmployeeNumber为员工ID，将其删除
train.drop(['EmployeeNumber'], axis = 1, inplace = True)
print(train)
# 将Attrition（该字段为标签）移至最后一列，方便索引
Attrition = train['Attrition']
train.drop(['Attrition'], axis = 1, inplace = True)
train.insert(0, 'Attrition', Attrition)

print(train)

