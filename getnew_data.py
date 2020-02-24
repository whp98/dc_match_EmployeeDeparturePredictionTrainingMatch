import pandas as pd
import numpy as np
# 读取数据源
df_train = pd.read_csv('data/pfm_train.csv')
df_test = pd.read_csv('data/pfm_test.csv')
# 删除无关特征
def drop_no_use(df):
    df.drop(columns=['EmployeeNumber', 'StandardHours', 'Over18'], inplace=True)
drop_no_use(df_train)
drop_no_use(df_test)
# 可视化相关系数矩阵，将相关性极高的特征保留其一
import seaborn as sns
import matplotlib.pyplot as plt
corr = df_train.corr()
#%matplotlib inline
sns.heatmap(corr,xticklabels=corr.columns.values, yticklabels=corr.columns.values)

# 删除相关性极高的其中之一
df_train.drop(['MonthlyIncome', ],axis=1,inplace=True)
df_test.drop(['MonthlyIncome', ],axis=1,inplace=True)

from pyecharts import Bar,Line,Grid
from pyecharts import Overlap

def draw_chatrs(df, col):
    # 分组取离职情况
    data = df.groupby([col])["Attrition"]
    # 由于0-1标签， 求和就是离职人数，求均值就是离职人数百分比
    data_mean = data.mean()
    line = Line()
    line.add('离职率', data_mean.index, data_mean.values, mark_point=['max'], mark_line=['average'],
        yaxis_max=1.0)
    overlap = Overlap(width=900, height=400)
    overlap.add(line, is_add_yaxis=True, yaxis_index=1,)
    return overlap

from pyecharts import Page
page = Page()
for col in df_train.columns:
    if col != "Attrition":
        page.add(draw_chatrs(df_train, col))
page.render('data/analysis.html')
page

df_label = df_train['Attrition']
df_train.drop(columns=['Attrition'], inplace=True)
# 手动one-hot编码
name_list = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
for item in name_list:
    temp = list(set(df_train[item].tolist() + df_test[item].tolist()))
    print(temp)
    for i in range(len(temp)):
        df_train.loc[df_train[item]==temp[i], item] = i+1
        df_test.loc[df_test[item]==temp[i], item] = i+1

# 属性过少，进行属性构造，不使用模型，暴力拼合
df_train['Age'] = pd.DataFrame({'Age': df_train['Age'].values // 5})
df_test['Age'] = pd.DataFrame({'Age': df_test['Age'].values // 5})
df_train['AgeDistance']=pd.DataFrame({'AgeDistance':df_train['Age'].values*100+df_train['DistanceFromHome'].values})
df_test['AgeDistance']=pd.DataFrame({'AgeDistance':df_test['Age'].values*100+df_test['DistanceFromHome'].values})
df_train['AgeEnvir']=pd.DataFrame({'AgeEnvir':df_train['Age'].values*10+df_train['EnvironmentSatisfaction'].values})
df_test['AgeEnvir']=pd.DataFrame({'AgeEnvir':df_test['Age'].values*10+df_test['EnvironmentSatisfaction'].values})
df_train['JobRoleLevel']=pd.DataFrame({'JobRoleLevel':df_train['JobRole'].values*10+df_train['JobLevel'].values})
df_test['JobRoleLevel']=pd.DataFrame({'JobRoleLevel':df_test['JobRole'].values*10+df_test['JobLevel'].values})
df_train['OverPerRating']=pd.DataFrame({'OverPerRating':df_train['OverTime'].values*10+df_train['PerformanceRating'].values})
df_test['OverPerRating']=pd.DataFrame({'OverPerRating':df_test['OverTime'].values*10+df_test['PerformanceRating'].values})
df_train['InvolvementPerRating']=pd.DataFrame({'InvolvementPerRating':df_train['JobInvolvement'].values*10+df_train['PerformanceRating'].values})
df_test['InvolvementPerRating']=pd.DataFrame({'InvolvementPerRating':df_test['JobInvolvement'].values*10+df_test['PerformanceRating'].values})
df_train['StockYear']=pd.DataFrame({'StockYear':df_train['StockOptionLevel'].values*10+df_train['YearsAtCompany'].values})
df_test['StockYear']=pd.DataFrame({'StockYear':df_test['StockOptionLevel'].values*10+df_test['YearsAtCompany'].values})



# one-hot编码(pandas接口，不好用，实际上是拆分特征进行0-1编码，此题合适)
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)
print(df_train.shape)

# 数据落地，避免重复处理，占用内存开销
df_train.to_csv('data/train_solved_1.csv', index=False)
df_test.to_csv('data/test_solved_1.csv', index=False)
df_label.to_csv('data/label_1.csv', index=False, header=['label'])



