import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
#导入数据
#训练集数据
train=pd.read_csv('data/pfm_train.csv')
#测试集数据
test=pd.read_csv('data/pfm_test.csv')

print('训练集:',train.shape)
print('测试集:',test.shape)
#合并数据集，方便同时对两个数据集进行清洗
full=pd.concat([train,test],ignore_index=True)
print('合并后的数据集:',full.shape)
#查看数据前5行
full.head()
full.describe()
#查看数据字段信息
full.info()

fullDf = pd.get_dummies(full)
print('新数据集大小：',fullDf.shape)
print('*'*50)
fullDf.info()

print('*'*50)

#相关系数：计算各个特征与标签的相关系数
print(fullDf.corr().Attrition.sort_values(ascending=False))
print('*'*50)
#删除不相关的数据

"""这里处理数据除去无用的列"""
fullDf.drop(['Over18_Y','StandardHours','EmployeeNumber','OverTime_No','MonthlyIncome'],axis=1,inplace=True)

print('*'*50)
fullDf.info()

print('*'*50)

# 属性过少，进行属性构造，不使用模型，暴力拼合
fullDf['Age'] = pd.DataFrame({'Age': fullDf['Age'].values // 5})
fullDf['AgeDistance']=pd.DataFrame({'AgeDistance':fullDf['Age'].values*100+fullDf['DistanceFromHome'].values})
fullDf['AgeEnvir']=pd.DataFrame({'AgeEnvir':fullDf['Age'].values*10+fullDf['EnvironmentSatisfaction'].values})
fullDf['JobRoleLevel']=pd.DataFrame({'JobRoleLevel':fullDf['JobSatisfaction'].values*10+fullDf['JobLevel'].values})
fullDf['OverPerRating']=pd.DataFrame({'OverPerRating':fullDf['OverTime_Yes'].values*10+fullDf['PerformanceRating'].values})
fullDf['InvolvementPerRating']=pd.DataFrame({'InvolvementPerRating':fullDf['JobInvolvement'].values*10+fullDf['PerformanceRating'].values})
fullDf['StockYear']=pd.DataFrame({'StockYear':fullDf['StockOptionLevel'].values*10+fullDf['YearsAtCompany'].values})


print('*'*50)
fullDf.info()
print('*'*50)


#原始数据集特征
source_X=fullDf[:1100]
source_X.drop('Attrition',axis=1,inplace=True)
source_X.head()

#原始数据集标签
source_y=fullDf.loc[:1099,'Attrition']
source_y.head()

#预测数据集特征
pred_X=fullDf[1100:]
pred_X.drop('Attrition',axis=1,inplace=True)
pred_X.head()

print('原始集特征：',source_X.shape)
print('原始集标签：',source_y.shape)
print('预测集特征：',pred_X.shape)


"""标准化和归一化"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#特征缩放
scaler=StandardScaler()
# source_X_scaler=scaler.fit_transform(source_X)
# pred_X_scaler=scaler.transform(pred_X)

print(source_X.shape)
print(pred_X.shape)
print(type(source_X.iloc[0]))
print(source_X.info())
#增加线性回归拟合程度
#独热编码

from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
for i in range(0,source_X.shape[1]):
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.hstack((source_X.iloc[:, i].values, pred_X.iloc[:, i].values)).reshape(-1, 1))
    x_train1 = encoder.transform(source_X.iloc[:, i].values.reshape(-1, 1))
    x_test1 = encoder.transform(pred_X.iloc[:, i].values.reshape(-1, 1))
    if i == 0:
        # 第一个不需要拼合到最终矩阵，因为是起点
        source_X_scaler = x_train1
        pred_X_scaler = x_test1
    else:
        # 后面的拼合到第一矩阵，为稀疏矩阵
        source_X_scaler = sparse.hstack((source_X_scaler, x_train1))
        pred_X_scaler = sparse.hstack((pred_X_scaler, x_test1))

print(source_X_scaler.shape)
print(pred_X_scaler.shape)

"""数据划分"""
#将原始集按照4:1分割成训练集和测试集
train_X,test_X,train_y,test_y=train_test_split(source_X_scaler,source_y,test_size=0.2,random_state=222)
print('训练数据集特征：',train_X.shape)
print('测试数据集特征：',test_X.shape)
print('训练数据标签：',train_y.shape)
print('测试数据标签：',test_y.shape)

#逻辑回归模型
from sklearn.linear_model import LogisticRegression
#网格搜索
from sklearn.model_selection import GridSearchCV
#利用GridSearch网格搜索选择最优参数
"""网格搜索"""
lg=LogisticRegression()
clf=GridSearchCV(lg,param_grid=[{'C':np.arange(0.001,0.05,0.001)}],cv=5)
#训练模型，并得到最好的参数C=0.032
#lg.fit(train_X,train_y)
clf.fit(train_X,train_y)
best_model=clf.best_estimator_
#print(clf.best_params_)
# 分类问题，score得到的是模型的准确率
print("逻辑回归:{:.5f}".format(best_model.score(test_X,test_y)))
#print("逻辑回归:{:.8f}".format(accuracy_score(test_y,lg.predict(test_X))))

"""数据生成"""
# predict预测
result=best_model.predict(pred_X_scaler)
result

#按照数据城堡规定的格式保存并上传
result=result.astype(int)
resultDf=pd.DataFrame({'result':result})

resultDf.to_csv('result1.csv',index=False)

""""






# 线性判别分析
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf_lda = LinearDiscriminantAnalysis()
clf_lda.fit(train_X, train_y)
print("线性判别分析:{:.5f}".format(accuracy_score(test_y,clf_lda.predict(test_X))))
# tt = clf_lda.predict(pred_X_scaler)
# tt = tt.astype(int)
# ttDF = pd.DataFrame({'result':tt})
# ttDF.to_csv("tt.csv",index=False)



#随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(train_X,train_y)
print("随机森林准确率:{:.5f}".format(accuracy_score(test_y,rf.predict(test_X))))
#支持向量机
from sklearn.svm import SVC
svc = SVC()
svc.fit(train_X,train_y)
print("支持向量机准确率:{:.5f}".format(accuracy_score(test_y,svc.predict(test_X))))

#KNN分类器
# 最近邻居法
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train_X, train_y)
print("KNN准确率:{:.5f}".format(accuracy_score(test_y,knn.predict(test_X))))

#决策树分类器
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(train_X, train_y)
print("决策树准确率:{:.5f}".format(accuracy_score(test_y,clf_tree.predict(test_X))))

#梯度提升树
from sklearn.ensemble import GradientBoostingClassifier
clf_gdbc = GradientBoostingClassifier(n_estimators=200)
clf_gdbc.fit(train_X, train_y)
print("GBDT准确率:{:.5f}".format(accuracy_score(test_y,clf_gdbc.predict(test_X))))

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf_adab = AdaBoostClassifier()
clf_adab.fit(train_X, train_y)
print("AdaBoost准确率:{:.5f}".format(accuracy_score(test_y,clf_adab.predict(test_X))))

### 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(train_X, train_y)
print("GNB准确率:{:.5f}".format(accuracy_score(test_y,clf_gnb.predict(test_X))))
### 二次判别分析
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf_qda = QuadraticDiscriminantAnalysis()
clf_qda.fit(train_X, train_y)
print("二次判别分析分类准确率:{:.5f}".format(accuracy_score(test_y,clf_qda.predict(test_X))))

### 支持向量机
from sklearn.svm import SVC
clf_svm = SVC(kernel='rbf', probability=True)
clf_svm.fit(train_X, train_y)
print("SVM Classifier准确率:{:.5f}".format(accuracy_score(test_y,clf_svm.predict(test_X))))

#多项式朴素贝叶斯
# ### Multinomial Naive Bayes Classifier
# from sklearn.naive_bayes import MultinomialNB
# clf_mnb = MultinomialNB(alpha=0.01)
## clf_mnb.fit(train_X, train_y)
# print("Multinomial Naive Bayes准确率:{:.5f}".format(accuracy_score(test_y,clf_mnb.predict(test_X))))


"""
