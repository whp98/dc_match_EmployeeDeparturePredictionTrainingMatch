import pandas as pd
import numpy as np
df_train = pd.read_csv('data/train_solved_1.csv')
df_test = pd.read_csv('data/test_solved_1.csv')
df_label = pd.read_csv('data/label_1.csv')
print(df_train.shape)
# 数据归一化，效果较差
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scale_list = ['Age','DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction',
              'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'RelationshipSatisfaction', 'StockOptionLevel',
         'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsAtCompany',
              'YearsWithCurrManager']
for item in df_train.columns:
    scaler = MinMaxScaler()
    scaler.fit(df_train[item].values.reshape(-1, 1).astype('float64'))
    df_train[item] = scaler.transform(df_train[item].values.reshape(-1, 1).astype('float64'))
    scaler = MinMaxScaler()
    scaler.fit(df_test[item].values.reshape(-1, 1).astype('float64'))
    df_test[item] = scaler.transform(df_test[item].values.reshape(-1, 1).astype('float64'))

# 主成分分析，效果一般
def pca_data(df):
    from sklearn.decomposition import PCA
    pca = PCA(n_components='mle', copy=False,)
    df = pca.fit_transform(df)
    return df

# 对属性值进行one-hot编码，避免某些特征权重过大
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

feats = ["Age","BusinessTravel","Department","DistanceFromHome","Education","EducationField",
    "EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome",
    "NumCompaniesWorked","OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction",
    "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole",
    "YearsSinceLastPromotion","YearsWithCurrManager","AgeDistance","AgeEnvir","JobRoleLevel","OverPerRating"]
for (i, feat) in enumerate(feats):
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.hstack((df_train[feat].values, df_test[feat].values)).reshape(-1, 1))
    x_train = encoder.transform(df_train[feat].values.reshape(-1, 1))
    x_test = encoder.transform(df_test[feat].values.reshape(-1, 1))
    if i == 0:
        # 第一个不需要拼合到最终矩阵，因为是起点
        X_train = x_train
        X_test = x_test
    else:
        # 后面的拼合到第一矩阵，为稀疏矩阵
        X_train = sparse.hstack((X_train, x_train))
        X_test = sparse.hstack((X_test, x_test))

x_train = X_train
y_train = df_label['label']
x_test = X_test
print(X_train.shape)
# 多模型交叉验证
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import sklearn.neural_network as sk_nn
from sklearn.model_selection import cross_val_score
models = {
    'LR': LogisticRegression(solver='liblinear', penalty='l2', C=1),
    'SVM': SVC(C=1, gamma='auto'),
    'DT': DecisionTreeClassifier(),
    'RF' : RandomForestClassifier(n_estimators=100),
    'AdaBoost': AdaBoostClassifier(n_estimators=100),
    'GBDT': GradientBoostingClassifier(n_estimators=100),
    'NN': sk_nn.MLPClassifier(activation='relu',solver='adam',alpha=0.0001,learning_rate='adaptive',learning_rate_init=0.001, max_iter=1000)
}

for k, clf in models.items():
    print("the model is {}".format(k))
    scores = cross_val_score(clf, x_train, y_train, cv=10)
    print(scores)
    print("Mean accuracy is {}".format(np.mean(scores)))
    print("*" * 100)

# 网格搜索调参
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
penaltys = ['l1', 'l2']
Cs = np.arange(1, 10, 0.1)
parameters = dict(penalty=penaltys, C=Cs )
lr_penalty= LogisticRegression(solver='liblinear')
grid= GridSearchCV(lr_penalty, parameters,cv=10)
grid.fit(x_train,y_train)
grid.cv_results_
print(grid.best_score_)
print(grid.best_params_)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
print(clf)
scores = cross_val_score(clf, x_train, y_train, cv=10)
print(scores)
print("Mean accuracy is {}".format(np.mean(scores)))


clf.fit(x_train, y_train)
result = clf.predict(x_test)
file = pd.DataFrame()
file['result'] = result
file.to_csv('data/result.csv', index=False, encoding='utf-8',)

