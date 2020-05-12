# -*- coding:utf8 -*-
"""
Created on 2020/3/12 17:43

@author: minc

# 接口
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor



# 数据切割
def ai_data_cut(df,xlst,ysgn,path,fgPr=False):
    x,y = df[xlst],df[ysgn]
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
    fss = path+'xtrain.csv';x_train.to_csv(fss,index=False);
    fss = path+'xtest.csv';x_test.to_csv(fss,index=False);
    fss = path+'ytrain.csv';y_train.to_csv(fss,index=False,header=True);
    fss = path+'ytest.csv';y_test.to_csv(fss,index=False,header=True);
    if fgPr:
        print(x_train.tail())
        print('-------------')
        print(y_train.tail())
        print('-------------')
        print(x_test.tail())
        print('-------------')
        print(y_test.tail())


# 读取数据
def ai_dat_rd(path,k0=1,fgPr=False):
    fss = path+'xtrain.csv';x_train=pd.read_csv(fss,index_col=False);
    fss = path+'xtest.csv';x_test=pd.read_csv(fss,index_col=False);
    fss = path+'ytrain.csv';y_train=pd.read_csv(fss,index_col=False);
    fss = path+'ytest.csv';y_test=pd.read_csv(fss,index_col=False);
    ysgn = y_train.columns[0]
    y_train[ysgn] = round(y_train[ysgn]*k0).astype(int)
    y_test[ysgn] = round(y_test[ysgn]*k0).astype(int)
    if fgPr:
        print(x_train.tail())
        print('-------------')
        print(y_train.tail())
        print('-------------')
        print(x_test.tail())
        print('-------------')
        print(y_test.tail())
    return x_train,x_test,y_train,y_test


# 效果评估函数
def ai_acc_xed(df9,ky0=5,fgDebug=False):
    '''
    df9,pandas的DatFrame格式，结果数据保存变量
    ky0,结果数据误差k值，默认5，表示5%；整数模式设置为1
    fgDebug,调试模式变量，默认是False
    '''
    ny_test,ny_pred = len(df9['y_test']),len(df9['y_pred'])
    df9['ysub'] = df9['y_test'] - df9['y_pred']
    df9['ysub2'] = np.abs(df9['ysub'])
    df9['y_test_div'] = df9['y_test']
    df9.loc[df9['y_test'] == 0, 'y_test_div']=0.00001
    df9['ysubk'] = (df9['ysub2']/df9['y_test_div'])*100
    dfk = df9[df9['ysubk']<ky0]

    dsum = len(dfk['y_pred'])

    dacc = dsum/ny_test*100

    if fgDebug:
        print(df9.head())
        y_test,y_pred = df9['y_test'],df9['y_pred']
        # 平均绝对误差
        dmae = metrics.mean_absolute_error(y_test,y_pred)
        # 均方差
        dmse = metrics.mean_squared_error(y_test,y_pred)
        # 均方根
        drmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
        print('acc-kok：{0:.2f}%，MAE:{1:.2f},MSE:{2:.2f},RMSE:{3:.2f}'.format(dacc,dmae,dmse,drmse))
    return dacc


#------------机器学习函数------------
# 线性回归
def mx_line(train_x,train_y):
    mx = LinearRegression()
    mx.fit(train_x,train_y)
    return mx


# 逻辑回归算法
def mx_log(train_x,train_y):
    mx = LogisticRegression(penalty='l2')
    mx.fit(train_x,train_y)
    return mx


# 朴素贝叶斯算法
def mx_bayes(train_x,train_y):
    mx = MultinomialNB(alpha=0.01)
    mx.fit(train_x,train_y)
    return mx


# KNN邻近算法
def mx_knn(train_x,train_y):
    mx = KNeighborsClassifier()
    mx.fit(train_x,train_y)
    return mx


# 随机森林算法
def mx_forest(train_x,train_y):
    mx = RandomForestClassifier(n_estimators=8)
    mx.fit(train_x,train_y)
    return mx


# 决策树算法
def mx_dtree(train_x,train_y):
    mx = DecisionTreeClassifier()
    mx.fit(train_x,train_y)
    return mx


# GBDT迭代决策树算法
def mx_GBDT(train_x,train_y):
    mx = GradientBoostingClassifier(n_estimators=200)
    mx.fit(train_x,train_y)
    return mx


# SVM向量机算法
def mx_SVM(train_x,train_y):
    mx = SVC(kernel='rbf',probability=True)
    mx.fit(train_x,train_y)
    return mx


# SVM-cross项链及交叉算法
def mx_svm_cross(train_x,train_y):
    mx = SVC(kernel='rbf',probability=True)
    param_grid = {'C':[1e-3,1e-2,1e-1,1,10,100,1000],'gamma':[0.001,0.0001]}
    grid_search = GridSearchCV(mx,param_grid,n_jobs=1,verbose=1)
    grid_search.fit(train_x,train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    mx = SVC(kernel='rbf',C=best_parameters['C'],gamma=best_parameters['gamma'],probability=True)
    mx.fit(train_x,train_y)
    return mx


# 神经网络算法
def mx_MLP(train_x,train_y):
    #mx = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
    mx = MLPClassifier()
    mx.fit(train_x,train_y)
    return mx


#  MLP_reg神经网络回归算法
def mx_MLP_reg(train_x,train_y):
    #mx = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
    mx = MLPRegressor()
    mx.fit(train_x,train_y)
    return mx


#------------机器学习函数------------

# 函数调用名称
mxfunLst = ['line','log','bayes','knn','forest','dtree','gbdt','svm','svmcr','mlp','mlpreg']
mxfunSgn = {'line':mx_line,
           'log':mx_log,
           'bayes':mx_bayes,
           'knn':mx_knn,
           'forest':mx_forest,
           'dtree':mx_dtree,
           'gbdt':mx_GBDT,
           'svm':mx_SVM,
           'svmcr':mx_svm_cross,
           'mlp':mx_MLP,
           'mlpreg':mx_MLP_reg}


# 统一接口函数
def mx_fun010(funSgn,x_train,x_test,y_train,y_test,yk0=5,fgInt=False,fgDebug=False):
    df9 = x_test.copy()
    mx_fun = mxfunSgn[funSgn]
    mx = mx_fun(x_train.values,y_train.values)
    y_pred = mx.predict(x_test.values)
    df9['y_test'],df9['y_pred'] = y_test,y_pred
    if fgInt:
        df9['y_predsr'] = y_pred
        df9['y_pred'] = round(df9['y_predsr']).astype(int)

    #print(df9)
    #print('123')
    dacc = ai_acc_xed(df9,yk0,fgDebug)

    if fgDebug:
        df9.to_csv('tmp/pred_result.csv')

    print('@mx:mx_sum,kok:{0:.2f}'.format(dacc))

    return dacc,df9


# 批量调用接口
def mx_funlst(funlst,x_train,x_test,y_train,y_test,yk0=5,fgInt=False):
    for funSgn in funlst:
        print('function---',funSgn)
        mx_fun010(funSgn,x_train,x_test,y_train,y_test,yk0,fgInt)


# 一体化调用接口，对多个函数的封装
def mx_fun_call(df,xlst,ysgn,funSgn,yksiz=1,yk0=5,fgInt=True,fgDebug=False):
    '''

    :param df: 数据源，pandas.DataFrame格式
    :param xlst: 参数数据集字段名
    :param ysgn: 结果数据集字段名
    :param funSgn: 调用的函数名
    :param yksiz: 结果数据缩放比例
    :param yk0: 结果数据误差值
    :param fgInt: 整数结果模式变量
    :param fgDebug: 调试模式变量
    :return:
    '''
    df[ysgn] = df[ysgn].astype(float)
    df[ysgn] = round(df[ysgn]*yksiz).astype(int)
    x,y = df[xlst],df[ysgn]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # num_train,num_feat = x_train.shape
    # num_test,num_feat = x_test.shape
    # 预测
    df9 = x_test.copy()
    mx_fun = mxfunSgn[funSgn]
    mx = mx_fun(x_train.values,y_train.values)
    y_pred = mx.predict(x_test.values)
    df9['y_test'],df9['y_pred'] = y_test,y_pred
    if fgInt:
        df9['y_predsr'] = y_pred
        df9['y_pred'] = round(df9['y_predsr']).astype(int)

    dacc = ai_acc_xed(df9,yk0,fgDebug)

    if fgDebug:
        df9.to_csv('tmp/pred_result.csv')

    return dacc,df9

# 模型保存函数
def ai_f_mxWr(ftg,funSgn,x_train,y_train):
    mx_fun = mxfunSgn[funSgn]
    mx = mx_fun(x_train.values,y_train.values)
    joblib.dump(mx,ftg)


# 模型预测函数
def mx_fun8mx(mx,x_test,y_test,yk0=5,fgInt=True,fgDebug=False):

    df9 = x_test.copy()
    y_pred = mx.predict(x_test.values)
    df9['y_test'],df9['y_pred'] = y_test,y_pred
    if fgInt:
        df9['y_predsr'] = y_pred
        df9['y_pred'] = round(df9['y_predsr']).astype(int)

    dacc = ai_acc_xed(df9, yk0, fgDebug)

    if fgDebug:
        df9.to_csv('tmp/pred_result.csv')

    return dacc, df9


# 批量存储算法函数
def ai_f_mxWrlst(ftg0,mxlst,x_train,y_train):
    for funSgn in mxlst:
        ftg = ftg0+funSgn+'.pkl'
        print(ftg)
        ai_f_mxWr(ftg, funSgn, x_train, y_train)



# 读取单个文件
def ai_f_datRd010(fsr,k0=0,fgPr=False):
    df = pd.read_csv(fsr,index_col = False)
    if k0>0:
        ysgn = df.columns[0]
        df[ysgn] = round(df[ysgn]*k0).astype(int)

    if fgPr:
        print(df.tail())
    return df

xmodel = {}

# 批量加载算法函数
def ai_f_mxRdlst(fsr0,funlst):

    for funSgn in funlst:
        fss = fsr0 + funSgn + '.pkl'
        xmodel[funSgn] = joblib.load(fss)


# 批量调用模型
def mx_funlst8mx(mxlst,x_test,y_test,yk0=5,fgInt=False):
    for msgn in mxlst:
        mx = xmodel[msgn]
        dacc,df9 = mx_fun8mx(mx,x_test, y_test,yk0,fgInt)
        print(msgn,dacc)



def mx_mul(mlst,x_test,y_test,yk0=5,fgInt=False,fgDebug=False):
    df9,xc,mxn9 = x_test.copy(),0,len(mlst)
    df9['y_test'] = y_test
    for msgn in mlst:
        xc+=1
        mx = xmodel[msgn]
        y_pred = mx.predict(x_test.values)
        if xc==1:
            df9['y_sum'] = y_pred
        else:
            df9['y_sum'] = df9['y_sum'] + y_pred

        df9['y_pred'] = y_pred

        dacc = ai_acc_xed(df9,1,fgDebug)

        xss='y_pred{0:02},kok:{1:.2f}%'.format(xc,dacc)
        ysgn = 'y_pred' + str(xc)
        df9[ysgn] = y_pred

    df9['y_pred'] = df9['y_sum']/mxn9
    dacc = ai_acc_xed(df9,yk0,fgDebug)

    if fgDebug:
        df9.to_csv('tmp/pred_result.csv')

    return dacc, df9


if __name__ == '__main__':
    fsr0 = 'tmp/ccpp_'
    x_train, x_test, y_train, y_test = ai_dat_rd(fsr0)
    funSgn = 'line'
    dacc, df9 = mx_fun010(funSgn, x_train, x_test, y_train, y_test, 5, False, True)