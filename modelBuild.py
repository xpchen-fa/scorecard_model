# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 12:49:16 2018

@author: XiaSiYang
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV
from sklearn import linear_model
from sklearn.externals import joblib
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import os
import sys
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


sys.path.append(r'D:\Users\Desktop\my\learning\model')
#增加模型包导入路径
from preprocess import *

#system setting
sys.getfilesystemencoding()
#获取文件系统使用编码方式，Windows下返回'mbcs'，mac下返回'utf-8'
sys._enablelegacywindowsfsencoding()

#############################一、数据导入#######################################
#设定本次建模模型名称、文件路径

Model_Name = 'td_old_9_1_10'
code_path = 'D:\\Users\\Desktop\\my\\learning\\model\\'
os.chdir(code_path) #指定工作路径
model_path = code_path + Model_Name
if not os.path.exists(model_path):#如果没有model_path这个路径，就增加
    os.makedirs(model_path)
os.chdir(model_path) #更改成新的工作路径

#读取数据
with open(r'D:\Users\Desktop\data_9.csv',encoding='GBK') as f:
    model_data = pd.read_csv(f,header=0,index_col=0)#index_col是设置索引列
model_data.rename(columns={'label':'y'}, inplace=True)
model_data['y'] = model_data['y'].astype(np.int64)
# 处理部分数据
# model_data = model_data.rename(columns={'labels': 'y'})

# 学历改变
# model_data['educationDegree'] = model_data['educationDegree'].map(edu)

#指出因变量名称
dep = 'y'

#数据集分离
train_x,test_x,train_y,test_y = train_test_split(
        model_data.drop(dep,1),model_data[dep],test_size=0.2,random_state=20) #按照0.2的比例分割
train_y.sum() / len(train_y),test_y.sum() / len(test_y)

#训练集与测试集
input_df_train = train_x
input_df_train[dep] = train_y
input_df_train = input_df_train.reset_index(drop=True)

input_df_test = test_x
input_df_test[dep] = test_y
input_df_test = input_df_test.reset_index(drop=True)

#最终训练模型样本
input_df = input_df_train


###############################二、数据初始化#################################################
#调用数据预处理模块
dp = DataProcess(dep)

#检查字符变量,得到数值型|浮点型的数据，以及删除的非数值型和浮点型的列名
input_df,input_df_rm_str_col = dp.str_var_check(input_df)

#剔除无用的变量（非必须步骤）
#delete_col = ['apply_no','id']
# input_df = input_df.drop(delete_col,1)

#特定变量生产哑变量（非必须步骤）
#dummy_var = ['mobile_city','nfcs_education']
#input_df = dp.df_dummy(input_df,dummy_var) 

#变量基本描述信息（如饱和度，均值，IV值等）
var_desc = dp.var_status(input_df)

#保存var_desc
var_desc.to_csv(model_path+'\\'+ Model_Name +'_train_var_desc.csv',index=False)

#变量初步筛选（根据饱和度、集中度和IV阈值）
input_df_select = dp.var_filter(input_df,var_desc,0.01,0.99,0.001)

#变量初步筛选结果（变量名、变量数量等)
var_namelist = dp.get_varname(input_df_select)
n_col = len(var_namelist)


###############################三、变量可视化（可选执行模块）##################################################
#Plot_vars的输入参数依次为：输入数据集、因变量、不需要离散化的字段名、需要画图的字段（默认全面）、目标离散组数
display = Plot_vars(input_df_select,dep=dep,nodiscol=None,plotcol=None,disnums=20,file_name='varplot')

#批量字段可视化
#display.plot_vars()

#单个变量可视化
#display.plot_var('age')


################################四、变量分箱（最优分箱、基于IV，信息熵等指标最优）（常规执行模块）######################################################
#BestBin分箱
mybin = BestBin(input_df_select,method=4,group_max=3,per_limit=0.05,min_count=0.05)

#数据集批量分箱处理,同时保存分组后的数据到csv
mybin.df_bin_con_var()

best_bin_path = os.getcwd() + '\\bestbin'
#人工变量分箱调整（非必须步骤，如果有手动调整需要执行）
"""
adjust_var = ['age','bfd_com_consume_amt_12m'] #输入需要调整分箱的变量名
for i in adjust_var:
    bin_adjust = pd.read_csv(best_bin_path + '\\' + i + '.csv',index_col=0)
    mybin.ApplyMap(input_df_select,i,bin_adjust)
"""

#运行完df_bin_con_var后，原数据集df_OutlierDone分箱前后的变量并存，运行下面的程序去除原变量，保留分箱后的变量
dataout1 = pd.DataFrame(input_df_select[dep])
dataout2 = pd.DataFrame(input_df_select.iloc[:,(n_col + 1):(2 * n_col + 1)].as_matrix(),
                                             columns=[i for i in input_df_select.iloc[:,1:n_col + 1].columns])
#其中input_df_select前面是input_df_select,后面是分组后的_g
df_BinDone = pd.merge(dataout1,dataout2,left_index=True,right_index=True)

#保存分箱后的DF
df_BinDone.to_csv(model_path + '\\' + Model_Name + '_train_df_BinDone.csv')


###################################五、变量WOE，适用于回归类模型##############################################
#加载WOE_iv类，调用woe_iv_vars函数，获取x_woe_dict字典
mywoe = Woe_iv(df_BinDone,dep=dep,event=1,nodiscol=None,ivcol=None,disnums=20,X_woe_dict=None)

#WOE规则
mywoe.woe_iv_vars()

#WOE规则保存
pd.DataFrame(mywoe.X_woe_dict).to_csv(model_path + '\\' + Model_Name + '_train_woe_rule.csv')

# 调用apply_woe_replace函数，用woe后的变量代替原变量,
# 在df_BinDone的基础上替换的变量依据X_woe_dict.keys
woeapply = Woe_iv(df_BinDone,dep=dep,event=1,nodiscol=None,ivcol=None,disnums=20,X_woe_dict=mywoe.X_woe_dict)
woeapply.apply_woe_replace()
df_WoeDone = woeapply._data_new

#保留WOE后的DF
df_WoeDone.to_csv(model_path+ '\\' + Model_Name + '_train_df_WoeDone.csv')


##############################六、基于WOE后的IV值、变量间的相关性进行#################
#特征筛选（可执行模块
#对WOE后的数据集进行描述统计
var_desc_woe = dp.var_status(df_WoeDone)

#保存描述统计的结果
var_desc_woe.to_csv(model_path + '\\' + Model_Name + '_train_var_woe_desc.csv')

#对WOE后的数据集饱和度下限、集中度上限和IV阈值筛选
df_WoeDone_select = dp.var_filter(df_WoeDone,var_desc_woe,0.01,0.99,0.01)

#对WOE后的数据集、剔除相关系数高于阈值的变量（保留IV值大的）
df_model_data = dp.var_corr_delete(df_WoeDone_select,var_desc_woe,0.8)

#剔除无用变量（非必须步骤）
#delete_col = ['appl_no']
#df_model_data = df_model_data.drop(delete_col,1)


#############################七、建模数据划分训练测试集（常规执行模块）##############################################################
#对于回归类模型，用WOE后的数据训练模型
train_x,train_y = df_model_data.drop(dep,1),df_model_data[dep]


##############################八、模型构建与调试######################################################################
#人工剔除指定变量，可以根据变量含义、P值、VIF、贡献度进行剔除


# 9天逾期
x_remove = ['deposit_hg','pass_taobao','culture_sx',
            'deposit_dd','operator_nwd','purpose_ms',
            'save_jf','wd_number','user_id','deposit_cy',
            'operator','culture','industry','marriage',
            'one_phone_loan_small_loan','one_phone_many',
            'salary','seven_id_device','three_id_phone',
            'three_man_loan_big_finance','three_man_loan_property',
            'three_man_num_kind_consume','worklife','income']


'''
# 3天逾期
x_remove = ['deposit_hg','pass_taobao','culture_sx',
            'deposit_dd','operator_nwd','purpose_ms',
            'save_jf','wd_number','user_id','deposit_cy',
            'operator','one_phone_loan_kind_consume',
            'one_id_loan_kind_consume','one_phone_loan_retailers',
            'three_man_num_kind_consume','marriage','house',
            'salary']
'''


# 9天逾期旧
x_remove = ['deposit_hg','pass_taobao','culture_sx',
            'deposit_dd','operator_nwd','purpose_ms',
            'save_jf','wd_number','user_id','deposit_cy',
            'operator','one_man_loan_credit','one_man_loan_internet',
            'one_man_loan_retailers','one_phone_loan_credit',
            'one_phone_loan_net_finance','one_phone_loan_retailers',
            'purpose','seven_id_appear','three_id_loan_small_loan',
            'three_man_loan_credit']


x_update = [x for x in train_x.columns if x not in x_remove]

#自动建模过程，Lr_model_iter的输入参数依次为：自变量集、因变量集、因变量、最大P值、LASSO约束程度（越大入的变量越多，一般0.1-1之间）
lr_model_train = Lr_model()
model_output,model,model_var = lr_model_train.lr_model_iter(
        train_x[x_update],train_y,dep,p_max=0.05,alpha=1,penalty='l2')
#model_output 含 回归系数、截距、贡献度、P值、VIF值、和相关系数

##############################九、模型效果评估############################################################################
#预测概率值
Prob_Train = (model.predict_proba(train_x[model_var]))[:,1]
# predict 是预测类型，返回一列，predict_proba是返回预测类的概率，总共有m行，n列，n是预测类型的概率
Model_Y_Prob = pd.merge(pd.DataFrame(train_y).reset_index(drop=True),pd.DataFrame(Prob_Train).reset_index(drop=True),left_index=True,right_index=True)
Model_Y_Prob.columns = ['y','prob']

#保存模型效果评估
Model_Y_Prob.to_csv(model_path + '\\' + Model_Name + '_train_y_prob.csv')

#训练集模型效果评价
me = Model_evaluation(pd.Series(Model_Y_Prob['y']),pd.Series(Model_Y_Prob['prob']))
me.ks_curve(file_name='train_ks') #画KS曲线并保存
me.roc_curve(file_name='train_roc') #画ROC曲线并保存
train_sample_group,train_cut_point,train_risk_curve = me.group_risk_curve(20,file_name='train_risk_curve')
#画出分组风险倍数图（保存），返回分组、切点、风险倍数


################################十、保存整个建模过程的相关中间结果###########################################################################
#设置上线需要的文件路径
online_path = model_path + '\\online'
if not os.path.exists(online_path):
    os.makedirs(online_path) #先建立路径
os.chdir(model_path) #再默认为工作路径

#模型评测结果入模型
model_output.to_csv(model_path + '\\' + Model_Name + '_train_model_output.csv')

#入模变量顺序结果输出
lr_model_var_loc = online_path + '\\' + Model_Name + "_lr_model_var.pkl"
with open(lr_model_var_loc,'wb') as f:
    pickle.dump(model_var,f,pickle.HIGHEST_PROTOCOL)
    
#获取并保存模型结果,调用时直接 joblib.load 就行
joblib.dump(model,online_path + '\\' + Model_Name + '_lr_model.m')

#加载Woe_iv类，调用woe_iv_vars函数，获取X_woe_dict字典
model_var.insert(0,dep) #在0处插入dep
df_save_woe = df_BinDone[model_var] 
modelvar_woe = Woe_iv(df_save_woe,dep=dep,X_woe_dict=None) #调用权重计算类
modelvar_woe.woe_iv_vars()
woe_loc = online_path + '\\' + Model_Name + "_woe_rule.pkl"
modelvar_woe.woe_dict_save(modelvar_woe.X_woe_dict,woe_loc) #调用保存函数

#储存入模变量的风险曲线图
display = Plot_vars(df_BinDone[model_var],dep=dep,nodiscol=None,plotcol=None,disnums=20,file_name='model_varplot')

#批量字段可视化
display.plot_vars()


############################十一、模型测算与应用#####################################################
#导入测算数据
#with open(r'path.csv') as f:
#    CeSuan_df = pd.read_csv(f1,header=0,index_col=0)
CeSuan_df = input_df_test

#特定变量生产哑变量（非必须步骤）
#dp = DataProcess()
#dummy_var = ['INDUSTRY','EDUCATION_DEGREE']
#CeSuan_df = dp.df_dummy(dummy_var)

#获取变量名称
dp = DataProcess()
CeSuan_model = pd.read_csv(model_path + '\\' + Model_Name + '_train_model_output.csv',index_col=0)
var_in_model = [x for x in CeSuan_df.columns if x in list(CeSuan_model['var'])]
CeSuan_model_df = CeSuan_df[var_in_model]
var_namelist = dp.get_varname(CeSuan_model_df) 

#加载分箱类，对测算样本进行同样的分箱操作，并获取分箱后的数据
cesuan_bin = BestBin(CeSuan_model_df)

#应用分箱结果
best_bin_dict = {}
best_bin_path_file = os.listdir(best_bin_path) #获取文件夹下所以的文件
for i in var_in_model:
    if i + '.csv' not in best_bin_path_file:
        print('not_best_bin:'+i)
        CeSuan_model_df[i+'_g'] = CeSuan_model_df[i]
        CeSuan_model_df.loc[CeSuan_model_df[i+'_g'].isnull(),[i+'_g']] = -1
    else:
        print('best_bin:' + i)
        cesuan_bin_adjust = pd.read_csv(best_bin_path + '\\' + i + '.csv',index_col=0)
        best_bin_dict[i] = cesuan_bin_adjust
        cesuan_bin.ApplyMap(CeSuan_model_df,i,cesuan_bin_adjust) #应用分箱结果，新数据在后面
        
#应用分箱结果保存
lr_model_bestbin_loc = online_path + '\\' + Model_Name + "_bestbin,pkl"
with open(lr_model_bestbin_loc,'wb') as f:
    pickle.dump(best_bin_dict,f,pickle.HIGHEST_PROTOCOL)
    
#获取分箱后的数据
n_col = len(CeSuan_df[var_in_model].columns)  
CeSuan_BinDone = DataFrame(CeSuan_model_df.iloc[:,n_col:].as_matrix(),columns=[i for i in CeSuan_model_df.iloc[:,0:n_col].columns])

#保存结果
CeSuan_BinDone.to_csv(model_path+'\\' + Model_Name + '_cesuan_BinDone.csv')

#调用apply_woe_replace函数，用woe后的变量代替原变量（#加载woe类，对测算样本进行同样的WOE操作，注意：训练模型时没有用WOE的话就不需要这一步啦）
cesuan_woe = Woe_iv(CeSuan_BinDone)
cesuan_woe_dict = cesuan_woe.woe_dict_load(woe_loc) #读取的是WOE的字典，
woeapply = Woe_iv(CeSuan_BinDone,dep=dep,X_woe_dict=cesuan_woe_dict)
woeapply.apply_woe_replace() #应用WOE
CeSuan_df_WoeDone = woeapply._data_new

#调用保存的模型
ModelStored = joblib.load(online_path + '\\' + Model_Name + "_lr_model.m")

#测算样本上的概率值
Prob_CeSuan = pd.DataFrame(ModelStored.predict_proba(CeSuan_df_WoeDone)).iloc[:,1:] #这里取1结果就是Series,取1:就是DataFrame
Prob_CeSuan.columns = ['prob']

#加载建模样本上的概率值和真实Y
#Model_Y_Prob = pd.read_csv(model_path+'\\'+Model_Name+'_train_y_prob.csv',index_col=0)

#画出测算风险曲线，并保存结果
cesuanplot = Model_evaluation(pd.Series(CeSuan_df['y']),pd.Series(Prob_CeSuan['prob']))
cesuan_prob,cesuan_cut_points,cesuan_risk_curve = cesuanplot.group_risk_curve(20,file_name='test_risk_curve') #风险分组


#保存测算分组和切点
cesuan_prob.to_csv(model_path + '\\' + Model_Name + '_cesuan_y_prob_tw.csv',index=False)
cesuan_risk_curve.to_csv(model_path + '\\' + Model_Name + '_cesuan_risk_curve.csv')
cutpoint_loc = online_path + '\\' + Model_Name + "_cesuan_cut_points.pkl"
with open(cutpoint_loc,'wb') as f:
    pickle.dump(cesuan_cut_points,f,pickle.HIGHEST_PROTOCOL)
    
#如果测算样本也有Y,或是测试样本，可进一步验证模型外推效果
del CeSuan_df['y']
DataPredict = CeSuan_df.merge(cesuan_prob,how='inner',left_index=True,right_index=True)[['y','prob','group']]
csks = Model_evaluation(pd.Series(DataPredict['y']),pd.Series(DataPredict['prob']))
csks.ks_curve(file_name='test_ks')
csks.roc_curve(file_name='test_roc')


#整体样本交叉验证ROC和KS
CeSuan_df = model_data
dp = DataProcess()
CeSuan_model = pd.read_csv(model_path + '\\' + Model_Name + '_train_model_output.csv',index_col=0)
var_in_model = [x for x in CeSuan_df.columns if x in list(CeSuan_model['var'])]
CeSuan_model_df = CeSuan_df[var_in_model]
var_namelist = dp.get_varname(CeSuan_model_df) 
cesuan_bin = BestBin(CeSuan_model_df)
best_bin_dict = {}
best_bin_path_file = os.listdir(best_bin_path)

for i in var_in_model:
    if i + '.csv' not in best_bin_path_file:
        print('not_best_bin:'+i)
        CeSuan_model_df[i+'_g'] = CeSuan_model_df[i]
        CeSuan_model_df.loc[CeSuan_model_df[i+'_g'].isnull(),[i+'_g']] = -1
    else:
        print('best_bin:' + i)
        cesuan_bin_adjust = pd.read_csv(best_bin_path + '\\' + i + '.csv',index_col=0)
        best_bin_dict[i] = cesuan_bin_adjust
        cesuan_bin.ApplyMap(CeSuan_model_df,i,cesuan_bin_adjust) #应用分箱结果，新数据在后面
n_col = len(CeSuan_df[var_in_model].columns)  
CeSuan_BinDone = DataFrame(CeSuan_model_df.iloc[:,n_col:].as_matrix(),columns=[i for i in CeSuan_model_df.iloc[:,0:n_col].columns])
cesuan_woe = Woe_iv(CeSuan_BinDone)
cesuan_woe_dict = cesuan_woe.woe_dict_load(woe_loc) #读取的是WOE的字典，
woeapply = Woe_iv(CeSuan_BinDone,dep=dep,X_woe_dict=cesuan_woe_dict)
woeapply.apply_woe_replace() #应用WOE
CeSuan_df_WoeDone = woeapply._data_new
CeSuan_df_WoeDone['y'] = model_data['y']
CeSuan_df_WoeDone = shuffle(CeSuan_df_WoeDone).reset_index(drop=True)

ModelStored = LogisticRegression(penalty='l2',class_weight='balanced')
scores = cross_val_score(ModelStored,CeSuan_df_WoeDone.drop('y',axis=1),
                         CeSuan_df_WoeDone['y'],cv=10,scoring='roc_auc')
pd.DataFrame(data=scores).to_csv(model_path+'\\'+ Model_Name +'scores.csv',
            index=False)

ks_scores = cross_val_score(ModelStored,CeSuan_df_WoeDone.drop('y',axis=1),
                            CeSuan_df_WoeDone['y'],cv=10,scoring=ks_score)
pd.DataFrame(data=ks_scores).to_csv(model_path+'\\'+ Model_Name +'ks_scores.csv',
            index=False)







