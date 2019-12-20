# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:48:27 2018

@author: XiaSiYang
"""
import numpy as np,math
from math import log
import pandas as pd,matplotlib,matplotlib.pyplot as plt
from patsy import dmatrices
from patsy import dmatrix
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf 
from copy import copy
import pickle,xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_curve,auc
import os
import re
pd.set_option('precision',4) #设置小数点后面4位，默认是6位
matplotlib.rcParams['font.sans-serif'] = ['simHei'] #指定默认字体,防止乱码
matplotlib.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
matplotlib.rcParams['font.serif'] = ['SimHei']
font = {'family':'SimHei'}
matplotlib.rc(*('font',),**font)
#import warnings
#warnings.filterwarnings('ignore')

class DataProcess:
    """
    About:数据预处理相关函数功能集
    """
    
    def __init__(self,dep='y'): #默认是y
        """
        about:初始参数
        :param dep: str,the label of y;
        """
        self.dep = dep
    
    def str_var_check(self,df):
        """
        about:移除df中的字符型变量
        :param df: DataFrame
        :return: DataFrame and 字符型变量名列表
        """
        col_type = pd.DataFrame(df.dtypes) #获取每个变量的类型,返回Series
        col_type.columns = ['type'] #增加column名称
        save_col = list(col_type.loc[(col_type['type']== 'float64')|(col_type['type'] == 'int64')].index)
        #得到是浮点型、数值型的列名
        rm_col = [x for x in df.columns if x not in save_col] #得到df不含浮点型和数值的变量名
        save_df = df[save_col] #得到浮点型和数值的变量数据
        return (save_df,rm_col)
    
    def df_dummy(self,df,col_list):
        """
        about:变量离散化
        :param df:DataFrame
        :param col_list:list 需要dummy的变量列表
        :return: DataFrame
        """
        dummy_df = pd.get_dummies(df,prefix=None,prefix_sep='_',dummy_na=False,columns=col_list,sparse=False,drop_first=True) #转化哑变量,结果要转化的在后面，不需要转化的在前面
        return dummy_df
    
    def get_varname(self,df):
        """
        about:获取特征变量名
        :param df:DataFrame
        :return DataFrame
        """
        if self.dep in df.columns:
            dfClearnVar = df.drop(self.dep,1)
        else:
            dfClearnVar = df
        var_namelist = dfClearnVar.T
        var_namelist['varname'] = var_namelist.index
        var_namelist = var_namelist['varname'].reset_index()
        var_namelist = var_namelist.drop('index',1)
        return var_namelist
    
    def calcConcentric(self,df):
        """
        about:集中度计算
        :calcuate the concentration rate
        :mydat,pd.DataFrame
        """
        sumCount = len(df.index)
        colsDf = pd.DataFrame({'tmp':['tmp',np.nan,np.nan]})
        for col in self.notempty_var_list: #此处的self.notempty_var_list在下面的var_status定义
            print(col)
            valueCountDict = {}
            colDat = df.loc[:,col]
            colValueCounts = pd.value_counts(colDat).sort_values(ascending=False) #按照colDat的值计数，再排序，按照大到小
            concentElement = colValueCounts.index[0] #得到计算最大频率的值
            valueCountDict[col] = [concentElement,colValueCounts.iloc[0],colValueCounts.iloc[0]* 1.0 / sumCount]# 得到值、出现频率、出现占总数据比
            colDf = pd.DataFrame(valueCountDict)
            colsDf = colsDf.join(colDf) #join 方法添加
        #通过循环得到所有变量的取值的频率、占比
        colsDf = (colsDf.rename(index={0:'concentricElement',1:'concentricCount',2:'concentricRate'})).drop('tmp',axis=1) #替换index的值，同时删除多余的tmp
        return colsDf
    
    def nmi(self,A,B):
        """
        about:1、介绍NMI(Normalized Mutual Information)标准化互信息，常用在聚类中，度量两个聚类结果的相近程度。是社区发现(community detection)的重要衡量指标，
        基本可以比较客观地评价出一个社区划分与标准划分之间相比的准确度。NMI的值域是0到1，越高代表划分得越准。
        """
        B = B.fillna(-1)
        total = len(A)
        A_ids = set(A)
        B_ids = set(B)
        MI = 0
        eps = 1.4e-45 #这个是1.4的负45次方
        for idA in A_ids:
            for idB in B_ids:
                idAOccur = np.where(A == idA) #返回符合条件的坐标
                idBOccur = np.where(B == idB)
                idABOccur = np.intersect1d(idAOccur,idBOccur) #返回交集并排序
                px = 1.0 * len(idAOccur[0]) / total
                py = 1.0 * len(idBOccur[0]) / total
                pxy = 1.0 * len(idABOccur) / total
                MI = MI + pxy * math.log(pxy / (px * py) + eps,2)
        Hx = 0
        for idA in A_ids:
            idAOccurCount = 1.0 * len(np.where(A == idA)[0])
            Hx = Hx - idAOccurCount / total * math.log(idAOccurCount / total + eps,2)
        Hy = 0
        for idB in B_ids:
            idBOccurCount = 1.0 * len(np.where(B == idB)[0])
            Hy = Hy - idBOccurCount / total * math.log(idBOccurCount / total + eps,2)
        MIhat = 2.0 * MI /(Hx + Hy)
        return MIhat
    
    def nmi_corr(self,df):
        """
        about:返回所有预测数据和实际的相关系数和标准化互信息
        """
        nmi_dict = {}
        corr_dict = {}
        for col in self.notempty_var_list:
            nmi_value = self.nmi(df[self.dep],df[col]) #求标准化互信息
            cor_value = np.corrcoef(df[self.dep],df[col])[(0,1)] #求相关系数，得到的结果是一个相关矩阵，取（0，1）和（1，0）是一样的
            nmi_dict[col] = nmi_value
            corr_dict[col] = cor_value
            
        nmi_df = pd.DataFrame.from_dict(nmi_dict,orient='index') # orient='index'是以字典的键为INDEX，相当于行列置换
        corr_df = pd.DataFrame.from_dict(corr_dict,orient='index')
        nmi_corr_df = nmi_df.merge(corr_df,left_index=True,right_index=True) #以左右的index连接，默认是inner
        nmi_corr_df.columns = ['nmi','corr']
        return nmi_corr_df
    
    def var_status(self,df):
        """
        about:变量描述性统计
        :param df:
        :return: Dataframe
        """
        print('正在进行变量描述统计...')
        CleanVar = df.drop(self.dep,1)
        describe = CleanVar.describe().T
        self.notempty_var_list = list(describe.loc[describe['count'] > 0].index) #得到变量个数大于0的变量名
        sample_num = int(describe['count'].max())
        describe['varname'] = describe.index
        describe.rename(columns={'count':'num','mean':'mean_v','std':'std_v','max':'max_v','min':'min_v'},inplace=True)
        describe['saturation'] = describe['num'] / sample_num
        describe = describe.drop(['25%','50%','75%'],1) #删除多余的列
        describe = describe.reset_index(drop=True) #重设index，且放弃之前的index
        describe['index'] = describe.index
        print('正在计算变量集中度...')
        Concent = self.calcConcentric(df)
        concentricRate = pd.DataFrame(Concent.T['concentricRate'])
        concentricRate['index'] = concentricRate.index
        print('正在计算变量IV值...')
        mywoeiv = Woe_iv(df,dep=self.dep,event=1,nodiscol=None,ivcol=None,disnums=20,X_woe_dict=None)
        mywoeiv.woe_iv_vars()
        iv = mywoeiv.iv_dict
        iv = pd.DataFrame(pd.Series(iv,name='iv'))
        iv['index'] = iv.index
        var_des = pd.merge(describe,concentricRate,how='inner',left_on=['varname'],right_on=['index'])
        var_des = pd.merge(var_des,iv,how='inner',left_on =['varname'],right_on=['index'])
        var_des = var_des[['varname','num','mean_v','std_v','min_v','max_v','saturation','concentricRate','iv']]
        print('描述统计完毕...')
        return var_des
    
    def var_filter(self,df,varstatus,min_saturation=0.01,max_concentricRate=0.98,min_iv=0.01):
        """
        about:变量初筛
        :param df:
        :param varstatus: dataframe.var_status函数的输出结果
        :param min_saturation:float 变量筛选值饱和度下限
        :param max_concentricRate: folat 变量筛选值集中度上限
        :param min_iv: float 变量筛选之IV值下限
        :return: DataFrame 这里nan会变成false
        """
        var_selected = list(varstatus['varname'].loc[(varstatus['saturation'] >= min_saturation) & (varstatus['concentricRate'] <= max_concentricRate)
        & (varstatus['iv'] >= min_iv)]) #依据要求得到符合条件的变量名
        var_selected.insert(0,self.dep) #插入y
        df_selected = df[var_selected]
        return df_selected #返回筛选后的数据
    
    def var_corr_delete(self,df_WoeDone_select,var_desc_woe,corr_limit=0.95):
        """
        about:剔除相关系数高的变量
        :param df_WoeDone_select: 变量初删后返回的值
        :param var_desc_woe: 变量描述统计后返回的值
        :param corr_limit:
        :return:
        """
        deleted_vars = []
        high_IV_var = list((df_WoeDone_select.drop(self.dep,axis=1)).columns)
        for i in high_IV_var:
            if i in deleted_vars:
                continue
            for j in high_IV_var:
                if not i == j:
                    if not j in deleted_vars:
                        roh = np.corrcoef(df_WoeDone_select[i],df_WoeDone_select[j])[(0,1)] #比较每两个之间的相关系数，最后删除IV值较小的数
                        if abs(roh) > corr_limit:
                            x1_IV = var_desc_woe.iv.loc[var_desc_woe.varname == i].values[0] #这个就是IV值
                            y1_IV = var_desc_woe.iv.loc[var_desc_woe.varname == j].values[0]
                            if x1_IV > y1_IV:
                                deleted_vars.append(j)
                                print('变量' + i + '和' + '变量' + j + '高度相关,相关系数达到' + str(abs(roh)) + ',已删除' + j)
                            else:
                                deleted_vars.append(i)
                                print('变量' + i + '和' + '变量' + j + '高度相关,相关系数达到' + str(abs(roh)) + ',已删除' + i)
        df_corr_select = df_WoeDone_select.drop(deleted_vars,axis=1) #删除选中的变量
        print('已对相关系数达到' + str(corr_limit) + '以上的变量进行筛选，剔除的变量列表如下' + str(deleted_vars))
        return df_corr_select

    
class Plot_vars:
    """
    about:风险曲线图
    cut_points_bring:单变量,针对连续变量离散化，目标是均分xx组，但当某一值重复过高时会适当调整，最后产出的是分割点，不包含首尾
    dis_group:单变量，离散化连续变量，并生产一个groupby数据:
    nodis_group:不需要离散化的变量，生产一个groupby数据:
    plot_var:单变量绘图
    plot_vars:多变量绘图
    """
    
    def __init__(self,mydat,dep='y',nodiscol= None,plotcol=None,disnums=5,file_name=None):
        """
        abount:变量风险图
        :param model_name:
        :param mydat:DataFrame,包含X,y的数据集
        :param dep: str,the label of y
        :param nodiscol: list,defalut None,当这个变量有数据时，会默认这里的变量不离散，且只画nodis_group,
        其余的变量都需要离散化，且只画dis_group.当这个变量为空时，系统回去计算各变量的不同数值的数量，若小于15，则认为不需要离散，直接丢到nodiscol中
        :param disnums: int,连续变量需要离散的组数
        """
        self.mydat = mydat
        self.dep = dep
        self.plotcol = plotcol #这个是制定多个变量，批量跑多个变量
        self.nodiscol = nodiscol
        self.disnums = disnums
        self.file_path = os.getcwd()
        self.file_name = file_name
        self.col_cut_points = {}
        self.col_notnull_count = {}
        for i in self.mydat.columns:
            if i != self.dep:
                self.col_cut_points[i] = []
        for i in self.mydat.columns:
            if i != self.dep:
                col_notnull = len(self.mydat[i][pd.notnull(self.mydat[i])].index)
                self.col_notnull_count[i] = col_notnull
        
        if self.nodiscol is None:
            nodiscol_tmp = []
            for i in self.mydat.columns:
                if i != self.dep:
                    col_cat_num = len(set(self.mydat[i][pd.notnull(self.mydat[i])]))
                    if col_cat_num < 5: #非空的数据分类小于5个
                        nodiscol_tmp.append(i)
            if len(nodiscol_tmp) > 0:
                self.nodiscol = nodiscol_tmp
        if self.file_name is not None:
            self.New_Path = self.file_path + '\\' + self.file_name + '\\'
            if not os.path.exists(self.New_Path):
                os.makedirs(self.New_Path) #增加新的文件夹
        
    def cut_point_bring(self,col_order,col):
        """
        about:分割函数
        :param col_order:DataFrame,非null得数据集，包含y，按变量值顺序排列
        :param col:str 变量名
        :return:
        """
        PCount = len(col_order.index)
        min_group_num = self.col_notnull_count[col] / self.disnums #特定变量的非null数据量除以默认分组5组
        disnums = int(PCount / min_group_num)   #数据集的数量除以最小分组的数量
        if PCount /self.col_notnull_count[col] >= 1 /self.disnums:         
            if disnums > 0 :
                n_cut = int(PCount /disnums)
                cut_point = col_order[col].iloc[n_cut - 1]
                for i in col_order[col].iloc[n_cut:]:
                    if i == cut_point:
                        n_cut += 1
                    else:
                        self.col_cut_points[col].append(cut_point) #得到切点值
                        break #这里为了解决分割点的值多个相当，因此一直分到不相等，才退出
                self.cut_point_bring(col_order[n_cut:],col) #这里是递归函数,用剩下的数据继续跑
    
    def dis_group(self,col):
        """
        abount:连续性变量分组
        :param col:str,变量名称
        :return:
        """
        dis_col_data_notnull = self.mydat.loc[(pd.notnull(self.mydat[col]),[self.dep,col])]
        Oreder_P = dis_col_data_notnull.sort_values(by=[col],ascending=True)
        self.cut_point_bring(Oreder_P,col)
        dis_col_cuts = []
        dis_col_cuts.append(dis_col_data_notnull[col].min()) #得到变量的最小值
        dis_col_cuts.extend(self.col_cut_points[col]) #加入变量的切点值
        dis_col_data = self.mydat.loc[:,[self.dep,col]]
        dis_col_data['group'] = np.nan
        for i in range(len(dis_col_cuts) - 1): #这里开始依据分组的切点，改变group的值
            if i == 0:
                dis_col_data.loc[dis_col_data[col] <= dis_col_cuts[i+1],['group']] = i
            elif i == len(dis_col_cuts) - 2:
                dis_col_data.loc[dis_col_data[col] > dis_col_cuts[i],['group']] = i
            else:
                dis_col_data.loc[(dis_col_data[col] > dis_col_cuts[i]) & (dis_col_data[col] <= dis_col_cuts[i+1]),['group']] = i
        dis_col_data[col] = dis_col_data['group']
        dis_col_bins = []
        dis_col_bins.append('nan')
        dis_col_bins.extend(['(%s,%s]' % (dis_col_cuts[i],dis_col_cuts[i+1]) for i in range(len(dis_col_cuts) - 1)])
        dis_col = dis_col_data.fillna(-1) 
        col_group = (dis_col.groupby([col],as_index=False))[self.dep].agg({'count':'count','bad_num':'sum'}) #按照切点分组，按dep计数
        avg_risk = self.mydat[self.dep].sum() /self.mydat[self.dep].count() #平均风险度
        col_group['per'] = col_group['count'] / col_group['count'].sum() #占比
        col_group['bad_per'] = col_group['bad_num'] / col_group['count'] #坏客户占比
        col_group['risk_times'] = col_group['bad_per'] / avg_risk #风险倍数
        if -1 in list(col_group[col]):
            col_group['bins'] = dis_col_bins
        else:
            col_group['bins'] = dis_col_bins[1:]
        col_group[col] = col_group[col].astype(np.float) #转换数据类型
        col_group = col_group.sort_values([col],ascending=True)
        col_group = pd.DataFrame(col_group,columns=[col,'bins','per','bad_per','risk_times'])
        return col_group  

    def nodis_group(self,col):
        """
        aount:离散型变量分组，主要处理null值
        :param col:str 变量名称
        :return:
        """
        nodis_col_data = self.mydat.loc[:,[self.dep,col]]
        is_na = pd.isnull(nodis_col_data[col]).sum() > 0 #判断是否有null
        col_group = (nodis_col_data.groupby([col],as_index=False))[self.dep].agg({'count':'count','bad_num':'sum'})
        col_group = pd.DataFrame(col_group,columns=[col,'bad_num','count'])
        if is_na:
            y_na_count = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].count()
            y_na_sum = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].sum()
            col_group.loc[len(col_group.index),:] = [-1,y_na_sum,y_na_count] #添加到最后一行
        avg_risk = self.mydat[self.dep].sum() /self.mydat[self.dep].count()
        col_group['per'] = col_group['count'] / col_group['count'].sum()
        col_group['bad_per'] = col_group['bad_num'] /col_group['count']
        col_group['risk_times'] = col_group['bad_per'] / avg_risk #风险倍数
        if is_na:
            bins = col_group[col][:len(col_group.index) - 1]
            bins[len(col_group.index)] = 'nan'
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index) - 1))
            col_labels.append(-1)
            col_group[col] = col_labels
        else:
            bins = col_group[col]
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index)))
            col_group[col] = col_labels
        col_group[col] = col_group[col].astype(np.float)
        col_group = col_group.sort_values([col],ascending=True)
        col_group = pd.DataFrame(col_group,columns=[col,'bins','per','bad_per','risk_times'])
        return col_group
    
    def plot_var(self,col):
        """
        about:单变量泡泡图
        :param col: str 变量名称
        :return:
        """
        if self.nodiscol is not None:
            if col in self.nodiscol:
                col_group = self.nodis_group(col)
            else:
                col_group = self.dis_group(col)
        fix,ax1 = plt.subplots()
        ax1.bar(x=list(col_group[col]),height=col_group['per'],width=0.6,align='center') #bar是柱形图
        for a,b in zip(col_group[col],col_group['per']):
            ax1.text(a,b+0.005,'%.4f' %b,ha='center',va='bottom',fontsize=8) #给柱形图加上文字标示
            
        ax1.set_xlabel(col)
        ax1.set_ylabel('percent',fontsize=12)
        ax1.set_xlim([-2,max(col_group[col])+2]) #设置刻度长度
        ax1.set_ylim([0,max(col_group['per'])+0.3])
        ax1.grid(False) #不显示网格
        ax2 = ax1.twinx() #在原有的图像上添加坐标轴，类似双轴
        ax2.plot(list(col_group[col]),list(col_group['risk_times']),'-ro',color='red')
        for a,b in zip(col_group[col],col_group['risk_times']):
            ax2.text(a,b+0.05,'%.4f' % b,ha='center',va='bottom',fontsize=7)
            
        ax2.set_ylabel('risktimes',fontsize=12)
        ax2.set_ylim([0,max(col_group['risk_times']) + 0.5])
        ax2.grid(False)
        plt.title(col)
        the_table = plt.table(cellText=col_group.round(4).values,colWidths=[0.12]*len(col_group.columns), #round(4)保留4位小数
        rowLabels=col_group.index,colLabels=col_group.columns,loc=1) #loc=1显示在右对齐.loc=2显示在左对齐.colWidths是每列表格的长度占比，5列，每列0.2，就占满
        the_table.auto_set_font_size(False) #自动设置字体大小
        the_table.set_fontsize(10) #设置表格的大小
        if self.file_path is not None:
            plt.savefig(self.New_Path+col+'.pdf',dpi=600)
        # plt.show()
        
    def plot_vars(self):
        """
        about:批量跑多个变量，如果未制定多列或者单个变量，将会跑所有的变量
        """
        if self.plotcol is not None: #这个是制定多个变量，批量跑多个变量
            for col in self.plotcol:
                self.plot_var(col)
        else:
            cols = self.mydat.columns
            for col in cols:
                print(col)
                if col != self.dep:
                    self.plot_var(col)

                    
class Woe_iv:
    """
    about: woe iv 类计算函数
    check_target_binary:检查是否是二分类问题
    target_count:计算好坏样本数目
    cut points_bring:单变量，针对连续变量离散化，目标是均分xx组，但当某一数值重复过高时会适当调整，最后产出的是分割点，不包含首尾
    dis_group: 单变量，离散化连续变量，并生产一个groupby数据
    nodis_group:不需要离散化的变量，生产一个groupby数据
    woe_iv_var:单变量，计算各段的woe值和iv
    woe_iv_vars:多变量，计算多变量的woe和iv
    apply_woe_replace:将数据集中的分段替换成对应的woe值
    一般应大于0.02，默认选IV大于0.1的变量进模型，但具体要结合实际。如果IV大于0.5，就是过预测（over-predicting）变量
    AUC在 0.5～0.7时有较低准确性， 0.7～0.8时有一定准确性, 0.8~0.9则高，AUC在0.9以上时有非常高准确性。AUC=0.5时，说明诊断方法完全不起作用，无诊断价值
    psi 判断：index <= 0.1，无差异；0.1< index <= 0.25，需进一步判断；0.25 <= index，有显著位移，模型需调整。
    """
    def __init__(self,mydat,dep='y',event=1,nodiscol=None,ivcol=None,disnums=20,X_woe_dict=None):
        """
        about:初始化参数设置
        :param mydat:DataFrame 输入的数据集，包含y
        :param dep: str,the label of y
        :param event: int y中bad的标签
        :param nodiscol: list,defalut
        None,不需要离散的变量名，当这个变量有数据时，会默认这里的变量不离散，且只跑nodis_group,其余的变量都需要离散化，且只跑
        dis_group。当这个变量为空时，系统会去计算各变量的不同数值的数量，若小于15，则认为不需要离散，直接丢到nodiscol中
        :param ivcol: list 需要计算woe,iv的变量名，该变量不为None时，只跑这些变量，否则跑全体变量
        :param disnums: int 连续变量离散化的组数
        :param X_woe_dict: dict 每个变量每段的woe值，这个变量主要是为了将来数据集中的分段替换成对应的woe值
        即输入的数据集已经超过离散化分段处理，只需要woe化而已
        """
        self.mydat = mydat
        self.event = event
        self.nodiscol = nodiscol
        self.ivcol = ivcol
        self.disnums = disnums
        self._WOE_MIN = -20
        self._WOE_MAX = 20
        self.dep = dep
        self.col_cut_points = {}
        self.col_notnull_count = {}
        self._data_new = self.mydat.copy(deep=True)
        self.X_woe_dict = X_woe_dict
        self.iv_dict = None
        for i in self.mydat.columns:
            if i != self.dep:
                self.col_cut_points[i] = []
                
        for i in self.mydat.columns:
            if i != self.dep:
                col_notnull = len(self.mydat[i][pd.notnull(self.mydat[i])].index)
                self.col_notnull_count[i] = col_notnull
                
        if self.nodiscol is None:
            nodiscol_tmp = []
            for i in self.mydat.columns:
                if i != self.dep:
                    col_cat_num = len(set(self.mydat[i][pd.notnull(self.mydat[i])]))
                    if col_cat_num < 20:
                        nodiscol_tmp.append(i)
            
            if len(nodiscol_tmp) > 0:
                self.nodiscol = nodiscol_tmp
                
    def check_target_binary(self,y):
        """
        about: 检测因变量是否为二元变量
        :param y: the target variable,series type
        :return:
        """
        y_type = type_of_target(y) #检测数据是不是二分类数据
        if y_type not in ('binary',):
            raise ValueError('Label tyoe must be binary!!!')
        
    def target_count(self,y):
        """
        about：计算Y值得数量
        :param y: the target variable,series type
        :return: 0,1的数量 
        """
        y_count = y.value_counts()
        if self.event not in y_count.index:
            event_count = 0
        else:
            event_count = y_count[self.event]
        non_event_count = len(y) - event_count
        return (event_count,non_event_count) #返回好坏客户的数量
    
    def cut_points_bring(self,col_order,col):
        """
        about:分割函数
        :param col_order: DataFrame 非null的数据集，包含y，按变量值顺序排列
        :param col: str 变量名
        :return 
        """
        PCount = len(col_order.index)
        min_group_num = self.col_notnull_count[col] /self.disnums
        disnums = int(PCount/min_group_num)
        if PCount / self.col_notnull_count[col] >= 1 / self.disnums:
            if disnums >0 :
                n_cut = int(PCount / disnums)
                cut_point = col_order[col].iloc[n_cut-1]
                for i in col_order[col].iloc[n_cut:]:
                    if i == cut_point:
                        n_cut += 1
                    else:
                        self.col_cut_points[col].append(cut_point)
                        break
                self.cut_points_bring(col_order[n_cut:],col)
    
    def dis_group(self,col):
        """
        abount:连续型变量分组
        :param col:str 变量名称
        :return:
        """
        dis_col_data_notnull = self.mydat.loc[(pd.notnull(self.mydat[col]),[self.dep,col])]
        Oreder_P = dis_col_data_notnull.sort_values(by=[col],ascending=True)
        self.cut_points_bring(Oreder_P,col)
        dis_col_cuts = []
        dis_col_cuts.append(dis_col_data_notnull[col].min()) #得到变量的最小值
        dis_col_cuts.extend(self.col_cut_points[col]) #加入变量的切点值
        dis_col_cuts.append(dis_col_data_notnull[col].max()) #得到变量的最大值
        dis_col_data = self.mydat.loc[:,[self.dep,col]]
        dis_col_data['group'] = np.nan
        for i in range(len(dis_col_cuts) - 1): #这里开始依据分组的切点，改变group的值
            if i == 0:
                dis_col_data.loc[dis_col_data[col] <= dis_col_cuts[i+1],['group']] = i
            elif i == len(dis_col_cuts) - 2:
                dis_col_data.loc[dis_col_data[col] > dis_col_cuts[i],['group']] = i
            else:
                dis_col_data.loc[(dis_col_data[col] > dis_col_cuts[i]) & (dis_col_data[col] <= dis_col_cuts[i+1]),['group']] = i
        dis_col_data[col] = dis_col_data['group']
        dis_col_bins = []
        dis_col_bins.append('nan')
        dis_col_bins.extend(['(%s,%s]' % (dis_col_cuts[i],dis_col_cuts[i+1]) for i in range(len(dis_col_cuts) - 1)])
        dis_col = dis_col_data.fillna(-1) 
        col_group = (dis_col.groupby([col],as_index=False))[self.dep].agg({'count':'count','bad_num':'sum'}) #按照切点分组，按dep计数       
        col_group['per'] = col_group['count'] / col_group['count'].sum() #占比
        col_group['good_num'] = col_group['count'] - col_group['bad_num'] #好客户数量
        if -1 in list(col_group[col]):
            col_group['bins'] = dis_col_bins
        else:
            col_group['bins'] = dis_col_bins[1:]
        for i in range(len(dis_col_cuts) - 1):
            if i == 0:
                self._data_new.loc[self.mydat[col] <= dis_col_cuts[i + 1],[col]] = dis_col_bins[i + 1]
            else:
                if i == len(dis_col_cuts) - 2:
                    self._data_new.loc[self.mydat[col] > dis_col_cuts[i],[col]] = dis_col_bins[i + 1]
                else:
                    self._data_new.loc[(self.mydat[col] > dis_col_cuts[i]) & (self.mydat[col] <= dis_col_cuts[i + 1]),[col]] = dis_col_bins[i + 1]
        self._data_new[col].fillna(value='nan',inplace=True)
        return col_group
       
    def nodis_group(self,col):
        """
        about:离散型变量分组
        :param col: str 变量名称
        :return:
        """
        nodis_col_data = self.mydat.loc[:,[self.dep,col]]
        is_na = pd.isnull(nodis_col_data[col]).sum() > 0 #判断是否有null
        col_group = (nodis_col_data.groupby([col],as_index=False))[self.dep].agg({'count':'count','bad_num':'sum'})
        col_group = pd.DataFrame(col_group,columns=[col,'bad_num','count'])
        if is_na:
            y_na_count = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].count()
            y_na_sum = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].sum()
            col_group.loc[len(col_group.index),:] = [-1,y_na_sum,y_na_count] #添加到最后一行
        col_group['per'] = col_group['count'] / col_group['count'].sum()
        col_group['good_num'] = col_group['count'] - col_group['bad_num']
        if is_na:
            bins = col_group[col][:len(col_group.index) - 1]
            bins.loc[len(bins.index)] = 'nan'
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index) - 1))
            col_labels.append(-1)
            col_group[col] = col_labels
        else:
            bins = col_group[col]
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index)))
            col_group[col] = col_labels
        return col_group
    
    def woe_iv_var(self,col,adj=1):
        """
        about:单变量woe分布及iv值计算
        :param col: str 变量名称
        :param adj: float 分母为0时的异常调整
        :return:
        """
        self.check_target_binary(self.mydat[self.dep]) #检查目标值是否时二分类
        event_count,non_event_count = self.target_count(self.mydat[self.dep]) #计算坏、好客户数量
        if self.nodiscol is not None:
            if col in self.nodiscol:
                col_group = self.nodis_group(col)
            else:
                col_group = self.dis_group(col)
        x_woe_dict = {}
        iv = 0
        for cat in col_group['bins']:
            cat_event_count = col_group.loc[(col_group.loc[:,'bins'] == cat,'bad_num')].iloc[0] #分组中坏客户的数量
            cat_non_event_count = col_group.loc[(col_group.loc[:,'bins'] == cat,'good_num')].iloc[0] #分组中好客户的数量
            rate_event = cat_event_count * 1.0 / event_count #本组的坏客户/总的坏客户
            rate_non_event = cat_non_event_count * 1.0 / non_event_count #本周的好客户/总的好客户
            if rate_non_event == 0:#这个是让分子、分母都不为0
                woe1 = math.log((cat_event_count * 1.0 + adj) / event_count / ((cat_non_event_count * 1.0 + adj) / non_event_count)) #本组坏客户占比除以好客户占比
            else:
                if rate_event == 0:
                    woe1 = math.log((cat_event_count * 1.0 +adj) / event_count / ((cat_non_event_count * 1.0 + adj) / non_event_count))
                else:
                    woe1 = math.log(rate_event / rate_non_event)
            x_woe_dict[cat] = woe1
            iv += abs((rate_event - rate_non_event) * woe1)        
        return (x_woe_dict,iv)
    
    def woe_iv_vars(self,adj=1):
        """
        about:多变量woe分布及IV值计算
        :param adj:folat,分母为0时的异常调整
        :return:
        """
        X_woe_dict = {}
        iv_dict = {}
        if self.ivcol is not None: #ivcol这里也是制定批量处理的
            for col in self.ivcol:
                print(col)
                x_woe_dict,iv = self.woe_iv_var(col,adj)
                X_woe_dict[col] = x_woe_dict
                iv_dict[col] = iv
        else:
            for col in self.mydat.columns: #这里时全部处理
                print(col)
                if col != self.dep:
                    x_woe_dict,iv = self.woe_iv_var(col,adj)
                    X_woe_dict[col] = x_woe_dict
                    iv_dict[col] = iv
                    
        
        self.X_woe_dict = X_woe_dict #储存IV和WOE值
        self.iv_dict = iv_dict
        
    def apply_woe_replace(self):
        """
        about:变量woe值替换
        :return：
        """
        for col in self.X_woe_dict.keys():#这个格式是字典中带字典
            for binn in self.X_woe_dict[col].keys():
                self._data_new.loc[(self._data_new.loc[:,col] == binn,col)] = self.X_woe_dict[col][binn]
                
        self._data_new = self._data_new.astype(np.float64) #得到_data_new是所有分箱后的woe值
        
    def woe_dict_save(self,obj,woe_loc):
        """
        about:woe 字典保存
        :param obj:
        :param woe_loc:保存路径及文件名
        :return:
        """
        with open(woe_loc,'wb') as f:
            pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL) #将obj对象保存在f中，HIGHEST_PROTOCOL是整型，最高协议版本
         
    def woe_dict_load(self,woe_loc):
        """
        about: woe 字典读取
        :param woe_loc:读取路径及文件名
        :return:
        """
        with open(woe_loc,'rb') as f:
            return pickle.load(f)
        

class BestBin:
    """
    about:分箱类函数
    """
    def __init__(self,data_in,dep='y',method=4,group_max=3,per_limit=0.05,min_count=0.05):
        """
        about:参数设置
        :param data_in:DataFrame 包含X,y的数据集
        :param dep:str,the label of y;
        :param method:int,defaut 4,分箱指标选取：1 基尼系数，2 信息熵，3 皮尔逊卡方统计量、4IV值
        :param group_max: int,defalut 3, 最大分箱组数
        :param per_limit: float,小切割范围占比
        """
        self.data_in = data_in
        self.dep = dep
        if self.dep in self.data_in.columns:
            self.CleanVar = self.data_in.drop(self.dep,1) #删除Y
        else:
            self.CleanVar = self.data_in
        self.method = method
        self.group_max = group_max
        self.per_limit = per_limit
        self.min_count = min_count
        self.file_path = os.getcwd()
        self.bin_num = int(1 / per_limit)
        self.nrows = data_in.shape[0]
        self.col_cut_points = {}
        self.col_notnull_count = {}
        for i in self.CleanVar.columns:
            self.col_cut_points[i] = []
            col_notnull = len(self.data_in[i][pd.notnull(self.data_in[i])].index)
            self.col_notnull_count[i] = col_notnull
        
    def cut_points_bring(self,col_order,col):
        """
        about:分割函数
        :param col_order:DataFrame,非null的数据集，包含y，按变量值顺序排列
        :param col: str 变量名
        :return:
        """
        PCount = len(col_order.index)  #得到数据长度
        min_group_num = self.col_notnull_count[col] * self.per_limit #得到最小分组数量，这个是不变的
        disnums = int(PCount/min_group_num) #得到分组数，每次都会变化
        if PCount / self.col_notnull_count[col] >= self.per_limit * 2: #剩余的数量/总的数量 大于等于最小分割的2呗，留下最后一份
            if disnums > 0 : #最后分组要大于0
                n_cut = int(PCount / disnums) 
                cut_point = col_order[col].iloc[n_cut-1]
                for i in col_order[col].iloc[n_cut:]:
                    if i == cut_point:
                        n_cut += 1
                    else:
                        self.col_cut_points[col].append(cut_point)
                        break
                self.cut_points_bring(col_order[n_cut:],col) #递归调用自己
                
    def groupbycount(self,input_data,col):
        """
        abount:分组
        :param col:str 变量名称
        :param input_data:DataFrame 输入数据
        :return:
        """
        dis_col_data_notnull = self.data_in.loc[(pd.notnull(self.data_in[col]),[self.dep,col])] #得到变量函和Y的数据
        Oreder_P = dis_col_data_notnull.sort_values(by=[col],ascending=True) #将所有的数据排序
        self.cut_points_bring(Oreder_P,col) #调用分割函数,按照
        dis_col_cuts = []
        dis_col_cuts.append(dis_col_data_notnull[col].min()) #得到变量的最小值
        dis_col_cuts.extend(self.col_cut_points[col]) #加入变量的切点值
        dis_col_cuts.append(dis_col_data_notnull[col].max()) #加入变量最大值
        input_data['group'] = input_data[col]
        for i in range(len(dis_col_cuts) - 1): #这里开始依据分组的切点，改变group的值
            if i == 0:  # 第一组
                input_data.loc[input_data[col] <= dis_col_cuts[i+1],['group']] = i + 1
            elif i == len(dis_col_cuts) - 2:  # 最后一组
                input_data.loc[input_data[col] > dis_col_cuts[i],['group']] = i + 1
            else:
                input_data.loc[(input_data[col] > dis_col_cuts[i]) & (input_data[col] <= dis_col_cuts[i+1]),['group']] = i + 1

        return (input_data,dis_col_cuts) #返回分组值，分组切点

    def bin_con_var(self,data_in_stable,indep_var,group_now):
        """
        about:分箱
        :param data_in_stable:输入数据集
        :param indep_var:自变量
        :param group_now:分箱切点映射表,也是那个最大分组数
        :return:
        """
        data_in = data_in_stable.copy() #复制
        data_in,bin_cut = self.groupbycount(data_in,indep_var) #调用分组函数
        Mbins = len(bin_cut) - 1
        Temp_DS = data_in.loc[:,[self.dep,indep_var,'group']]
        Temp_DS['group'].fillna(0, inplace=True) #将null值用0来填充-未实施此步骤
        temp_blimits = pd.DataFrame({'Bin_LowerLimit':[],'Bin_UpperLimit':[],'group':[]}) 
        for i in range(Mbins): #得到变量分组后的上下限值
            temp_blimits.loc[(i,['Bin_LowerLimit'])] = bin_cut[i]
            temp_blimits.loc[(i,['Bin_UpperLimit'])] = bin_cut[i+1]
            temp_blimits.loc[(i,['group'])] = i + 1
            
        grouped = Temp_DS.loc[:,[self.dep,'group']].groupby(['group']) #分组
        g1 = grouped.sum().rename(columns={self.dep:'y_sum'})
        g2 = grouped.count().rename(columns={self.dep:'count'})
        g3 = pd.merge(g1,g2,left_index=True,right_index=True)
        g3['good_sum'] = g3['count'] - g3['y_sum']
        g3['group'] = g3.index
        g3['PDVI'] = g3.index
		g3.rename(columns={'group': 'id'},inplace=True)   # 此处是后来有问题增加的
        temp_cont = pd.merge(g3,temp_blimits,how='left',on='group') #得到分组统计的数据
        for i in range(Mbins):
            mx = temp_cont.loc[(temp_cont['group'] == i+1,'group')].values
            if mx:
                Ni1 = temp_cont['good_sum'].loc[temp_cont['group'] == i+1].values[0]
                Ni2 = temp_cont['y_sum'].loc[temp_cont['group'] == i+1].values[0]
                count = temp_cont['count'].loc[temp_cont['group'] == i+1].values[0]
                bin_lower = temp_cont['Bin_LowerLimit'].loc[temp_cont['group'] == i+1].values[0]
                bin_upper = temp_cont['Bin_UpperLimit'].loc[temp_cont['group'] == i+1].values[0]
                if i == Mbins - 1: #如果i是最后一位
                    i1 = temp_cont['group'].loc[temp_cont['group'] < Mbins].values.max() #最后一位就取前面中最大的
                else:
                    i1 = temp_cont['group'].loc[temp_cont['group'] > i+1].values.min() #其他的就取后面中最小的
                if Ni1 == 0 or Ni2 == 0 or count == 0: 
                    #如果有一组好客户、或者坏客户、或者总数为0，就把本组的所有人数加到后一组中，如果是最后一组，就加到上一组中，同时改正分组的阈值，再删除当组的数据
                    temp_cont['good_sum'].loc[temp_cont['group'] == i1] = temp_cont['good_sum'].loc[temp_cont['group'] == i1].values[0] + Ni1
                    temp_cont['y_sum'].loc[temp_cont['group'] == i1] = temp_cont['y_sum'].loc[temp_cont['group'] == i1].values[0] + Ni2
                    temp_cont['count'].loc[temp_cont['group'] == i1] = temp_cont['count'].loc[temp_cont['group'] == i1].values[0] + count
                    if i < Mbins -1:
                        temp_cont['Bin_LowerLimit'].loc[temp_cont['group'] == i1] = bin_lower
                    else:
                        temp_cont['Bin_UpperLimit'].loc[temp_cont['group'] == i1] = bin_upper
                    delete_indexs = list(temp_cont.loc[temp_cont['group'] == i+1].index)
                    temp_cont = temp_cont.drop(delete_indexs)
        
        temp_cont['new_index'] = (temp_cont.reset_index(drop=True)).index + 1
        temp_cont['var'] = temp_cont['group']
        temp_cont['group'] = 1
        Nbins = 1
        while Nbins < group_now: #这会一直分组
            Temp_Splits = self.CandSplits(temp_cont)
            temp_cont = Temp_Splits
            Nbins = Nbins + 1
        
        temp_cont.rename(columns={'var':'OldBin'},inplace=True)
        temp_Map1 = temp_cont.drop(['good_sum','PDVI','new_index'],axis=1)
        temp_Map1 = temp_Map1.sort_values(by=['group'])
        min_group = temp_Map1['group'].min()
        max_group = temp_Map1['group'].max()
        lmin = temp_Map1['Bin_LowerLimit'].min()
        notnull = temp_Map1.loc[temp_Map1['Bin_LowerLimit'] > lmin - 10]
        var_map = pd.DataFrame({'group':[],'LowerLimit':[],'UpperLimit':[],'count':[],'risk':[]})
        for i in range(min_group,max_group+1):
            ll = notnull['Bin_LowerLimit'].loc[notnull['group'] == i].min()
            uu = notnull['Bin_UpperLimit'].loc[notnull['group'] == i].max()
            con = notnull['count'].loc[notnull['group'] == i].sum()
            yy = notnull['y_sum'].loc[notnull['group'] == i].sum()
            if con > 0:
                risk = yy * 1.0 / (con + 0.0001)
                var_map = var_map.append({'group':i,'LowerLimit':ll,'UpperLimit':uu,'count':con,'risk':risk},ignore_index=True)
        
        null_group = temp_Map1['group'].loc[temp_Map1['Bin_LowerLimit'].isnull()]
        if null_group.any():
            temp_Map_null = temp_Map1.loc[temp_Map1['Bin_LowerLimit'].isnull()]
            ll = temp_Map_null['Bin_LowerLimit'].min()
            uu = temp_Map_null['Bin_UpperLimit'].max()
            con = temp_Map_null['count'].sum()
            yy = temp_Map_null['y_sum'].sum()
            i = temp_Map_null['group'].max()
            risk = yy * 1.0 / con
            var_map = var_map.append({'group':i,'LowerLimit':ll,'UpperLimit':uu,'count':con,'risk':risk},ignore_index=True)
        var_map = var_map.sort_values(by=['LowerLimit','UpperLimit'])
        var_map['newgroup'] = var_map.reset_index().index + 1
        var_map = var_map.reset_index()
        var_map = var_map.drop('index',1)
        if null_group.any():
            ng = var_map['group'].loc[var_map['LowerLimit'].isnull()].max()
            notnull = var_map.loc[var_map['LowerLimit'].notnull()]
            cng = var_map['group'].loc[var_map['group'] == ng].count()
            if cng > 1:
                var_map['newgroup'].loc[var_map['LowerLimit'].isnull()] = notnull['newgroup'].loc[notnull['group'] == ng].max()
                var_map['group'] = var_map['newgroup']
            else:
                var_map['group'] = var_map['newgroup']
                var_map['group'].loc[var_map['LowerLimit'].isnull()] = 0
        else:
            var_map['group'] = var_map['newgroup']
        var_map = var_map.drop('newgroup',1)
        var_map['per'] = var_map['count'] * 1.0 / var_map['count'].sum()
        return var_map
    
    def df_bin_con_var(self):
        """
        about:分箱
        :return
        """
        ToBin = self.CleanVar.T #删除后的Y转置
        ToBin['varname'] = ToBin.index #得到变量名
        for i in ToBin['varname']:
            print(i)
            varnum = self.data_in.loc[:,[self.dep,i]].groupby([i]) #以变量分组,groupby 自动删除null计数
            vcount = len(varnum.count().index) #得到分组的个数
            sum_vcount = sum(varnum.count().index.tolist()) #这里判断是否是2分类，且只有0和1
            if vcount <= 2:
                self.data_in.loc[:,i + '_g'] = self.data_in[i]
                self.data_in.loc[self.data_in[i + '_g'].isnull(),[i + '_g']] = -1 #将_g变量中所有的null变为-1
            else:
                var_group_map = self.bin_con_var(self.data_in,i,self.group_max) #开始分组,得到分组值
                mincount = var_group_map['per'].min() #得到占比最小
                group_now = len(var_group_map.group.unique()) #得到分组数
                while mincount < self.min_count and group_now > 3:#判断最小分组占比小于设定的比率，且现在分组大于2组，则一直分下去
                    group_now = group_now - 1
                    var_group_map = self.bin_con_var(self.data_in,i,group_now)
                    mincount = var_group_map['count'].min() / var_group_map['count'].sum()
                    
                self.ApplyMap(self.data_in,i,var_group_map) #
                print(var_group_map)
                if self.file_path is not None:
                    self.New_Path = self.file_path + '\\bestbin\\'
                    if not os.path.exists(self.New_Path):
                        os.makedirs(self.New_Path)
                var_group_map.to_csv(self.New_Path + i + '.csv')
            
    def CandSplits(self,BinDS):
        """
        about:Generate all candidate splits from currentBins and select the best new bins,first we sort the dataset OldBins by PDVI and Bin
        :param BinDS
        :return
        """
        BinDS.sort_values(by=['group','PDVI'],inplace=True) #先排序
        Bmax = BinDS['group'].values.max() #取分组最大
        m = []
        names = locals() #返回当前的所有局部变量
        for i in range(Bmax): #当两组之后的，就把数据集合给分割成不同的部分
            names['x%s' % i] = BinDS.loc[BinDS['group'] == i + 1]
            m.append(BinDS['group'].loc[BinDS['group'] == i + 1].count()) #不同部分的组别个数，大于1下面就分割，否则不分割
            
        temp_allVals = pd.DataFrame({'BinToSplit':[],'DatasetName':[],'Value':[]})
        for i in range(Bmax):  # 对每个组别
            if m[i] > 1:  # 如果组别的个数大于1
                testx = self.BestSplit(names['x%s' % i],i) #调用函数
                names['temp_trysplit%s' % i] = testx
                names['temp_trysplit%s' % i].loc[names['temp_trysplit%s' % i]['Split'] == 1,['group']] = Bmax +1
                d_indexs = list(BinDS.loc[BinDS['group'] == i + 1].index)
                names['temp_main%s' % i] = BinDS.drop(d_indexs)
                names['temp_main%s' % i] = pd.concat([names['temp_main%s' % i], names['temp_trysplit%s' % i]])
                Value = self.GValue(names['temp_main%s' % i]) #这里是求按照上述分组后的值
                temp_allVals = temp_allVals.append({'BinToSplit':i,'DatasetName':'temp_main%s' % i,'Value':Value},ignore_index=True)
                #这里是对分了两组后的数据，再此分组，加入其的Value值，下面来判断哪个是最大的，那下次返回的数据就是最大的一个的分组，这样1分为2，2分为3，不会到4
                
        ifsplit = temp_allVals['BinToSplit'].max()
        if ifsplit >= 0:
            temp_allVals = temp_allVals.sort_values(by=['Value'],ascending=False)
            bin_i = int(temp_allVals['BinToSplit'][0])
            NewBins = names['temp_main%s' % bin_i].drop('Split',1)
        else:
            NewBins = BinDS
        return NewBins
    
    def BestSplit(self,BinDs,BinNo):
        """
        about:
        :param BinDs
        :param BinNo
        :return:
        """
        mb = BinDs['group'].loc[BinDs['group'] == BinNo +1].count()
        BestValue = 0
        BestI = 1
        for i in range(mb - 1):#重复循环，计算出最大的Value值
            Value = self.CalcMerit(BinDs,i+1) #再次调用函数,以i+1为分割点，计算相应两部分的值
            if BestValue < Value:
                BestValue = Value
                BestI = i + 1
        BinDs.loc[:,'Split'] = 0
        BinDs.loc[BinDs['new_index'] <= BestI,['Split']] = 1 #可以找到最佳的二分点
        BinDs = BinDs.drop('new_index',1)
        BinDs = BinDs.sort_values(by=['Split','PDVI'],ascending=True)
        BinDs['testindex'] = (BinDs.reset_index(drop=True)).index
        BinDs['new_index'] = BinDs['testindex'] + 1
        BinDs.loc[BinDs['Split'] == 1,['new_index']] = BinDs['new_index'].loc[BinDs['Split'] == 1] - BinDs['Split'].loc[BinDs['Split'] == 0].count()
        NewBinDs = BinDs.drop('testindex',1)
        return NewBinDs
    
    def CalcMerit(self,BinDs,ix): 
        """
        about:
        :param BinDs
        :param ix
        :return
        """
        n_11 = BinDs['good_sum'].loc[BinDs['new_index'] <= ix].sum()
        n_21 = BinDs['good_sum'].loc[BinDs['new_index'] > ix].sum()
        n_12 = BinDs['y_sum'].loc[BinDs['new_index'] <= ix].sum()
        n_22 = BinDs['y_sum'].loc[BinDs['new_index'] > ix].sum()
        n_1s = BinDs['count'].loc[BinDs['new_index'] <= ix].sum()
        n_2s = BinDs['count'].loc[BinDs['new_index'] > ix].sum()
        n_s1 = BinDs['good_sum'].sum()
        n_s2 = BinDs['y_sum'].sum()
        if self.method == 1: #基尼系数
            N = n_1s + n_2s
            G1 = 1 - (n_11 * n_11 + n_12 * n_12) / (n_1s * n_1s) #1-好人占比平方-坏人占比平方
            G2 = 1 - (n_21 * n_21 + n_22 * n_22) / (n_2s * n_2s) # 同上
            G = 1 - (n_s1 * n_s1 + n_s2 * n_s2) / (N * N) # 好坏占总人数的G值
            Gr = 1 - (n_1s * G1 + n_2s * G2) / (N * G)  # 这里将原来分之后的/分之前的G值，再被1减去，反应改变后的变化
            M_value = Gr
            
        if self.method ==2: #信息熵
            N = n_1s + n_2s
            E1 = -(n_11 / n_1s * log(n_11 /n_1s) + n_12 / n_1s * log(n_12 / n_1s)) / log(2) #这里用换底公式
            E2 = -(n_21 / n_2s * log(n_21 / n_2s) + n_22 / n_2s * log(n_22 / n_2s)) / log(2)
            E = -(n_s1 / N * log(n_s1 / N) + n_s2 /N * log(n_s2 / N)) / log(2)
            Er = 1 - (n_1s * E1 + n_2s * E2) / (N * E) #此处为了方便统一，都是用分组前和分组后的比值来体现
            M_value = Er
        
        if self.method == 3: #皮尔逊卡方统计量
            N = n_1s + n_2s
            m_11 = n_1s * n_s1 / N #计算好客户预期值，在区间1
            m_12 = n_1s * n_s2 / N #计算坏客户预期值，在区间1
            m_21 = n_2s * n_s1 / N #计算好客户预期值，在区间2
            m_22 = n_2s * n_s2 / N #计算坏客户预期值，在区间2
            X2 = (n_11 - m_11) * (n_11 - m_11) / m_11 + (n_12 - m_12) * (n_12 - m_12) / m_12 + (n_21 - m_21) * (n_21 - m_21) / \
                m_21 + (n_22 - m_22) * (n_22 - m_22) / m_22 #实际减去预期**2之和
            M_value = X2
            
        if self.method == 4: #IV值
            IV = (n_11 / n_s1 - n_12 / n_s2) * log(n_11 * n_s2 / (n_12 * n_s1 + 0.01) + 1e-05) + (n_21 / n_s1 - n_22 / \
                 n_s2) * log(n_21 * n_s2 / (n_22 * n_s1 + 0.01) + 1e-05) #两个的IV值相加
            M_value = IV
            
        return M_value
    
    def GValue(self,BinDs):
        """
        about:
        :param BinDs
        :return
        """
        R = BinDs['group'].max()
        N = BinDs['count'].sum()
        nnames = locals()
        for i in range(R):
            nnames['N_1%s' % i] = BinDs['good_sum'].loc[BinDs['group'] == i + 1].sum()
            nnames['N_2%s' % i] = BinDs['y_sum'].loc[BinDs['group'] == i + 1].sum()
            nnames['N_s%s' % i] = BinDs['count'].loc[BinDs['group'] == i + 1].sum()
            N_s_1 = BinDs['good_sum'].sum()
            N_s_2 = BinDs['y_sum'].sum()
        
        if self.method == 1:
            aa = locals()
            for i in range(R):
                aa['G_%s' % i] = 0
                aa['G_%s' % i] = aa['G_%s' % i] + nnames['N_1%s' % i] * nnames['N_1%s' % i] + nnames['N_2%s' % i] * nnames['N_2%s' % i] *\
                nnames['N_2%s' % i]
                aa['G_%s' % i] = 1 - aa['G_%s' % i] / (nnames['N_s%s' % i] * nnames['N_s%s' % i])
                
            G = N_s_1 * N_s_1 + N_s_2 * N_s_2
            G = 1 - G / (N * N)
            Gr = 0
            for i in range(R):
                Gr = Gr + nnames['N_s%s' % i] * aa['G_%s' % i] /N
            
            M_Value = 1 - Gr / G
        
        if self.method == 2:
            for i in range(R):
                if nnames['N_1%s' % i] == 0 or nnames['N_1%s' % i] == '' or nnames['N_2%s' % i] == 0 or nnames['N_2%s' % i] == '':
                    M_Value = ''
                    return
            
            nnames['E_%s' % i] = 0
            for i in range(R):
                nnames['E_%s' % i] = nnames['E_%s' % i] - nnames['N_1%s' % i] / nnames['N_s%s' % i] * log(nnames['N_1%s' % i] / \
                      nnames['N_s%s' % i])
                nnames['E_%s' % i] = nnames['E_%s' % i] - nnames['N_2%s' % i] / nnames['N_s%s' % i] * log(nnames['N_2%s' % i] / \
                      nnames['N_s%s' % i])
                nnames['E_%s' % i] = nnames['E_%s' % i] / log(2)
                
            E = 0
            E = E - N_s_1 / N * log(N_s_1 / N) - N_s_2 / N * log(N_s_2 / N)
            E = E / log(2)
            Er = 0
            for i in range(R):
                Er = Er + nnames['N_s%s' % i] * nnames['E_%s' % i] / N
                
            M_Value = 1 - Er / E
        
        if self.method == 3:
            N = N_s_1 + N_s_2
            X2 = 0
            for i in range(R):
                nnames['m_1%s' % i] = nnames['N_s%s' % i] * N_s_1 / N
                X2 = X2 + (nnames['N_1%s' % i] - nnames['m_1%s' % i]) * (nnames['N_1%s' % i] - nnames['m_1%s' % i]) / nnames['m_1%s' % i]
                nnames['m_2%s' % i] = nnames['N_s%s' % i] * N_s_2 / N
                X2 = X2 + (nnames['N_2%s' % i] - nnames['m_2%s' % i]) * (nnames['N_2%s' % i] - nnames['m_2%s' % i]) / nnames['m_2%s' % i]
                nnames['m_2%s' % i]
                
            M_Value = X2
            
        if self.method == 4:
            IV = 0
            for i in range(R):
                if nnames['N_1%s' % i] == 0 or nnames['N_1%s' % i] == '' or nnames['N_2%s' % i] == 0 or nnames['N_2%s' % i] == '' \
                or N_s_1 == 0 or N_s_1 == '' or N_s_2 == 0 or N_s_2 == '':
                    M_Value = ''
                    return
            for i in range(R):
                IV = IV + (nnames['N_1%s' % i] / N_s_1 - nnames['N_2%s' % i] / N_s_2) * log(nnames['N_1%s' % i] * N_s_2 / \
                          (nnames['N_2%s' % i] * N_s_1))
            M_Value = IV
            
        return M_Value
    
    def ApplyMap(self,DSin,VarX,DSVapMap):
        """
        about:分箱组数替换 Dataframe
        :param Dsin 原始数据
        :param VarX 需要更新的变量名 str
        :param DSVapMap 读取的需要更新的分组数据 Dataframe 
        :return
        """
        null_g = DSVapMap['group'].loc[DSVapMap['LowerLimit'].isnull()].max() #判断是否有null的group,有的话，把值改成group
        DSin.loc[:,VarX + '_g'] = 0
        if null_g > 0:
            DSin.loc[DSin[VarX].isnull(),[VarX + '_g']] = int(null_g) #有的话，全部是最大
        lmin = DSVapMap['LowerLimit'].min()
        nnull = DSVapMap.loc[DSVapMap['LowerLimit'] > lmin - 10] #这里计算数值分类是在10组之外的要用到分组，10组之内的不用替换
        nnull = nnull.sort_values(by=['LowerLimit'],ascending=True) 
        nnull = nnull.reset_index(drop=True)
        mm = nnull['group'].count()
        for i in range(mm): #开始替换数据
            ll = nnull['LowerLimit'][i]
            uu = nnull['UpperLimit'][i]
            gg = nnull['group'][i]
            if i == 0:
                DSin.loc[DSin[VarX] <= uu,[VarX + '_g']] = gg
            elif i == mm - 1 and uu > ll:
                DSin.loc[DSin[VarX] > ll,[VarX + '_g']] = gg
            elif i == mm - 1:
                DSin.loc[DSin[VarX] >= ll,[VarX + '_g']] = uu == ll and gg
            else:
                DSin.loc[(DSin[VarX] > ll) & (DSin[VarX] <= uu),[VarX + '_g']] = gg
                
class Feature_importance:
    """
    about:特征重要性排序
    """
    def __init__(self,indata,dep='y',select_num=None):
        """
        about:参数
        :param indata
        :param dep:y,pd.Series,binary,(0,1)
        :param select_num
        :param save_file
        """
        self.select_num = select_num
        self.indata = indata
        self.dep = dep
        
    def feature_select(self):
        """
        about:特征选择
        :return
        """
        predictors = [x for x in self.indata.columns if x not in [self.dep]]
        #超参数选优
        xgb_feature = XGBClassifier(learning_rate=0.1,#学习率
                                    n_estimators = 1000,#决定树的数量
                                    max_depth=5,# 构建树的深度，越大越容易过拟合
                                    min_child_weight=5,# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting
                                    gama=0,# 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子
                                    subsample=0.8,# 随机采样训练样本
                                    colsample_bytree=0.8,# 生成树时进行的列采样
                                    objective='binary:logistic',#binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。
                                    nthread=4,#cup 线程数
                                    scale_pos_weight=1,#在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
                                    seed=1)
        xgb_param = xgb_feature.get_xgb_params()
        xgtrain = xgb.DMatrix(self.indata[predictors].values,label=self.indata[self.dep].values) #数据矩阵化
        cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=xgb_feature.get_params()['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=50)
        xgb_feature.set_params(n_estimators=cvresult.shape[0])
        xgb_feature.fit(self.indata[predictors],self.indata[self.dep],eval_metric='auc') #
        featureimportance = pd.DataFrame(xgb_feature.feature_importances_) #得到特征重要性
        feat_names = pd.DataFrame(predictors)
        df_feature = (pd.merge(feat_names,featureimportance,left_index=True,right_index=True)).sort_values(by=['0_y'],ascending=False) #和变量名连表后排序
        df_feature.rename(columns={'0_x':'Feature','0_y':'FeatureImportanceScore'},inplace=True)
        df_feature = df_feature.reset_index().drop('index',1)
        self._feature_importance = df_feature
        if self.select_num is not None: #这里是依据要求选出最重要的几个特征
            if self.select_num <= len(self.indata.columns) - 1:
                selected_feature = list(df_feature.ix[0:self.select_num - 1,:]['Feature'])
                selected_feature.insert(0,self.dep)
                self._df_selected = self.indata[selected_feature]
                self._selected_feature = selected_feature[1:]
            else:
                print('选择的特征数目必须小于总特征数目')
            return pd.DataFrame(df_feature)

         
class Lr_model:
    """
    about:IR相关
    """
    def check_same_var(self,x):
        var_corr = x.corr(method='pearson',min_periods=1)
        same_var = []
        for i in x.columns:
            if var_corr[i].loc[var_corr[i] == 1].count() > 1:
                same_var.append(i)
                
        if len(same_var) > 0:
            for i in range(len(same_var)):
                print('完全相同的变量如下,请进行删除：' + same_var[i])
                
        else:
            print('无完全相同的变量，请进行后续操作')
            
    def in_model_var(self,in_x,model,m_type):
        if m_type == 'l1':#如果是l1则直接使用回归系数
            weight = pd.DataFrame(model.coef_).T #.coef_是回归系数,intercept_是截距
        else:
            if m_type == 'l2':#如果是l2则是回归系数的转置
                weight = pd.DataFrame(model.coef_).T
            else:
                print('type error')
        weight['index'] = weight.index
        var_namelist = in_x.T #得到X的转置
        var_namelist['varname'] = var_namelist.index
        varname = var_namelist['varname'].reset_index()
        varname['index'] = varname.index #得到变量的名
        model_tmp = pd.merge(varname,weight,how='inner',on=['index']) #将变量和回归系数结合起来
        model_var = model_tmp.drop('index',1)
        model_var.columns = ['var','coef'] #修改columns名
        var = list(model_var['var'].loc[abs(model_var['coef']) > 0]) #得到回归系数大于0的变量
        return var
        
    def lr_model_iter(self,x,y,dep,p_max=0.05,alpha=0.1,penalty='l2'): 
        model = LogisticRegression(C=alpha,penalty=penalty,class_weight='balanced',max_iter=100,random_state=1)
        #建立模型，用L1线性正则化.penalty是正则化选择参数，solver是优化算法选择参数。L1向量中各元素绝对值的和，作用是产生少量的特征，
        #而其他特征都是0，常用于特征选择；L2向量中各个元素平方之和再开根号，作用是选择较多的特征，使他们都趋近于0。
        #C值的目标函数约束条件：s.t.||w||1<C，默认值是0，C值越小，则正则化强度越大。
        #n_jobs表示bagging并行的任务数
        model.fit(x,y) #开始训练模型
        select_var = self.in_model_var(x,model,'l2') #调用函数，得到回归系数大于0的变量
        xx = x[select_var] #取相应的变量
        lr_model = LogisticRegression(penalty='l2',class_weight='balanced',max_iter=100,random_state=1)
        #采用l2正则化，其实是用l1正则化选取变量，用l2正则化拿到回归系数
        lr_model.fit(xx,y)
        modelstat = Lasso_var_stats(xx,y,lr_model,dep=dep) #调用类
        modelstat.all_stats_l2() #调用类的方法
        model_output = modelstat.var_stats  #得到结果
        p = model_output['pvalue'].max() #最大的P值
        c = model_output['coef'].min()  #最小的回归系数
        while p >= p_max or c < 0:
            if p >= p_max:#如果P值较大
                max_var = model_output['var'].loc[model_output['pvalue'] == p].values[0] #得到自变量
                x = x.drop(max_var,axis=1)
                print('删除P值最高的变量' + max_var + 'P值为' + str(p))
                model.fit(x,y)
                select_var = self.in_model_var(x,model,'l2')
                xx = x[select_var]
                lr_model.fit(xx,y)
                modelstat = Lasso_var_stats(xx,y,lr_model,dep=dep)
                modelstat.all_stats_l2()
                model_output = modelstat.var_stats
                p = model_output['pvalue'].max()
                c = model_output['coef'].min()
            elif c < 0: # 回归系数为负
                fu_var = model_output['var'].loc[model_output['coef'] == c].values[0]
                x = x.drop(fu_var,axis=1)
                print('删除系数异常' + fu_var)
                model.fit(x,y)
                select_var = self.in_model_var(x,model,'l2')
                xx = x[select_var]
                lr_model.fit(xx,y)
                modelstat = Lasso_var_stats(xx,y,lr_model,dep=dep)
                modelstat.all_stats_l2()
                model_output = modelstat.var_stats
                c = model_output['coef'].min()
                p = model_output['pvalue'].max()
                
        return (model_output,lr_model,select_var) #最后返回模型各个参数、模型、变量名

    
class Lasso_var_stats:
    """
    about:LR模型统计
    vars_vif:计算变量的vif值
    vars_contribute:计算变量的贡献度，且倒序排列
    vars_pvalue:计算变量的P值
    vars_corr:计算变量的相关系数
    all_stats:整合上面的统计指标
    """
    def __init__(self,xx,yy,lr_model,dep='y',alpha=1,penalty='l1',solver='liblinear',max_iter=100,random_state=1):
        """
        :param xx
        :param yy
        :param lr_model model,建好的模型
        :param dep str,y
        :param alpha: int,lasso 中的惩罚系数，计算P值是调用statsmodels中的lasso，需要重新拟合
        :param penalty:正则项规划
        :param solver
        :param max_iter
        :param random_state
        :param n_jobs
        
        """
        self.xx = xx
        self.yy = yy
        self.dep = dep
        self.lr_model = lr_model
        self.alpha = alpha
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
            
    def model_coef(self): #和下面的一样
        weight = pd.DataFrame(self.lr_model.coef_).T
        weight['index'] = weight.index
        intercept = self.lr_model.intercept_
        var_namelist = self.xx.T
        var_namelist['varname'] = var_namelist.index
        varname = var_namelist['varname'].reset_index()
        varname['index'] = varname.index
        model_tmp = pd.merge(varname,weight,how='inner',on=['index'])
        model_output = model_tmp.drop('index',1)
        model_output.columns = ['var','coef']
        model_output = model_output.loc[abs(model_output['coef']) > 0]
        model_output['intercept'] = intercept[0]
        self.model_output = model_output.set_index(['var'],drop=False)
        modelvar = list(self.model_output['var'])
        myvar = [x for x in self.xx.columns if x in modelvar]
        train_y = pd.DataFrame(self.yy)
        train_y.columns = [self.dep]
        self.mydat = pd.merge(train_y,self.xx[myvar],left_index=True,right_index=True)
        self.vars_data = self.mydat[myvar]
        self.newmodel = LogisticRegression(C=self.alpha,penalty=self.penalty,solver=self.solver,max_iter=self.max_iter,
                                           random_state=self.random_state)
        self.newmodel.fit(self.vars_data,train_y)
        
    def vars_vif(self):
        features = ('+').join(self.vars_data.columns) #将变量连接
        X = dmatrix(features,self.vars_data,return_type='dataframe')#分数据
        vif = pd.DataFrame()
        vif['VIF_Factor'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])] #计算方差膨胀因子
        #方差膨胀因子（Variance Inflation Factor，VIF）：容忍度的倒数，VIF越大，显示共线性越严重。经验判断方法表明：当0＜VIF＜10，
        #不存在多重共线性；当10≤VIF＜100，存在较强的多重共线性；当VIF≥100，存在严重多重共线性
        vif['features'] = X.columns
        self.df_vif = vif.set_index(['features'])
    
    def vars_contribute(self):
        train_x_contri = pd.DataFrame(self.vars_data.std(),columns=['std']) #这里计算贡献度
        train_x_contri['coef'] = self.newmodel.coef_[0,:]
        train_x_contri['std_coef'] = 0.5513 * train_x_contri['std'] * train_x_contri['coef']
        train_x_contri['contribute_ratio'] = train_x_contri['std_coef'] / sum(train_x_contri['std_coef'])
        train_x_contri_sort = train_x_contri.sort_values(by='contribute_ratio',ascending=False) #排序
        self.df_contribute = pd.DataFrame(train_x_contri_sort['contribute_ratio'])
        
    def vars_pvalue(self):
        features = ('+').join(self.vars_data.columns)
        dep_ = self.dep + ' ~'
        y,X = dmatrices(dep_ + features, self.mydat,return_type='dataframe')
        logit = sm.Logit(y,X).fit_regularized(alpha=1.0/ self.alpha)
        self.df_pvalue = pd.DataFrame(logit.pvalues.iloc[1:],columns=['pvalue'])
        
        
    def vars_corr(self):#求相关系数
        self.df_corr = pd.DataFrame(np.corrcoef(self.vars_data,rowvar=0),columns=self.vars_data.columns,index=self.vars_data.columns)
        
    def all_stats(self): # 这个和下面的基本一样
        self.model_coef()
        self.vars_vif()
        self.vars_contribute()
        self.vars_pvalue()
        self.vars_corr()
        df_corr_trans = pd.DataFrame(self.df_corr,columns= self.vars_data.columns,index=self.df_contribute.index)
        self.var_stats = (((self.model_output.merge(self.df_contribute,left_index=True,right_index=True)).merge(self.df_pvalue,
                           left_index=True,right_index=True)).merge(self.df_vif,left_index=True,right_index=True)).merge(df_corr_trans,
                            left_index = True,right_index=True)
        self.var_stats = self.var_stats.reset_index(drop=True)
        
    def model_coef_l2(self):
        weight = pd.DataFrame(self.lr_model.coef_).T #得到回归系数
        weight['index'] = weight.index
        intercept = self.lr_model.intercept_ #得到截距
        var_namelist = self.xx.T
        var_namelist['varname'] = var_namelist.index
        var_namelist = var_namelist['varname'].reset_index()
        varname = pd.DataFrame(var_namelist['varname']).reset_index()
        varname['index'] = varname.index
        model_tmp = pd.merge(varname,weight,how='inner',on=['index'])
        model_output = model_tmp.drop('index',1)
        model_output.columns = ['var','coef']
        model_output = model_output.loc[abs(model_output['coef']) > 0]
        model_output['intercept'] = intercept[0] #增加回归系数
        self.model_output = model_output.set_index(['var'],drop=False) #增加index
        modelvar = list(self.model_output['var'])
        myvar = [x for x in self.xx.columns if x in modelvar] #得到符合要求的变量名
        train_y = pd.DataFrame(self.yy)
        train_y.columns = [self.dep]
        self.mydat = pd.merge(train_y,self.xx[myvar],left_index=True,right_index=True) #得到符合要求变量的数据
        self.vars_data = self.mydat[myvar]
        self.newmodel = self.lr_model
        
    def vars_pvalues_l2(self):
        features = ('+').join(self.vars_data.columns)
        dep_ = self.dep + ' ~'
        y,X = dmatrices(dep_ + features,self.mydat,return_type='dataframe') #分割出X，y
        logit = sm.Logit(y,X).fit() #执行逻辑回归,另一个包
        self.df_pvalue = pd.DataFrame(logit.pvalues.iloc[1:],columns=['pvalue']) #这里求P值
        self.summary = logit.summary() #这里是统计描述所有信息
        
    def all_stats_l2(self): #调用所有的方法
        self.model_coef_l2() #用l2方法得到符合要求的变量的数据
        self.vars_vif() #检验多重共线性问题
        self.vars_contribute() #计算自变量的贡献度
        self.vars_pvalues_l2() #这个也是建模，用的是statsmodels.api.Logit
        self.vars_corr() #求相关系数
        df_corr_trans = pd.DataFrame(self.df_corr,columns=self.df_contribute.index,index=self.df_contribute.index)
        self.var_stats = (((self.model_output.merge(self.df_contribute,left_index=True,right_index=True)).merge(self.df_pvalue,
                           left_index=True,right_index=True)).merge(self.df_vif,left_index=True,right_index=True)).merge(df_corr_trans,
                        left_index=True,right_index=True) #连接上述的所有表
        self.var_stats = self.var_stats.reset_index(drop=True)
        

class Model_evaluation:
    """
    about:模型评价
    roc_curve:画roc曲线和计算auc值
    ks_curve:画KS曲线和计算KS值
    group_risk_curve:画分组图和风险曲线，并产生每个样本的分组和总的聚合分组情况
    
    """
    def __init__(self,y_true,predict_prob):
        """
        about:初始化参数
        :param y_ture:y,pd.Series,binary,(0,1)
        :param predict_prob:prob,pd.Series,Continuous,(0,1)
        """
        self.y = y_true
        self.prob = predict_prob
        self.prob.index = self.y.index
        self.save_file = os.getcwd()
            
    def roc_curve(self,file_name=None):
        """
        about:roc 曲线
        :return
        """
        false_positive_rate,recall,thresholds = roc_curve(self.y,self.prob) 
        #roc_curve 参数第一个是真实的y,第二个是预测的概率，返回的是FPR、TPR、和真正概率
        roc_auc = auc(false_positive_rate,recall) #auc是计算面积
        plt.title('Receiver Operating Characteristic') #标题
        plt.plot(false_positive_rate,recall,'b',label='AUC = %0.2f' % roc_auc) #label是标签
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        if self.save_file is not None:
            if file_name is not None:
                plt.savefig(self.save_file + '\\' + file_name)
        plt.close()

    def roc_ks_score(self):
        """
        返回roc值和ks值
        """
        false_positive_rate,recall,thresholds = roc_curve(self.y,self.prob)
        roc_auc = auc(false_positive_rate,recall)
        data = pd.DataFrame([self.y,self.prob]).T
        data.columns = ['y','prob']
        y = data.y
        PosAll = pd.Series(y).value_counts()[1] #实际是1的个数
        NegAll = pd.Series(y).value_counts()[0] #实际是0的个数
        data = data.sort_values(by='prob',ascending=True) #按概率升序排列
        pCumsum = data['y'].cumsum() #累计1的
        nCumsum = np.arange(len(y)) - pCumsum + 1 #累计0的
        pCumsumPer = pCumsum / PosAll  #累计1的占总1的
        nCumsumPer = nCumsum / NegAll #累计0的占总0的
        ks = max(nCumsumPer - pCumsumPer) #这个是KS值
        return roc_auc,ks
    
    def ks_curve(self,file_name=None):
        """
        about:ks曲线
        :param file_name: file name of pic,str,'xx.pdf'
        :return
        """
        data = pd.DataFrame([self.y,self.prob]).T
        data.columns = ['y','prob']
        y = data.y
        x_axis = np.arange(len(y)) / float(len(y))
        PosAll = pd.Series(y).value_counts()[1] #实际是1的个数
        NegAll = pd.Series(y).value_counts()[0] #实际是0的个数
        data = data.sort_values(by='prob',ascending=True) #按概率升序排列
        pCumsum = data['y'].cumsum() #累计1的
        nCumsum = np.arange(len(y)) - pCumsum + 1 #累计0的
        pCumsumPer = pCumsum / PosAll  #累计1的占总1的
        nCumsumPer = nCumsum / NegAll #累计0的占总0的
        ks = max(nCumsumPer - pCumsumPer) #这个是KS值
        plt.figure(figsize=[8,6])
        plt.title('ks_curve(ks=%0.2f)' % ks)
        plt.plot(x_axis,pCumsumPer,color='red') #累计1占总1
        plt.plot(x_axis,nCumsumPer,color='blue') #累计0占总0
        plt.legend(('负样本洛伦兹曲线','正样本洛伦兹曲线'),loc='lower right') #打标记
        if self.save_file is not None:
            if file_name is not None:
                plt.savefig(self.save_file + '\\' + file_name)
        plt.close()
            
    def group_risk_curve(self,n,file_name=None):
        """
        about:风险曲线图
        :param n :分组的组数
        :param file_name:图片名,str,'xx.pdf'
        :return
        """
        data = pd.DataFrame([self.y,self.prob]).T 
        data.columns = ['y','prob']
        num = len(pd.qcut(data['prob'],q=n,retbins=True,duplicates='drop')[1]) #得到切点数 ,可再通过num来定位标签数
        prob_cuts = pd.qcut(data['prob'],q=n,labels=range(num)[1:],retbins=True,duplicates='drop') # 将qcut数据进行按频率n等份，cut是按值进行等分
        #其中labels是重新设置标签，retbins保留分割后的数据和分割点，以tuple打包
        cuts_bin = pd.Series(prob_cuts[0]) #分割都的数据
        cut_points = pd.Series(prob_cuts[1]) #分割点
        data['group'] = cuts_bin #分组
        data['lower_point'] = [0 for i in data.index] #初始化
        data['upper_point'] = [0 for i in data.index] #初始化
        for i in range(len(cut_points.index) - 1): #开始赋值
            data.loc[data['group'] == i + 1,['lower_point']] = cut_points[i]
            data.loc[data['group'] == i + 1,['upper_point']] = cut_points[i + 1]
            
        avg_risk = data['y'].sum() / data['y'].count() #平均风险倍数
        group = data.groupby(['group','lower_point','upper_point'],as_index=False) #分组
        group_df = group['y'].agg({'y_count':'sum','count':'count'}) #应用，计算出sum和count
        group_df['group_per'] = group_df['count'] / group_df['count'].sum()
        group_df['bad_per'] = group_df['y_count'] / group_df['count']
        group_df['risk_times'] = group_df['bad_per'] / avg_risk #风险倍数
        group_df = pd.DataFrame(group_df,columns=['group','count','y_count',
        'group_per','bad_per','risk_times','lower_point','upper_point'])
        fig,ax1 = plt.subplots()
        ax2 = ax1.twinx() #添加副坐标轴
        ax1.bar(x=group_df['group'],height= group_df['group_per'],width=0.6,align='center',yerr=1e-06) #yerr让柱形图顶端空出一些
        ax2.plot(list(group_df['group']),list(group_df['risk_times']),'o-r',color='red') # o是带有结点
        ax1.set_xlabel('group',fontsize=12)
        ax1.set_ylabel('percent',fontsize=12)
        ax2.set_ylabel('risktimes',fontsize=12)
        ax1.set_xlim([0,max(group_df['group']) + 1])
        plt.title('group_risk_curve')        
        if self.save_file is not None:
            if file_name is not None:
                plt.savefig(self.save_file + '\\' + file_name)
            return (data,cut_points,group_df) #返回数据、切点、切割都的统计指标
        
    def cesuan_risk_curve(self,cesuan_prob,n=20,file_name=None):
        """
        about:测算预测值分组图
        :param cesuan_prob:
        :param n 分组的组数
        :param file_name 图片名,str,'xx.pdf'
        :return:
        """
        prob_cuts = pd.qcut(cesuan_prob['prob'],q=n,retbins=True,duplicates='drop')
        cuts_bin = pd.Series(prob_cuts[0])
        cesuan_cut_points = pd.Series(prob_cuts[1])
        cesuan_prob['group'] = cuts_bin
        cesuan_prob['lower_point'] = [0 for i in cesuan_prob.index]
        cesuan_prob['upper_point'] = [0 for i in cesuan_prob.index]
        for i in range(len(cesuan_cut_points.index) - 1):
            cesuan_prob['lower_point'][cesuan_prob['group'] == i + 1] = cesuan_cut_points[i]
            cesuan_prob['upper_point'][cesuan_prob['group'] == i + 1] = cesuan_cut_points[i + 1]
            
        cesuan_group = cesuan_prob.groupby(['group','lower_point','upper_point'],as_index=False)
        cesuan_group_df = cesuan_group['group'].agg({'count':'count'})
        cesuan_group_df['group_per'] = cesuan_group_df['count'] / cesuan_group_df['count'].sum()
        data = pd.DataFrame([pd.Series(self.y),pd.Series(self.prob)]).T
        data.columns = ['y','prob']
        data['group'] = [0 for i in data.index]
        for i in range(len(cesuan_cut_points) - 1):
            if i == 0:
                data['group'].loc[data['prob'] <= cesuan_cut_points[i + 1]] = i + 1
            else:
                if i == len(cesuan_cut_points) -2:
                    data['group'].loc[data['prob'] > cesuan_cut_points[i]] = i + 1
                else:
                    data['group'].loc[(data['prob'] > cesuan_cut_points[i]) & (data['prob'] <= cesuan_cut_points[i + 1])] = i + 1
                    
        avg_risk = data['y'].sum() / data['y'].count()
        group = data.groupby(['group'],as_index=False)
        group_df = group['y'].agg({'y_count':'sum','count':'count'})
        group_df['bad_per'] = group_df['y_count'] / group_df['count']
        group_df['risk_times'] = group_df['bad_per'] / avg_risk
        cesuan_risk_curve = pd.merge(cesuan_group_df[['group','group_per','lower_point','upper_point']],group_df[['group',
                                     'count','y_count','bad_per','risk_times']],on='group')
        fig,ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.bar(x=cesuan_risk_curve['group'],height=cesuan_risk_curve['group_per'],width=0.6,align='center',yerr=1e-06)
        ax2.plot(list(cesuan_risk_curve['group']),list(cesuan_risk_curve['risk_times']),'-ro',color='red')
        ax1.set_xlabel('group',fontsize=12)
        ax1.set_ylabel('percent',fontsize=12)
        ax2.set_ylabel('risktimes',fontsize=12)
        ax1.set_xlim([0,max(cesuan_risk_curve['group']) + 1])
        plt.title('group_risk_curve')
        if self.save_file is not None:
            if file_name is not None:
                plt.savefig(self.save_file + '\\' + file_name)
            plt.show()
            return (cesuan_prob,cesuan_cut_points,cesuan_risk_curve)
        

class Apply_benchmark:
    """
    cut_points: 切点，包含头尾，pd.Series
    apply_benchmark_y: 测算样本带y值，在某一benchmark下的产出分组
    apply_benchmark:在某一benchmark下的产出分组
    """
    def __init__(self,cut_points):
        self.cut_points = cut_points
        
    def apply_benchmark_y(self,y_true,predict_prob):
        """
        about
        :param y_true:y,pd.Series,binary,(0,1)
        :param predict_prob:prob,pd.Series,Continuous,(0,1)
        :return
        """
        predict_prob.index = y_true.index
        data = pd.DataFrame([y_true,predict_prob]).T
        data.columns = ['y','prob']
        data['group'] = [0 for i in data.index]
        data['lower_point'] = [0 for i in data.index]
        data['upper_point'] = [0 for i in data.index]
        for i in range(len(self.cut_points) -1): #按照给的切点进行分组。其中切点是包含最低值，和最高值
            if i == 0:
                data['group'].loc[data['prob'] <= self.cut_points[i+1]] = i + 1
                data['lower_point'].loc[data['prob'] <= self.cut_points[i + 1]] = self.cut_points[i]
                data['upper_point'].loc[data['prob'] <= self.cut_points[i + 1]] = self.cut_points[i+1]
            elif i == len(self.cut_points) - 2:
                data['group'].loc[data['prob'] > self.cut_points[i]] = i + 1
                data['lower_point'].loc[data['prob'] > self.cut_points[i]] = self.cut_points[i]
                data['upper_point'].loc[data['prob'] > self.cut_points[i]] = self.cut_points[i + 1]
            else:
                data['group'].loc[(data['prob'] > self.cut_points[i]) & (data['prob'] <= self.cut_points[i+1])] = i + 1
                data['lower_point'].loc[(data['prob'] > self.cut_points[i]) & (data['prob'] <= self.cut_points[i+1])] = self.cut_points[i]
                data['upper_point'].loc[(data['prob'] > self.cut_points[i]) & (data['prob'] <= self.cut_points[i+1])] = self.cut_points[i+1]
                
        avg_risk = data['y'].sum() / data['y'].count()
        group = data.groupby(['group','lower_point','upper_point'],as_index=False)
        group_df = group['y'].agg({'y_count':'sum','count':'count'})
        group_df['group_per']  = group_df['count'] / group_df['count'].sum()
        group_df['bad_per'] = group_df['y_count'] / group_df['count']
        group_df['risk_times'] = group_df['bad_per'] / avg_risk
        group_df = pd.DataFrame(group_df,columns=['group','count','y_count','group_per','bad_per','risk_times','lower_point','upper_point'])
        return (data,group_df) 
    # 同样是划分

    
class Calc_psi:
    """
    about:
    var_psi:计算单变量的psi（包括模型分组)
    vars_psi:计算多个变量的psi
    """
    def __init__(self,data_actual=None,data_expect=None):
        """
        :param data_actual:DataFrame 实际占比，即外推样本分组后的变量
        :param data_expect:DataFrame 预期占比，即建模样本分组后的变量
        """
        self.data_actual = data_actual
        self.data_expect = data_expect
        
    def var_psi(self,series_actual,series_expect):
        """
        about:psi compute
        :param series_actual:Series,实际样本分组后的变量（或者样本分组）
        :param series_expect:Series,预测样本分组后的变量（或者样本分组）
        psi计算：sum((实际占比-预期占比)*In(实际占比/预期占比))
        一般认为psi小于0.1时候模型稳定性很高，0.1-0.25一般，大于0.25模型稳定性差
        :return
        """
        series_actual_counts = pd.DataFrame(series_actual.value_counts(sort=False,normalize=True))
        #normalize 是占比,sort=False是按index排列，
        series_actual_counts.columns = ['per_1'] #实际每组的占比
        series_expect_counts = pd.DataFrame(series_expect.value_counts(sort=False,normalize=True))
        series_expect_counts.columns = ['per_2'] #预期的每组的占比
        series_counts = series_actual_counts.merge(series_expect_counts,how='right',left_index=True,right_index=True) #进行连表
        series_counts['per_diff'] = series_counts['per_1'] - series_counts['per_2'] #求差
        series_counts['per_in_ratio'] = (series_counts['per_1'] / series_counts['per_2']).apply(lambda x: log(x)) #求比值再取log
        psi = (series_counts['per_diff'] * series_counts['per_in_ratio']).sum() #差值*上面的结果，再求和
        return psi
    
    def vars_psi(self): #对所有的变量全求psi
        col_psi_dict = {}
        for col in self.data_expect.columns:
            psi = self.var_psi(self.data_actual[col],self.data_expect[col])
            col_psi_dict[col] = psi
        
        return pd.Series(col_psi_dict)


# 处理学历
def edu(x):
    if x in ('专科', '专科(高职)', '夜大电大函大普通班'):
        return 1
    elif x in ('本科','第二本科'):
        return 2
    elif x in ('硕士研究生','硕士'):
        return 3
    elif x in ('博士', '博士研究生'):
        return 4


def status(x):
    """
    依据24期表现，如果当前逾期，返回1，否则返回0
    """
    if x == 'nan':
        return np.NaN
    else:
        if re.search('\d', x[-1]):
            return 1
        else:
            return 0


def nonetonan(x):
    """
    把字符串None替换成np.NaN,非Non不变
    """
    if x == 'None':
        return np.NaN
    else:
        return x


def ip_net(x):
    """
    把wifi设置为1,其他为0
    """
    if x in ('WIFI', 'wifi'):
        return 1
    else:
        return 0


def phonetype(x):
    """
    把iphone设置为1，其他的设置为0
    """
    if x == 'iPhone':
        return 1
    else:
        return 0


def ks_score(model,datas,y_true):
    """
    计算KS分数
    model: 训练后的模型
    datas: 测试数据
    y_true: 测试数据类型
    """
    y_prob = model.predict_proba(datas)[:,1]
    data = pd.DataFrame([y_true,y_prob]).T
    data.columns = ['y','prob']
    y = data.y
    PosAll = pd.Series(y).value_counts()[1] #实际是1的个数
    NegAll = pd.Series(y).value_counts()[0] #实际是0的个数
    data = data.sort_values(by='prob',ascending=True) #按概率升序排列
    pCumsum = data['y'].cumsum() #累计1的
    nCumsum = np.arange(len(y)) - pCumsum + 1 #累计0的
    pCumsumPer = pCumsum / PosAll  #累计1的占总1的
    nCumsumPer = nCumsum / NegAll #累计0的占总0的
    ks = max(pCumsumPer-nCumsumPer) #这个是KS值
    return ks


def selectData_srepwise(df,target,intercept=True,normalize=False,
                        criterion='bic',p_value_enter=0.05,f_pvalue_enter=0.05,
                        direction='backward',show_step=True,criterion_enter=None,
                        criterion_remove=None,max_iter=200,**kw):
    '''
    逐步回归
    df:数据集 target 为第一列
    target:str 回归相关的变量
    intercept:bool 模型是否存在截距
    criterion: str 默认bic 逐步回归优化规则
    f_pavalue_enter:float 默认0.05 当选择criterion='ssr'时,模型加入或移除变量的f_pvalue阈值
    p_values_enter: float 默认0.05 当选择derection='both'时，移除变量的pvalue阈值
    direction: str, 默认是'backward' 逐步回归方向
    show_step: bool, 默认是True  是否显示逐步回归过程
    criterion_enter:float 默认是None 当选择derection='both'或'forward'时，模型加入变量
    的相应的criterion阈值
    criterion_remove: float 默认是None 当选择derection='backward'时，模型移除变量的
    相应的criterion阈值
    max_iter: int 默认是200 模型最大迭代次数
    '''
    criterion_list = ['bic','aic','ssr','rsquared','rsquared_adj']
    if criterion not in criterion_list:
        raise IOError('请输入正确的criterion,必须是以下内容之一:','\n',criterion_list)
    
    direction_list = ['backward','forward','both']
    if direction not in direction_list:
        raise IOError('请输入正确的direction,必须是以下内容之一:','\n',direction_list)
    
    # 默认p_enter参数
    p_enter= {'bic':0.0,'aic':0.01,'ssr':0.1,'rsquared':0.05,'rsquared_adj':-0.05}
    if criterion_enter:
        p_enter[criterion] = criterion_enter
        
    # 默认p_remove参数
    p_remove = {'bic':0.01,'aic':0.01,'ssr':0.1,'rsquared':0.05,'rsquared_adj':-0.05}
    if criterion_remove:
        p_remove[criterion] = criterion_remove
    
    if normalize:  # 如果需要标准化数据
        intercept = False
        df_std = StandardScaler.fit_transform(df)
        df = pd.DataFrame(df_std,columns=df.columns,index=df.index)
    
    '''
    forward
    '''
    if direction == 'forward':
        remaining = list(df.columns)  # 自变量集合
        remaining.remove(target)
        selected = []  # 初始化选入模型的变量列表
        if intercept:  # 判断是否有截距
            formula = '{}~{}+1'.format(target,remaining[0])
        else:
            formula = '{}~{}-1'.format(target,remaining[0])
        result = smf.ols(formula,df).fit()  # 最小二乘法回归模型拟合
        current_score = eval('result.'+criterion)
        best_new_score = eval('result.'+criterion)
        if show_step:
            print('\nstepwise staring:\n')
        iter_times = 0
        # 当变量未删除完，并且当前评分更新时进行循环
        while remaining and (current_score == best_new_score) and (
                iter_times < max_iter):
            scores_with_candidates = []  # 初始化变量及评分列表
            for candidate in remaining:  # 在未删除的变量中每次选择一个变量进入模型
                if intercept:
                    formula = '{}~{}+1'.format(target,'+'.join(selected+[candidate]))
                else:
                    formula = '{}~{}-1'.format(target,'+'.join(selected+[candidate]))
                result = smf.ols(formula,df).fit() #最小二乘回归模型拟合
                fvalue = result.fvalue
                f_pvalue = result.f_pvalue
                score = eval('result.'+criterion)
                scores_with_candidates.append((score,candidate,fvalue,f_pvalue))
                #  记录此次循环的参数、变量、F值、F_P值

            if criterion == 'ssr':  # 这几个指标取最小值进行优化
                scores_with_candidates.sort(reverse=True)
                (best_new_score, best_candidate,best_new_fvalue,
                 best_new_f_pvalue) = scores_with_candidates.pop()
                if ((current_score-best_new_score) > p_enter[criterion]) and (
                        best_new_f_pvalue < f_pvalue_enter):
                #  如果当前评分大于最新评分
                    remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优对应的变量
                    selected.append(best_candidate)  # 将最小最优分对应的变量放入已选变量列表
                    current_score = best_new_score
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, SSR=%.3f, Fstat=%.3f, FpValue=%.3e' %
                              (best_candidate,best_new_score,best_new_fvalue,
                               best_new_f_pvalue))
                elif (current_score-best_new_score) >= 0 and (
                        best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:
                    # 当评分差小于p_enter，且为第一次迭代，不能没有变量
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:
                        print('Adding %s,%s=%.3f' % (best_candidate,criterion,
                                                     best_new_score))
                elif (best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:
                    #   当评分差小于p_enter,且为第一次迭代
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    if show_step:
                        print('Adding %s,%s=%.3f' % (remaining[0],criterion,
                                                     best_new_score))

            elif criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
                scores_with_candidates.sort(reverse=True)  # 对评分降序排列
                (best_new_score, best_candidate, best_new_fvalue,
                 best_new_f_pvalue) = scores_with_candidates.pop()
                # 提取最小分数及其对应变量
                if (current_score-best_new_score)>p_enter[criterion]:
                    # 如果当前评分大于最新评分
                    remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
                    selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
                    current_score = best_new_score  # 更新当前评分
                    if show_step:
                        print('Adding %s,%s=%.3f' % (best_candidate,criterion,
                                                     best_new_score))
                elif (current_score-best_new_score) >=0 and iter_times==0:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:
                        print('Adding %s,%s=%.3f' % (best_candidate,criterion,
                                                     best_new_score))
                elif iter_times == 0:
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    if show_step:
                        print('Adding %s,%s=%.3f' % (remaining[0],criterion,
                                                     best_new_score))

            else:
                scores_with_candidates.sort()
                # 前面的'bic','aic','ssr'都是越小越好，后面的'rsquared','rsquared_adj'
                # 是越大越好
                (best_new_score,best_candidate,best_new_fvalue,
                best_new_f_pvalue) = scores_with_candidates.pop()
                if (best_new_score-current_score)>p_enter[criterion]:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:
                        print('Adding %s,%s=%.3f' %(best_candidate,criterion,
                                                    best_new_score))
                elif (best_new_score-current_score) >=0 and iter_times == 0:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:
                        print('Adding %s,%s=%.3f' % (best_candidate,criterion,
                                                     best_new_score))
                elif iter_times == 0:
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    if show_step:
                        print('Adding %s,%s=%.3f' % (remaining[0],criterion,
                                                     best_new_score))
            iter_times += 1

        if intercept:
            formula = '{} ~ {} + 1'.format(target,' + '.join(selected))
        else:
            formula = '{} ~ {} - 1'.format(target,' + '.join(selected))
        stepwise_model = smf.ols(formula,df).fit()
        if show_step:
            print('\nLinear regression model:', '\n ',stepwise_model.model.formula)
            print('\n', stepwise_model.summary())
    '''
    backward
    '''
    if direction == 'backward':
        remaining,selected = set(df.columns),set(df.columns)  # 自变量集合
        remaining.remove(target)
        selected.remove(target)
        # 初始化当前评分，最优化新评分
        if intercept:
            formula = '{} ~ {} + 1'.format(target,' + '.join(selected))
        else:
            formula = '{} ~ {} - 1'.format(target,' - '.join(selected))
        
        result = smf.ols(formula,df).fit()
        current_score = eval('result.'+criterion)
        worst_new_score = eval('result.'+criterion)
        
        if show_step:
            print('\nstepwise starting:\n')
        iter_times = 0
        # 当变量未剔除完，并且当前评分更新时进行循环
        while remaining and (current_score == worst_new_score) and (
                iter_times < max_iter):
            scores_with_eliminations = []  # 初始化变量以及其评分列表
            for elimination in remaining:  # 在未剔除的变量中每次选择一个变量进入模型
                if intercept:
                    formula = '{} ~ {} + 1'.format(
                            target,' + '.join(selected-set(elimination)))
                else:
                    formula = '{} ~ {} - 1'.format(
                            target,' + '.join(selected-set(elimination)))
                    
                result = smf.ols(formula,df).fit()
                fvalue = result.fvalue
                f_pvalue = result.f_pvalue
                score = eval('result.'+criterion)
                scores_with_eliminations.append((score,elimination,fvalue,f_pvalue))
                
            if criterion == 'ssr':
                scores_with_eliminations.sort(reverse=True)  # 降序排列
                (worst_new_score,worst_elimination,worst_new_fvalue,
                 worst_new_f_pvalue) = scores_with_eliminations.pop()
                if ((worst_new_score-current_score) < p_remove[criterion]) and (
                        worst_new_f_pvalue < f_pvalue_enter):
                    remaining.remove(worst_elimination)
                    selected.remove(worst_elimination)
                    current_score = worst_new_score
                    if show_step:
                        print('Removing %s,SSR=%.3f,Fstat=%.3f,FpValue=%.3e' %
                              (worst_elimination,worst_new_score,
                               worst_new_fvalue,worst_new_f_pvalue))

            elif criterion in ['bic','aic']:
                scores_with_eliminations.sort(reverse=True) 
                (worst_new_score,worst_elimination,worst_new_fvalue,
                 worst_new_f_pvalue) = scores_with_eliminations.pop()
                if (worst_new_score-current_score) < p_remove[criterion]:
                    remaining.remove(worst_elimination)
                    selected.remove(worst_elimination)
                    current_score = worst_new_score
                    if show_step:
                        print('Removing %s,%s=%.3f' % (worst_elimination,
                                                       criterion,worst_new_score))

            else:
                scores_with_eliminations.sort(reverse=False)
                (worst_new_score,worst_elimination,worst_new_fvalue,
                 worst_new_f_pvalue) = scores_with_eliminations.pop()
                if (current_score-worst_new_score) < p_remove[criterion]:
                    remaining.remove(worst_elimination)
                    selected.remove(worst_elimination)
                    current_score = worst_new_score
                    if show_step:
                        print('Removing %s,%s=%.3f' % (worst_elimination,
                                                       criterion,worst_new_score))
            iter_times += 1
            
        if intercept:
            formula = '{} ~ {} + 1'.format(target,' + '.join(selected))
        else:
            formula = '{} ~ {} - 1'.format(target,' + '.join(selected))
        stepwise_model = smf.ols(formula,df).fit()
        if show_step:
            print('\nLinear regression model:', '\n ',stepwise_model.model.formula)
            print('\n',stepwise_model.summary())
            
    '''
    both
    '''
    if direction == 'both':
        remaining = list(df.columns) # 自变量集合
        remaining.remove(target)
        selected = [] # 初始化选入模型的变量列表
        # 初始化当前评分，最优化评分
        if intercept:
            formula = '{} ~ {} + 1'.format(target,remaining[0])
        else:
            formula = '{} ~ {} - 1'.format(target,remaining[0])
        result = smf.ols(formula,df).fit()
        current_score = eval('result.'+criterion)
        best_new_score = eval('result.'+criterion)
        
        if show_step:
            print('\nstepwise starting:\n')
        # 当变量未剔除完，并且当前评分更新时进行循环
        iter_times = 0
        while remaining and (current_score==best_new_score) and (iter_times < max_iter):
            scores_with_candidates = []
            for candidate in remaining:
                if intercept:
                    formula = '{} ~ {} + 1'.format(target,' + '.join(selected+[candidate]))
                else:
                    formula = '{} ~ {} - 1'.format(target,' + '.join(selected+[candidate]))
                result = smf.ols(formula,df).fit()
                fvalue = result.fvalue
                f_pvalue = result.f_pvalue
                score = eval('result.' + criterion)
                scores_with_candidates.append((score,candidate,fvalue,f_pvalue))
            
            if criterion == 'ssr':
                scores_with_candidates.sort(reverse=True)
                (best_new_score,best_candidate,best_new_fvalue,
                 best_new_f_pvalue) = scores_with_candidates.pop()
                if ((current_score-best_new_score)>p_enter[criterion]) and (
                        best_new_f_pvalue < f_pvalue_enter):
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:
                        print('Adding %s,SSR=%.3f,Fstat=%.3f,FpValue=%.3e' % (
                                best_candidate, best_new_score,best_new_fvalue,
                                best_new_f_pvalue))
                elif (current_score-best_new_score) >= 0 and (
                        best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:
                        print('Adding %s,%s=%.3f' % (
                                best_candidate,criterion,best_new_score))
                elif (best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    if show_step:
                        print('Adding %s,%s=%.3f' % (remaining[0],criterion,
                                                     best_new_score))
            
            elif criterion in ['bic','aic']:
                scores_with_candidates.sort(reverse=True)
                (best_new_score,best_candidate,best_new_fvalue,
                 best_new_f_pvalue) = scores_with_candidates.pop()
                if (current_score-best_new_score) > p_enter[criterion]:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:
                        print('Adding %s,%s=%.3f' % (best_candidate,criterion,
                                                    best_new_score))
                elif (current_score-best_new_score) >=0 and iter_times == 0:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:
                        print('Adding %s,%s=%.3f' % (best_candidate,criterion,
                                                     best_new_score))
                elif iter_times == 0:
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    if show_step:
                        print('Adding %s,%s=%.3f' % (remaining[0], criterion,
                                                     best_new_score))
            else:
                scores_with_candidates.sort()
                (best_new_score,best_candidate,best_new_fvalue,
                 best_new_f_pvalue) = scores_with_candidates.pop()
                if (best_new_score-current_score) > p_enter[criterion]:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:
                        print('Adding %s,%s=%.3f' % (best_candidate,criterion,
                                                     best_new_score))
                elif (best_new_score-current_score) >= 0 and iter_times == 0:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    if show_step:
                        print('Adding %s,%s=%.3f' % (remaining[0],criterion,
                                                     best_new_score))
            if intercept:
                formula = '{} ~ {} + 1'.format(target,' + '.join(selected))
            else:
                formula = '{} ~ {} - 1'.format(target,' + '.join(selected))
            
            result = smf.ols(formula,df).fit()
            if iter_times >= 1:  #  当第二次循环时判断变量的pvalue是否达标
                if result.pvalues.max() > p_value_enter:
                    var_removed = result.pvalues[
                            result.pvalues==result.pvalues.max()].index[0]
                    p_value_removed = result.pvalues[
                            result.pvalues==result.pvalues.max()].values[0]
                    selected.remove(result.pvalues[
                            result.pvalues==result.pvalues.max()].index[0])
                    if show_step:
                        print('Removing %s,Pvalue=%.3f' % (var_removed,
                                                           p_value_removed))
            iter_times += 1
        
        if intercept:
            formula = '{} ~ {} + 1'.format(target,' + '.join(selected))
        else:
            formula = '{} ~ {} - 1'.format(target,' + '.join(selected))
            
        stepwise_model = smf.ols(formula,df).fit()
        if show_step:
            print('\nLinear regression model:','\n ',stepwise_model.model.formula)
            print('\n',stepwise_model.summary())
    if intercept:
        stepwise_feat_selected_ = list(stepwise_model.params.index[1:])
    else:
        stepwise_feat_selected_ = list(stepwise_model.params.index)
    return stepwise_feat_selected_
          

























