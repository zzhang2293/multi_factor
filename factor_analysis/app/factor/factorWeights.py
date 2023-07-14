from collections import Counter
import numpy.linalg as nlg  #线性代数函数
import numpy as np
import pandas as pd

class calcWeights:

    '''
        函数: calcFactorWeights (non-async)
        -------
        1) 摘要：此函数用于计算多因子策略中的权重
        2) 函数输入
            - mode [REQUIRED]
                计算方式，可以是全部等权重(equal)，也可以是按照大类等权重(category), 也可以是基于历史IC的权重(smart)
            - listOfFactors [REQUIRED]
                - list of factor names (list[str])
            - listOfCategories [OPTIONAL, REQUIRED IF mode == 'category']
                - list of factor categories (list[int]), 如[1, 1, 2, 3, 1, 4] 则代表有1, 2, 5是第一大类, 3是第二大类, 等等
            - listOfHistoricalIC [OPTIONAL, REQUIRED IF mode == 'smart']
                - pd.DataFrame, 每个指标要有一个对应的历史IC值
            - smartmode [OPTIONAL, REQUIRED IF mode == 'smart']
                - 如何使用历史IC智能计算, 可以通过优化得出最大的IR(IRsolver)
        3) 输出: list[float], 每个因子的权重
        4) 可控变量:
            - ICPeriod: int, 选择用于计算历史IC的时间长度
        5) 案例
            - 等权重模式 calcWeights('equal', ['a', 'b', 'c', 'd', 'e', 'f'])
            - 大类等权重模式 calcWeights('category', ['a', 'b', 'c', 'd', 'e', 'f'], listOfCategories = [1, 1, 2, 3, 1, 4])
            - 历史IC智能计算模式 calcWeights('smart', ['a', 'b', 'c', 'd', 'e', 'f'], HistoricalIC = {pd.DataFrame Object}, smartmode='IRSolver')
    '''
    
    def calcWeights(self, mode:str, listOfFactors:list, listOfCategories:list = [], HistoricalIC:list = [], smartmode = 'IRSolver') -> list:

        if mode not in ['equal', 'category', 'smart']:
            raise Exception('Wrong Category!')

        #所有的因子权重都一样
        if mode == 'equal':
            #create a list of size len(listOfFactors) with equal weights summing up to 1
            return [1 / len(listOfFactors)] * len(listOfFactors)
        
        #所有的因子大类权重一样 每大类里面的每个因子权重也一样
        elif mode == 'category':
            if len(listOfCategories) != len(listOfFactors):
                raise Exception('Error Category Size')

            
            ct = Counter(listOfCategories)
            eachCatWeight = 1 / len(ct.keys())
            
            res = []

            for i in listOfCategories:
                res.append(eachCatWeight / ct[i])

            return res
        
        #基于历史IC来选
        elif mode == 'smart':
            if HistoricalIC.shape[1] != len(listOfFactors):
                raise Exception('Error IC Size')
            
            if smartmode not in ['IRSolver']:
                raise Exception('Error Smart Mode')
            
            ICPeriod = 10

            IC = HistoricalIC.tail(min(ICPeriod, HistoricalIC.shape[0]))

            if smartmode == 'IRSolver':
                mat = nlg.inv(np.mat(IC.cov()))                     
                weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
                weight = np.array(weight.reshape(len(weight),))[0]
                return weight.tolist()
            
