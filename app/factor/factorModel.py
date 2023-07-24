from collections import Counter
import numpy.linalg as nlg 
import numpy as np
import pandas as pd
from multiprocessing import Pool
from app.factor.api.SelectData import SelectFromMongo
from app.factor.api.SelectFactorData import GetFactorFromMongoDB
import threading
import time
import datetime
import warnings
from collections import defaultdict
import math
import numpy as np
from sklearn.preprocessing import minmax_scale

class factorModel:

    def __init__(self):
        self.groupnum = 10         # 股票分组数
        self.trade_freq = 'm'      # 交易频率 "m" or "w"b
        self.end = '20230720'      # 因子分析结束日期
        self.start = '20201201' #hardcode this
        self.factor_name_lst = ['decay_panic', 'aShareholderZ', 'apbSkew', 'stopQ', 'aiDaNp30', 'sumRelatedCorp1Y', 'FlowerHidInForest']
        
        self.universe_index = ['000852.SH', '000905.SH', '000300.SH', '399303.SZ']
        self.universe = []             # 股票池列表
        self.hold_period = {}          # 历史持有期，字典中的键为调仓日、值为持有期交易日列表（按升序排序的区间交易日）
        self.bt_tradedate = []         # 调仓日
        self.pre_bt_tradedate = []     # 调仓日的前一个交易日
        self.Facor_IsAscending = False # 因子值默认为降序
        self.factor_api = GetFactorFromMongoDB()
        self.allfactorname_lst = self.factor_api.GetAllFactorName()
        self.lock = threading.Lock()
        self.stkapi = SelectFromMongo()

        self.factorWeightMode = 'smart'
        self.factorCategories = [1, 1, 2]
        self.factorWeightModeParams = 'Correlation'
        self.ICDecayHalfLife = 30
        self.EvalPeriod = 30
        self.benchmark = '000905.SH'

        self.rankLowestFirst = "0"

    def getData(self):

        warnings.filterwarnings('ignore')

        '''
        get_val:
            bt_tradedate: 调仓日列表 list   如 [20230103, 20230121, ...]
            hold_period: 持有期字典   key 为交易日列表值（首月第一个交易日）value 为 list，值为此月交易日、
            pre_bt_tradedate: 调仓日前一天列表 list
            universe: 股票池
        
        '''
        def get_val(bt_tradedate:list, hold_period:dict, pre_bt_tradedate:list, 
                universe:list, universe_index:list, factor_name_lst:list, item:int, all_period_data:dict
                , stk_api:SelectFromMongo, factor_api: GetFactorFromMongoDB, delete_list:list):
            if bt_tradedate[item] not in hold_period:
                return
            pre_day = pre_bt_tradedate[item]
            cur_day = bt_tradedate[item]
            hold_datelst = hold_period[cur_day]
            universe = SetUniverse(pre_day)
            start = time.time()
            df_f2= GetFactorFromDB(universe, factor_name_lst, pre_day, stk_api, factor_api) # df_f2: dataframe, index is ts_code, column is factors, period is monthly
            #print('getDataFrom db', time.time() - start)
            start = time.time()
            stock_daily_profit = GetHoldPeriodPriceRet(hold_datelst, df_f2, stk_api)
            #print('GetHoldPeriodPricedRet', time.time() - start)

            self.daily_profit = pd.concat([self.daily_profit, stock_daily_profit], axis=0, ignore_index=True)
            # for val in self.daily_profit['profit_daily']:
            #     if math.isnan(val):
            #         print('has nan in daily')
            #         exit(1)
            stk_holdret_monthly = StockHoldRet(stock_daily_profit) # monthly profit, use for analysis

            df_f2 = pd.merge(df_f2, stk_holdret_monthly, left_index=True, right_index=True)
            #print(len(df_f2))
        
            
            for val in df_f2['profit_month']:
                if math.isnan(val):
                    print('DF-f2 has nan')
                    exit(1)
            all_period_data[bt_tradedate[item]] = df_f2

            
        def GetSuspendInfo(trade_date, db_api, c=1):
            result_s = []  # 停牌股票列表

            df = db_api.Stock_Suspend([], trade_date)

            if not df.empty:
                df = df[['ts_code', 'suspend_type']]
                if c == 1:
                    df_temp = df[df['suspend_type'] == 'S']  # 返回停牌股票信息
                    result_s = list(df_temp['ts_code'])
                else:
                    df_temp = df[df['suspend_type'] == 'R']  # 返回复牌股票信息
                    result_s = list(df_temp['ts_code'])
            return result_s
        
        # 退市股票信息(最近200天即将退市的股票列表)
        def delst_code_lst(date, db_api:SelectFromMongo):
            result_lst = []
            df = db_api.GetStockStatus([], 'D')

            if not df.empty:
                temp_df = df[['ts_code', 'delist_date']]
                temp_df.reset_index(drop=True, inplace=True)
                for idx in temp_df.index:
                    temp_code = temp_df.loc[idx, 'ts_code']
                    tmep_date = temp_df.loc[idx, 'delist_date']
                    d1 = datetime.datetime.strptime(str(date), "%Y%m%d").date()
                    d2 = datetime.datetime.strptime(tmep_date, "%Y%m%d").date()

                    if (d2 - d1).days <= 200 and (d2 - d1).days >= -200:
                        result_lst.append(temp_code)
            return result_lst
            
        # 因子标准化
        def StandarDize(df):
            '''
            :param df: DataFrame 行索引为股票代码 列为因子值列，因子值列支持多列（多个因子）
            :return:   DataFrame 行索引为股票代码，列为因子值列
            '''
            df_factor = df.copy()
            df_factor_after = pd.DataFrame()  # DataFrame 行索引为股票代码，列为因子值列

            for idx in df_factor.columns:
                temp_df = df_factor[[idx]]
                z_m = temp_df[idx].mean()
                z_s = temp_df[idx].std()
                temp_df[idx] = (temp_df[idx] - z_m) / z_s
                df_factor_after[idx] = temp_df[idx]
            return df_factor_after

        # 因子去极值
        def WinSorizeNewMethod(df, winsorize_max_num=5, ):
            '''
            :param df: DataFrame 行索引为股票代码 列为因子值列，因子值列支持多列（多个因子）
            :param n_draw: int 正态分布去极值的迭代次数，默认为5次
            :return:       DataFrame   行索引为股票代码，列为因子值列
            '''
            df_factor = df.copy()
            df_factor_after = pd.DataFrame()  # 经过去极值处理后的因子值 行索引为股票代码，列为因子值列

            # 遍历因子去极值
            for clt in df_factor.columns:
                temp_df = df_factor[[clt]]  # DataFrame 行索引为股票代码 列为需要去极值的因子值
                count_num = 0  # 初始化单个因子去极值的次数

                while True:
                    m = temp_df[clt].mean()
                    s = temp_df[clt].std()
                    up = m + 3 * s
                    down = m - 3 * s

                    temp_df[temp_df[clt] > up] = up
                    temp_df[temp_df[clt] < down] = down

                    count_num += 1
                    if count_num >= winsorize_max_num:
                        break

                df_factor_after[clt] = temp_df[clt]
            return df_factor_after

        #获取某一天股票因子数据
        def GetFactorFromDB(universe:list, factor_name_lst:list, data_date:int, stk_api:SelectFromMongo, factor_api:GetFactorFromMongoDB, Factor_IsAscending=True):
            factor_df = factor_api.GetStockFactor(data_date, universe, factor_name_lst)
            factor_df.fillna(0, inplace=True)
            factor_df = StandarDize(WinSorizeNewMethod(factor_df))
            cls_name = list(factor_df.columns)[0]
            factor_df.sort_index(ascending=True)
            return factor_df

        # 从数据库获取持有期股票每日收益率数据
        def GetHoldPeriodPriceRet(hold_datelst,stock_factor_lst:pd.DataFrame, stk_api:SelectFromMongo):

            '''
            :param hold_datelst:      持有期交易日列表
            :param factor_group_dict: 因子dataframe index 为ts_code column 为因子名
            :return:值为股票每天收益率 DataFrame 'trade_date','ts_code','chgPct'
            '''
            s_date = hold_datelst[0]    # 持有的第一个交易日（调仓日）
            e_date = hold_datelst[-1]   #
            ret_group_dict = {}

            stock_list = list(stock_factor_lst.index)
            his_data = stk_api.MktEqudAdjGet(stock_list, [], s_date, e_date, ['trade_date','ts_code','pct_chg'], is_adj=False)

            his_data.dropna(inplace=True)

            his_data['profit_daily'] = his_data['pct_chg'] / 100
            his_data = his_data[['trade_date', 'ts_code', 'profit_daily']]

            return his_data
        
        def StockHoldRet(stock_daily_profit: pd.DataFrame) -> pd.DataFrame:
            # transform to cumulative product
            stock_daily_profit['profit_daily_plus_one'] = stock_daily_profit['profit_daily'] + 1
            
            monthly_profit = stock_daily_profit.groupby('ts_code')['profit_daily_plus_one'].prod() - 1

            res = monthly_profit.reset_index() 
            res.set_index('ts_code', inplace=True)

            res.columns = ['profit_month']

            return res

        # transform the data from a dict of dataframe to a 3D array
        def transform_data(all_period_data:dict, factor_name_lst:list):
            start = time.time()
            column_lst = factor_name_lst
            equity_idx_monthly_equity_returns = {}
            monthly_equity_returns = {}
            equity_idx_monthly_factor_score = {}
            monthly_factor_score = {}
            for month in all_period_data:
                monthly_equity_returns[month] = list(all_period_data[month]['profit_month'])
                for f in factor_name_lst:
                    if f not in monthly_factor_score:
                        monthly_factor_score[f] = {month : []}
                    monthly_factor_score[f][month] = list(all_period_data[month][f])
                for row in all_period_data[month].itertuples(name='DataFrame'):
                    stock_name = row.Index
                    if stock_name not in equity_idx_monthly_equity_returns:
                        equity_idx_monthly_equity_returns[stock_name] = {}
                    equity_idx_monthly_equity_returns[stock_name][month] = row.profit_month

                    if stock_name not in equity_idx_monthly_factor_score:
                        equity_idx_monthly_factor_score[stock_name] = {month : []}
                        for factor in factor_name_lst:
                            equity_idx_monthly_factor_score[stock_name][month].append(row[row._fields.index(factor)])
                    else:
                        equity_idx_monthly_factor_score[stock_name][month] = []
                        for factor in factor_name_lst:
                            equity_idx_monthly_factor_score[stock_name][month].append(row[row._fields.index(factor)])
            #end = time.time()
            #print('time', time.time() - start)
            return equity_idx_monthly_equity_returns, monthly_equity_returns, monthly_factor_score, equity_idx_monthly_factor_score

        def TradeDateDeal():
            his_trade_df = self.stkapi.trade_cal(self.start, self.end)
            his_trade_df = his_trade_df[his_trade_df['is_open'] == 1]
            historyTradeDate = list(his_trade_df['cal_date'])
            pre_historyTradeDate = list(his_trade_df['pretrade_date'])

            bt_TradeDate = []          # 调仓日
            pre_bt_TradeDate = []      # 调仓日上一个交易日

            #生成调仓日列表
            if self.trade_freq == 'm':
                for idx in range(1, len(historyTradeDate)):
                    c_dt = historyTradeDate[idx]
                    p_dt = historyTradeDate[idx-1]
                    if c_dt[4:6] != p_dt[4:6]:
                        bt_TradeDate.append(c_dt)
                        pre_bt_TradeDate.append(pre_historyTradeDate[idx])
            elif self.trade_freq == 'w':
                for idx in range(0, len(historyTradeDate)):
                    c_dt = historyTradeDate[idx]
                    if datetime.datetime.strptime(c_dt, "%Y%m%d").weekday() + 1 == 1:
                        bt_TradeDate.append(c_dt)
                        pre_bt_TradeDate.append(pre_historyTradeDate[idx])
            self.bt_tradedate = bt_TradeDate
            self.pre_bt_tradedate = pre_bt_TradeDate
            for item in range(0, len(bt_TradeDate)):
                s_period = bt_TradeDate[item]
                if item != len(bt_TradeDate) - 1:
                    e_period = bt_TradeDate[item + 1]
                else:
                    e_period = historyTradeDate[-1]
                s_idx = historyTradeDate.index(s_period)
                e_idx = historyTradeDate.index(e_period)
                if len(historyTradeDate[s_idx:e_idx+1]) >= 4:
                    self.hold_period[s_period] = historyTradeDate[s_idx:e_idx+1]
        
        def SetUniverse(data_date):
            if not self.universe_index:
                return
            else:
                temp_uni = []
                for idx in self.universe_index:
                    index_con_df = self.stkapi.index_weight(idx, [data_date])
                    if index_con_df.empty:
                        print('select stock in %s' % idx, 'but the index_con_df is empty')
                    index_con_df = list(index_con_df['con_code'])
                    temp_uni.extend(index_con_df)
                temp_uni_0 = list(set(temp_uni))
                susp_lst = GetSuspendInfo(data_date, self.stkapi)
                delist = delst_code_lst(data_date, self.stkapi)
                black_lst = susp_lst = delist
                temp_uni_0 = [x for x in temp_uni_0 if x not in black_lst]
                return temp_uni_0 

        def BenchmarkDailyPct():
            # 指数历史涨跌幅
            
            index_df = self.stkapi.MktIndexGet([self.benchmark], [], beginDate=self.start, endDate=self.end,field=["trade_date", "pct_chg"])
            if not index_df.empty:
                index_df['pct_chg'] = index_df['pct_chg'] / 100
                index_df = index_df[["trade_date", "pct_chg"]]
                index_df.sort_values(by="trade_date",ascending=True,inplace=True)
            return index_df
        
        def run():
            t0 = time.time()
            TradeDateDeal()
            all_period_data = {}
            IC_info_lst = []
            thread_list = []
            delete_list = []
            self.daily_profit = pd.DataFrame(columns=['trade_data', 'ts_code', 'profit_daily'])
            start = time.time()
            for item in range(len(self.bt_tradedate)):
                # process = threading.Thread(target=self.get_val, args=(item, all_period_data, IC_info_lst))
                get_val(self.bt_tradedate, self.hold_period, self.pre_bt_tradedate, 
                        self.universe, self.universe_index, self.factor_name_lst, 
                        item, all_period_data, self.stkapi, self.factor_api, delete_list)
                
            #print('get_val_time', time.time() - start)
            # for val in all_period_data:
            #     for row in all_period_data[val].itertuples():
            #         if math.isnan(row.profit_month):
            #             print('has non before enter func')
            #             exit(1)
            # delete_list = [item for sublist in delete_list for item in sublist]
            
            equity_idx_monthly_equity_returns, monthly_equity_returns, monthly_factor_score, equity_idx_monthly_factor_score = transform_data(all_period_data, self.factor_name_lst)

            return equity_idx_monthly_equity_returns, monthly_equity_returns, monthly_factor_score, equity_idx_monthly_factor_score, self.daily_profit, BenchmarkDailyPct()
    
        return run()

    def calcIC(self, dateList, scoreList, returnList):

        '''
            函数: calcIC (non-async)   TODO output method
            -------
            1) 摘要: 此函数用于计算单个因子的历史IC值
            2) 函数输入
                - dateList [REQUIRED]
                    每次算分的日期 (list)
                - scoreList [REQUIRED]
                    - list of list of scores, each list is a list of scores for a given date (list[list[float]])
                - returnList [REQUIRED]
                    - list of list of scores, each list is a list of scores for a given date (list[list[float]])
            3) 输出: (list, list[float])
            4) 可控变量:
                N/A
            5) 案例
                N/A
        '''
        
        ICList = []

        for i in dateList:
            scores = scoreList[i]
            returns = returnList[i]
            # nanIdx = np.where((np.isnan(returns)))[0]
            # if nanIdx.size > 0:
            #     for j in nanIdx:
            #         scores[j] = 0
            #         returns[j] = 0

            ICList.append(np.corrcoef(scores, returns)[0,1])

        if len(ICList) != len(dateList):
            raise Exception('Error: ICList Length != dateList Length | ERROR!')

        return dateList, ICList
    
    def calcFactorWeights(self, mode:str, listOfFactors:list[str], listOfCategories:list[str] = [], HistoricalIC:list[list[float]] = [], smartmode = 'IRSolver', equityScore = None) -> list:

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
                    - 如何使用历史IC智能计算, 可以优化最大化IR(IRsolver), 计算因子半衰后的最大化IR(IRSolverWithDecay)
            3) 输出: list[float], 每个因子的权重
            4) 可控变量:
                - ICPeriod: int, 选择用于计算历史IC的时间长度
                - half_life: int, 历史IC数据半衰周期
            5) 案例
                - 等权重模式 calcWeights('equal', ['a', 'b', 'c', 'd', 'e', 'f'])
                - 大类等权重模式 calcWeights('category', ['a', 'b', 'c', 'd', 'e', 'f'], listOfCategories = [1, 1, 2, 3, 1, 4])
                - 历史IC智能计算模式 calcWeights('smart', ['a', 'b', 'c', 'd', 'e', 'f'], HistoricalIC = {pd.DataFrame Object}, smartmode='IRSolver')
        '''

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


            IC = HistoricalIC.tail(min(self.EvalPeriod, HistoricalIC.shape[0]))
            

            if smartmode == 'IRSolver':
                mat = nlg.inv(np.cov(IC, rowvar=False))                 
                weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
                weight = np.array(weight.reshape(len(weight),))[0]
                weight = weight.tolist()
            
            elif smartmode == 'IRSolverWithDecay':

                #additional - decay IC values

                actualPeriod = min(self.EvalPeriod, IC.shape[0])

                weights = [2**((i-actualPeriod-1)/(self.ICDecayHalfLife))/np.sum([2**((-j)/self.ICDecayHalfLife) for j in range(1, actualPeriod+1)]) for i in range(1, actualPeriod+1)]
                
                newIC = IC.copy()
                
                for i in newIC:
                    newIC.loc[:,i] = newIC.loc[:,i] * weights

                mat = nlg.inv(np.cov(newIC, rowvar=False))                           
                weight = mat*np.mat(newIC.mean()).reshape(len(mat),1)
                weight = np.array(weight.reshape(len(weight),))[0]
                weight = weight.tolist()

            elif smartmode == 'Correlation': #best
                
                covar = np.cov(IC, rowvar=False)
                corr = np.corrcoef(equityScore, rowvar=False)
                D = np.diag(np.sqrt(np.diag(covar)))
                covar = D @ nlg.inv(corr) @ D
                mat = nlg.inv(covar)                 
                weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
                weight = np.array(weight.reshape(len(weight),))[0]
                weight = weight.tolist()

            elif smartmode == 'CorrelationDecay':

                actualPeriod = min(self.EvalPeriod, IC.shape[0])

                weights = [2**((i-actualPeriod-1)/(self.ICDecayHalfLife))/np.sum([2**((-j)/self.ICDecayHalfLife) for j in range(1, actualPeriod+1)]) for i in range(1, actualPeriod+1)]
                
                newIC = IC.copy()
                for i in newIC:
                    newIC.loc[:,i] = newIC.loc[:,i] * weights
                covar = np.cov(newIC, rowvar=False)

                corr = np.corrcoef(equityScore, rowvar=False)
                D = np.diag(np.sqrt(np.diag(covar)))
                covar = D @ nlg.inv(corr) @ D
                mat = nlg.inv(covar)                 
                weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
                weight = np.array(weight.reshape(len(weight),))[0]
                weight = weight.tolist()
            
                
            return weight
            
    def calcEquityScore(self, equityName:str, weights:list, scoresMap:dict, month:int):

        '''
            函数: calcEquityScores (non-async)
            -------
            1) 摘要: 此函数调用calcFactorWeights函数以及因子的分数来计算个股的分数
            2) 函数输入
                - equityName [REQUIRED]
                    股票名称
                - weights [REQUIRED]
                    现在所有因子的权重
                - scores [REQUIRED]
                    每个因子对应的分数
            3) 输出: 
                - (str, float), 个股的名称和分数
            4) 可控变量:
                N/A
            5) 案例
                N/A
        '''
        if equityName in scoresMap:
            if month in scoresMap[equityName]:
                return (equityName, np.dot(weights, scoresMap[equityName][month]))
            else:
                #print(f'{equityName} NO FACTOR SCORE DATA for [MONTH {month}]')
                return (None, None)
        else:
            #print(f'{equityName} NO FACTOR SCORE DATA [ALL PERIOD]')
            return (None, None)

    def rankEquity(self, equityNameList, equityScoreList) -> list[list[str]]:

        '''
            函数: rankEquity (non-async)
            -------
            1) 摘要: 此函数用于对个股的分数进行排序
            2) 函数输入
                - equityNameList [REQUIRED]
                    股票名称列表
                - equityScoreList [REQUIRED]
                    股票分数列表
                - numOfGroups [REQUIRED]
                    分组数量
            3) 输出:
                - list[list[str]], 每个list里面是一个分组
            4) 可控变量:
                numGroups - 分成多少组
            5) 案例
                N/A
        '''

        self.groupnum = 10  # total groups
    
        equityScoreList = np.array(equityScoreList).astype(float)
        if self.rankLowestFirst == "0":
            sort_index = np.argsort(equityScoreList)[::-1]
        elif self.rankLowestFirst == "1":
            sort_index = np.argsort(equityScoreList)
            
        sortedNames = np.array(equityNameList)[sort_index]
        groups = np.array_split(sortedNames, self.groupnum)

        # #return a list of list of equities
        return [list(group) for group in groups]
    
    def calcBasketWeights(self, equityBasket:list) -> list[float]:
            #for now, equal weights
            return [1/len(equityBasket)] * len(equityBasket)

    # 计算持有期每组的组合收益率
    def EachGroupPortRet(self,all_period_data):
        '''
        :param all_period_data: 所有持有期因子分组数据的日收益率序列 dict 键为调仓日 值为dict(键为分组名、值为每组股票对应的日收益率)
        :return: dict 键为分组名 值为DataFrame "trade_date","dailyRet"
        '''
        #遍历持有期并计算分组表现

        eachgroup_show = {}

        start = time.time()

        # Create a dictionary to hold DataFrames for each group in all_period_data
        group_frames = defaultdict(list)

        for idx, trade_day in enumerate(self.bt_tradedate):
            cur_each_period = all_period_data[trade_day]  

            if idx + 1 < len(self.bt_tradedate):
                next_each_period = all_period_data.get(self.bt_tradedate[idx + 1], {})
            else:
                next_each_period = {} 
                
            if not cur_each_period: continue

            for group_name, cur_temp_df in cur_each_period.items():
                result_df = cur_temp_df.groupby('trade_date')['profit_daily'].mean().reset_index(name='dailyRet')

                if next_each_period:
                    next_temp_df = next_each_period.get(group_name, pd.DataFrame())
                    cur_hold = set(cur_temp_df['ts_code'])
                    next_hold = set(next_temp_df['ts_code'])
                    fee = len(next_hold - cur_hold) * 2 * 0.0007 / max(len(cur_hold), len(next_hold))

                    if not result_df.empty:
                        result_df.iloc[-1, result_df.columns.get_loc('dailyRet')] -= fee

                group_frames[group_name].append(result_df)

        eachgroup_show = {group_name: pd.concat(frames, ignore_index=True) for group_name, frames in group_frames.items()}

        print(f'time used is {time.time() - start}')
        
        # 计算 第一组 - 最后一组(多空对冲)
        final_group_name = "group_%s"%(self.groupnum-1)      # 最后一组的组名
        first_group_df = eachgroup_show["group_0"]           # 第一组的数据
        finally_group_df_1 = eachgroup_show[final_group_name]  # 最后一组的数组
        finally_group_df = finally_group_df_1.copy()
        finally_group_df.columns = ['trade_date','final_dailyRet']

        first_group_df.sort_values(by="trade_date",ascending=True,inplace=True)
        finally_group_df.sort_values(by="trade_date", ascending=True, inplace=True)

        long_short_df = pd.merge(first_group_df,finally_group_df,on="trade_date",how="inner")  # trade_date dailyRet final_dailyRet
        long_short_df["ret"] = long_short_df['dailyRet'] - long_short_df['final_dailyRet']
        long_short_df = long_short_df[['trade_date','ret']]
        long_short_df.columns = ['trade_date','dailyRet']
        # eachgroup_show["group_0"+"-"+final_group_name] = long_short_df
        eachgroup_show["longshort_hedge"] = long_short_df

        # 处理在遍历调仓日时，调仓日那天重复计算组合收益率的问题
        for key in eachgroup_show:
            df_value = eachgroup_show[key]       # DataFrame 'trade_date','dailyRet'
            df_value.drop_duplicates(subset="trade_date",keep="first",inplace=True)
            eachgroup_show[key] = df_value

        return eachgroup_show

    # 历史累计收益率序列和策略评价指标
    def HistoryAccuRetAndIndicator(self,group_name,eachgroup_dailyret):
        '''
        :param group_name 分组名称
        :param eachgroup_dailyret: DataFrame trade_date dailyRet
        :return: temp_df DataFrame 行索引为交易日（升序），唯一列为累计净值 ； indicator_lst [] 依次为年化收益率、夏普比率、最大回撤
        '''
        temp_df = eachgroup_dailyret.copy()
        temp_df.sort_values(by="trade_date",ascending=True,inplace=True)  # DataFrame trade_date dailyRet
        temp_df.reset_index(drop=True,inplace=True)

        # 计算每日净值
        temp_df["net_values"] = np.nan  # DataFrame trade_date dailyRet net_values
        for h in temp_df.index:
            if h == 0:
                temp_df.loc[h, "net_values"] = 1 + temp_df.loc[h, "dailyRet"]
            else:
                temp_df.loc[h, "net_values"] = temp_df.loc[h - 1, "net_values"] + temp_df.loc[h, "dailyRet"]

        temp_df = temp_df[['trade_date', 'dailyRet', 'net_values']]
        cur_date = temp_df.loc[0,"trade_date"]
        cur_date = str(cur_date)
        init_date = datetime.datetime(int(cur_date[0:4]),int(cur_date[4:6]),int(cur_date[6:8])) - datetime.timedelta(days=1)
        init_date = init_date.strftime("%Y%m%d")
        init_df = pd.DataFrame([[init_date,0.0,1.0]],index=[0],columns=['trade_date', 'dailyRet', 'net_values'])
        temp_df = pd.concat([init_df,temp_df],axis=0)
        temp_df.reset_index(drop=True, inplace=True)

        # 计算评价指标
        year_ret = (list(temp_df["net_values"])[-1]-1) / (len(list(temp_df["net_values"]))-1) * 242  # 年化收益率
        sharpe_ratio = temp_df["dailyRet"].mean() / temp_df["dailyRet"].std() * math.sqrt(242)       # 夏普比率

        temp_df["max_drawn"] = np.nan
        for n in temp_df.index:
            temp_df.loc[n, "max_drawn"] = temp_df.loc[n, "net_values"] - temp_df.loc[0:n + 1, "net_values"].max()
        max_drawn = temp_df["max_drawn"].min()                                                       # 最大回撤

        year_ret = round(year_ret*100,4)
        sharpe_ratio = round(sharpe_ratio, 4)
        max_drawn = round(max_drawn * 100, 4)
        indicator_lst = [group_name,year_ret,sharpe_ratio,max_drawn]
        temp_df = temp_df[["trade_date","net_values"]]
        temp_df.set_index("trade_date",drop=True,inplace=True)
        temp_df.columns = [group_name]
        return temp_df,indicator_lst 

    # 策略相对基准指数的表现和评价指标分析
    def HistoryAlphaAndIndicator(self,group_name,eachgroup_dailyret,benchmark_df):
        '''
        :param group_name 分组名称
        :param eachgroup_dailyret: DataFrame trade_date dailyRet
        :param benchmark_df DataFrame "trade_date", "pct_chg"
        :return: temp_df DataFrame 行索引为交易日（升序），唯一列为累计净值 ； indicator_lst [] 依次为年化alpha、超额夏普比率、超额最大回撤
        '''
        if benchmark_df.empty:
            return pd.DataFrame(),[]

        merge_df = pd.merge(eachgroup_dailyret,benchmark_df,on="trade_date",how="inner")  # trade_date dailyRet pct_chg
        merge_df.dropna(inplace=True)
        merge_df["dailyAlpha"] = merge_df['dailyRet'] - merge_df['pct_chg']
        merge_df = merge_df[['trade_date','dailyAlpha']]
        merge_df.sort_values(by="trade_date",ascending=True,inplace=True)
        merge_df.set_index("trade_date",drop=True,inplace=True)
        merge_df = merge_df.cumsum(axis=0)
        merge_df.columns = ['accuAlpha']

        # 年化alpha
        year_alpha = list(merge_df["accuAlpha"])[-1] / len(list(merge_df["accuAlpha"])) * 242

        # alpha 最大回撤
        df_temp_1 = merge_df.copy()
        df_temp_1.index = range(len(df_temp_1["accuAlpha"]))
        df_temp_1["max_drawn"] = np.nan
        for n in df_temp_1.index:
            df_temp_1.loc[n, "max_drawn"] = (df_temp_1.loc[n, "accuAlpha"] - df_temp_1.loc[0:n + 1, "accuAlpha"].max())
        max_drawn = df_temp_1["max_drawn"].min()

        # 计算alpha的收益回撤比
        calmar_ratio = round(year_alpha, 4) / abs(round(max_drawn, 4))

        # 返回值
        cur_date = list(merge_df.index)[0]
        init_date = datetime.datetime(int(cur_date[0:4]), int(cur_date[4:6]), int(cur_date[6:8])) - datetime.timedelta(days=1)
        init_date = init_date.strftime("%Y%m%d")
        init_df = pd.DataFrame([0], index=[init_date], columns=['accuAlpha'])
        result_df = pd.concat([init_df,merge_df],axis=0)
        result_df.columns = [group_name]
        year_alpha = round(year_alpha*100 ,4)
        max_drawn = round(max_drawn*100, 4)
        calmar_ratio = round(calmar_ratio, 4)
        indicator_lst = [group_name,year_alpha,max_drawn,calmar_ratio]
        return result_df,indicator_lst
   
       # 拿回去期间基准指数的日涨跌幅数据
    

    def run(self):
        '''
        Return Object

        1. Equity_Idx_Monthly_Equity_Returns
            Type: Dict[list]
            {Stock1 : {Month1 : return, Month2 : return, etc...}}
        
        2. Monthly_Equity_Returns
            Type: Dict[list]
            {Month1 : [Stock1Return, Stock2Return, Stock3Return, etc...], Month2 : []...}

        3. Equity_Idx_Monthly_Factor_Score
            Type: Dict[Dict[list]]
            {Stock1 : {Month1 : [Stock1Factor1, Stock1Factor2, Stock1Factor3, etc...]}}
        
        4. Monthly_Factor_score
            Type: Dict[Dict[list]]
            {Factor1 : {Month1 : [Stock1FactorScore, Stock2FactorScore, Stock3FactorScore, etc...], Month2 : []...},}}    

        5. Daily_Equity_Returns
            Type: pd.DataFrame （这个别改）
        '''
        print(self.factorWeightMode, self.factorWeightModeParams)
        
        startTime = time.time()

        Equity_Idx_Monthly_Equity_Returns, Monthly_Equity_Returns, Monthly_Factor_Score, Equity_Idx_Monthly_Factor_Score, Daily_Equity_Returns, benchmark_dailyret= self.getData()

        currTime = time.time()
        print(f'Got data in {currTime - startTime} seconds')
        startTime = currTime

        try:    
            Daily_Equity_Returns = Daily_Equity_Returns.drop(columns=['trade_data'])
        except:
            pass

        factor_names = list(Monthly_Factor_Score.keys())
        month_names = list(Monthly_Equity_Returns.keys())
        self.month_names = month_names
        stock_names = list(Equity_Idx_Monthly_Factor_Score.keys())
        minMonths = 2
        maxMonths = 12
        groupedProfit = defaultdict(dict)
        totalIC = 0
        combinedIC = {'month': [], 'IC': [], 'cumulative': []}

        if self.factorWeightMode != 'smart':
            if self.factorWeightMode == 'equal':
                factorWeights = self.calcFactorWeights(self.factorWeightMode, factor_names)
            elif self.factorWeightMode == 'category':
                factorWeights = self.calcFactorWeights(self.factorWeightMode, factor_names, self.factorCategories)
        elif self.factorWeightMode == 'smart':
            ICList = []

            for factor in factor_names:
                ICList.append(self.calcIC(month_names, Monthly_Factor_Score[factor], Monthly_Equity_Returns)[1])
            ICList = pd.DataFrame(ICList).T

            ICList.index = month_names
            ICList.columns = factor_names
        
        for month in range((minMonths if self.factorWeightMode == 'smart' else 0), len(month_names)):

            nameList, scoreList = [], []

            if self.factorWeightMode == 'smart':
                #get current IC
                currList = ICList.loc[ICList.index[ICList.index <= month_names[month]]]
                if currList.shape[0] > maxMonths:
                    currList = currList.iloc[-maxMonths:]
                    
                if self.factorWeightModeParams in ['Correlation', 'CorrelationDecay']:

                    #get current monthly score
                    res = defaultdict(list)
                    endMonth = month_names[month]
                    if month > maxMonths:
                        startMonth = month_names[month - maxMonths]
                    else:
                        startMonth = month_names[0]

                    for factor in Monthly_Factor_Score:
                        for date in Monthly_Factor_Score[factor]:
                            #print(date, int(date) >= int(startMonth), int(date) <= int(endMonth))
                            if int(date) >= int(startMonth) and int(date) <= int(endMonth):
                                res[factor] += Monthly_Factor_Score[factor][date]

                    res = pd.DataFrame(res)
                    factorWeights = self.calcFactorWeights(self.factorWeightMode, factor_names, HistoricalIC =currList, smartmode=self.factorWeightModeParams, equityScore=res)
                else:
                    
                    factorWeights = self.calcFactorWeights(self.factorWeightMode, factor_names, HistoricalIC=currList, smartmode=self.factorWeightModeParams)
            else:
                if self.factorWeightMode == 'equal':
                    factorWeights = self.calcFactorWeights(self.factorWeightMode, factor_names)
                elif self.factorWeightMode == 'category':
                    factorWeights = self.calcFactorWeights(self.factorWeightMode, factor_names, self.factorCategories)

            #print(f'Weights - {factorWeights}')
            # print(factorWeights)
            for name in stock_names:
                name, score = self.calcEquityScore(name, factorWeights, Equity_Idx_Monthly_Factor_Score, month_names[month])
                if name:
                    nameList.append(name)
                    scoreList.append(score)
            
            equityGroups = self.rankEquity(nameList, scoreList)

            try:
                daily_returns_this_month = Daily_Equity_Returns[(Daily_Equity_Returns['trade_date'] >= month_names[month]) 
                                    & (Daily_Equity_Returns['trade_date'] < month_names[month+1])]
            except:
                daily_returns_this_month = Daily_Equity_Returns[Daily_Equity_Returns['trade_date'] >= month_names[month]]
            
            for i, group in enumerate(equityGroups):
                # Filter df where 'ts_code' is in the current group
                df_group = daily_returns_this_month[daily_returns_this_month['ts_code'].isin(group)]
                groupedProfit[month_names[month]][f'group_{i}'] = df_group.reset_index(drop=True)  
            
            returnArr = []
            for stock in nameList:
                returnArr.append(Equity_Idx_Monthly_Equity_Returns[stock][month_names[month]])

            currIC = np.corrcoef(returnArr, scoreList)[0,1]
            
            totalIC += currIC
            combinedIC['month'].append(month_names[month])
            combinedIC['IC'].append(currIC)
            combinedIC['cumulative'].append(totalIC)

        currTime = time.time()
        print(f'Processed in {currTime - startTime} seconds')
        startTime = currTime

        '''
        dict - groupedProfit
            key = 挪仓日 (6个key)
            value = list[]
                lst里面有10个df, 每个df有三个col, 叫ts_code profit trade_date
            
            {month1 : {group1:df, group2:df, group3:df}}
        '''
        df_group_net = pd.DataFrame()   # 回测期间分组净值曲线 output
        indicator_lst = []              # 回测期间分组的回测指标
        df_group_alpha = pd.DataFrame()   # 回测期间分组的alpha曲线 new output
        alpha_indicator_lst = []          # 回测期间分组的alpha回测指标
        
        group_dailyret_dict = self.EachGroupPortRet(groupedProfit)

        currTime = time.time()
        print(f'EachGroupPortRet in {currTime - startTime} seconds')
        startTime = currTime

        for name in group_dailyret_dict:
            # 策略回测指标
            df_group_indicator,indi_lst = self.HistoryAccuRetAndIndicator(name,group_dailyret_dict[name])
            df_group_net[name] = df_group_indicator[name]
            indicator_lst.append(indi_lst)
            
            #策略alpha
            group_alpha, alpha_indi_lst = self.HistoryAlphaAndIndicator(name, group_dailyret_dict[name], benchmark_dailyret)
            df_group_alpha[name] = group_alpha[name]
            alpha_indicator_lst.append(alpha_indi_lst)
            
        df_bt_indicator = pd.DataFrame(indicator_lst,index = range(len(indicator_lst)),columns=["group","年化收益率","夏普比率","最大回撤"]) # 回测期间分组的回测指标 output
        df_bt_alpha_indicator = pd.DataFrame(alpha_indicator_lst,index=range(len(alpha_indicator_lst)),columns=["group","年化超额收益率","超额最大回撤","calmar"])  # 超额评价指标 new output

        currTime = time.time()
        print(f'Last Section in {currTime - startTime} seconds')
        startTime = currTime
        
        return combinedIC, df_group_net, df_group_alpha, df_bt_indicator, df_bt_alpha_indicator