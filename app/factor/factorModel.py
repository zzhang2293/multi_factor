from collections import Counter
import numpy.linalg as nlg 
import numpy as np
import pandas as pd
from multiprocessing import Pool
from api.SelectData import SelectFromMongo
from api.SelectFactorData import GetFactorFromMongoDB
import threading
import time
import datetime
import warnings

class factorModel:

    def __init__(self):
        self.groupnum = 10         # 股票分组数
        self.trade_freq = 'm'      # 交易频率 "m" or "w"
        self.end = '20230630'      # 因子分析结束日期
        self.start = '20221215'
        self.factor_name_lst = ['Analyst_factor', 'daizhuerjiu', 'tps_sps']
        self.universe_index = ['000852.SH', '000905.SH', '000300.SH', '399303.SZ']
        self.universe = []             # 股票池列表
        self.hold_period = {}          # 历史持有期，字典中的键为调仓日、值为持有期交易日列表（按升序排序的区间交易日）
        self.bt_tradedate = []         # 调仓日
        self.pre_bt_tradedate = []     # 调仓日的前一个交易日
        self.Facor_IsAscending = False # 因子值默认为降序
        self.factor_api = GetFactorFromMongoDB()
        self.allfactorname_lstv = self.factor_api.GetAllFactorName()
        self.lock = threading.Lock()
        self.stkapi = SelectFromMongo()

        self.factorWeightMode = 'smart'
        self.factorCategories = [1, 1, 2]
        self.factorWeightModeParams = 'IRSolver'
        self.ICDecayHalfLife = 2
        self.ICEvalPeriod = 10

    def getData(self):

        '''
        Return Type

        1. 3D array of factor scores - list[list[list[float]]]
            - dim 1: factors
            - dim 2: dates
            - dim 3: stocks
        2. map of scores 
        2. 2D array of daily returns - list[list[float]]
            - dim 1: dates
            - dim 2: stocks
        3. 1D array of dates - list[str]
            - 和 1的dim2 还有 2的dim1 顺序对应
        4. 1D array of stock names - list[str]
            - 和 1的dim3 还有 2的dim2 顺序对应
        5. 1D array of factor names - list[str]
            - 和 1的dim1 顺序对应    
        '''

        warnings.filterwarnings('ignore')

        def get_val(bt_tradedate:list, hold_period:dict, pre_bt_tradedate:list, 
                universe:list, universe_index:list, factor_name_lst:list, item:int, all_period_data:dict
                , stk_api:SelectFromMongo, factor_api: GetFactorFromMongoDB, delete_list:list):
            if bt_tradedate[item] not in hold_period:
                return
            
            pre_day = pre_bt_tradedate[item]
            cur_day = bt_tradedate[item]
            hold_datelst = hold_period[cur_day]
            df_f2= GetFactorFromDB(universe, factor_name_lst, pre_day, stk_api, factor_api) # df_f2: dataframe, index is ts_code, column is factors, period is monthly
            need_to_delete1 = delst_code_lst(pre_day, stk_api)
            need_to_delete2 = GetSuspendInfo(pre_day, stk_api)
            delete_list.append(need_to_delete1)
            delete_list.append(need_to_delete2)
            stock_daily_profit = GetHoldPeriodPriceRet(hold_datelst, df_f2, stk_api)
            stk_holdret_monthly = StockHoldRet(stock_daily_profit) # monthly profit, use for analysis
            df_f2["profit_month"] = stk_holdret_monthly["profit_month"]
            all_period_data[item] = df_f2

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
            his_data = stk_api.MktEqudAdjGet(stock_list, [], s_date, e_date, ['trade_date','ts_code','close','pre_close'])
            his_data.dropna(inplace=True)
            his_data['profit_daily'] = his_data['close'] / his_data['pre_close'] - 1
            his_data = his_data[['trade_date', 'ts_code', 'profit_daily']]
            return his_data

        #持有期分层组合中每只股票的持有收益率计算
        def StockHoldRet(stock_daily_profit:pd.DataFrame):
            '''
            :param stkdailyret_group_dict: dict 键为分组名 值为DataFrame 持有期分组的每只股票的每日收益率
            :return: dict 健为分层名，值为DataFrame(行索引为股票代码，唯一列为股票区间累计收益率)
            '''

            res = stock_daily_profit.groupby('ts_code')['profit_daily'].sum().to_frame()
            res.columns = ['profit_month']
            return res

        # transform the data from a dict of dataframe to a 3D array
        def transform_data(all_period_data:dict, factor_name_lst:list):
            analysis_3D_list_label = []   # need a 3d list, just convert all_period_data
            for factor in factor_name_lst:
                temp_factor = []
                for month in all_period_data:
                    this_month_stocks_this_factor = list(all_period_data[month][factor])
                    temp_factor.append(this_month_stocks_this_factor)
                analysis_3D_list_label.append(temp_factor)
            analysis_2D_monthly_return = []
            for month in all_period_data:
                analysis_2D_monthly_return.append(list(all_period_data[month]['profit_month']))
            #print(analysis_3D_list_label)
            #print(analysis_2D_monthly_return)

            factor_map = {i:[] for i in all_period_data[0].index.tolist()}

            for df in all_period_data.values():
                for ticker in df.index.tolist():
                    factor_map[ticker].append(df.loc[ticker, factor_name_lst])           

            return analysis_3D_list_label, analysis_2D_monthly_return, all_period_data[0].index.tolist(), factor_map

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
        
        def run():
            t0 = time.time()
            TradeDateDeal()
            #print(self.hold_period)
            all_period_data = {}
            IC_info_lst = []
            thread_list = []
            delete_list = []
            #print(self.bt_tradedate)
            for item in range(len(self.bt_tradedate)):
                # process = threading.Thread(target=self.get_val, args=(item, all_period_data, IC_info_lst))
                get_val(self.bt_tradedate, self.hold_period, self.pre_bt_tradedate, 
                        self.universe, self.universe_index, self.factor_name_lst, 
                        item, all_period_data, self.stkapi, self.factor_api, delete_list)
            delete_list = [item for sublist in delete_list for item in sublist]
            for day in all_period_data:
                #print(all_period_data[day])
                new_data = all_period_data[day].drop(delete_list, errors='ignore')
                all_period_data[day] = new_data
            for day in all_period_data:
                #print(all_period_data[day])
                filtered = all_period_data[day][all_period_data[day].index.isin(list(set(all_period_data[0].index)))]
                all_period_data[day] = filtered
            factor_3D, month_profit, stock_list, factor_MAP = transform_data(all_period_data, self.factor_name_lst)


            return factor_3D, factor_MAP, month_profit, self.bt_tradedate, stock_list, self.factor_name_lst
    
        return run()
        
    def calcIC(self, dateList:list, scoreList:list[list[float]], returnList:list[list[float]]):

        '''
            函数: calcIC (non-async)
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

        if len(scoreList) != len(returnList) != len(dateList):
            raise Exception('Error: Lists Have Different Lengths')
        
        ICList = []

        for i in range(len(dateList)):
            scores = scoreList[i]
            returns = returnList[i]

            corr = np.corrcoef(scores, returns)[0,1]

            ICList.append(corr)

        if len(ICList) != len(dateList):
            raise Exception('Error: ICList Length != dateList Length | ERROR!')

        return dateList, ICList
    
    def calcFactorWeights(self, mode:str, listOfFactors:list[str], listOfCategories:list[str] = [], HistoricalIC:list[list[float]] = [], smartmode = 'IRSolver') -> list:

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
            
            if smartmode not in ['IRSolver', 'IRSolverWithDecay']:
                raise Exception('Error Smart Mode')

            IC = HistoricalIC.tail(min(self.ICEvalPeriod, HistoricalIC.shape[0]))

            if smartmode == 'IRSolver':
                mat = nlg.inv(np.mat(IC.cov()))                     
                weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
                weight = np.array(weight.reshape(len(weight),))[0]
                return weight.tolist()
            
            elif smartmode == 'IRSolverWithDecay':

                #additional - decay IC values

                self.ICEvalPeriod = min(self.ICEvalPeriod, IC.shape[0])

                weights = [2**((i-self.ICEvalPeriod-1)/(self.ICDecayHalfLife))/np.sum([2**((-j)/self.ICDecayHalfLife) for j in range(1, self.ICEvalPeriod+1)]) for i in range(1, self.ICEvalPeriod+1)]
                
                for i in IC:
                    IC.loc[:,i] = IC.loc[:,i] * weights

                mat = nlg.inv(np.mat(IC.cov()))                     
                weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
                weight = np.array(weight.reshape(len(weight),))[0]
                return weight.tolist()
            
    def calcEquityScore(self, equityName:str, weights:list, scoresMap:list, month:int):

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

        return (equityName, np.dot(weights, scoresMap[equityName][month]))

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

        self.groupnum = 10

        #use zip and numpy to make it fast
        zipped = list(zip(equityNameList, equityScoreList))
        zipped.sort(key = lambda x: x[1], reverse=True)
        zipped = np.array(zipped)
        zipped[:,1] = zipped[:,1].astype(float)
        zipped[:,1] = zipped[:,1].argsort()

        #split into groups
        groups = np.array_split(zipped, self.groupnum)

        #return a list of list of equities
        return [list(groups[i][:,0]) for i in range(self.groupnum)]
    
    def calcBasketWeights(self, equityBasket:list) -> list[float]:
            #for now, equal weights
            return [1/len(equityBasket)] * len(equityBasket)
    
    def calcBasketReturn(self, equityBasket:list, dateVal:str, dailyReturns:list):

        #get the basket weights
        basketWeights = self.calcBasketWeights(equityBasket)

        dailyEquityReturn = dailyReturns.loc[dateVal, equityBasket]

        returnVal = np.dot(basketWeights, dailyEquityReturn.T)

        return returnVal


    #计算持有期每组的组合收益率
    def EachGroupPortRet(self, all_period_data):
        eachgroup_show = {}
        for index in range(len(self.bt_tradedate)):
            cur_trade_day = self.bt_tradedate[index]
            if cur_trade_day not in all_period_data:
                continue
            cur_trade_day_data = all_period_data[cur_trade_day]
            next_trade_day_data = {}
            if index != len(self.bt_tradedate) - 1 and self.bt_tradedate[index + 1] in all_period_data:
                next_trade_day_data = all_period_data[self.bt_tradedate[index+1]]

            #iterate through the groups
            for group_name in cur_trade_day_data:
                cur_temp_df = cur_trade_day_data[group_name]
                fee = 0
                if next_trade_day_data:
                    next_temp_df = next_trade_day_data[group_name]
                    cur_hold = list(cur_temp_df['ts_code'])
                    next_hold = list(next_temp_df['ts_code'])
                    cur_stknum = len(cur_hold)
                    next_stknum = len(next_hold)
                    max_holdnum = max(cur_stknum, next_stknum)
                    diff_hold = [x for x in next_hold if x not in cur_hold]
                    fee = len(diff_hold) * 2 * 0.0007 / max_holdnum
                #计算该组每日组合收益率
                result_s = cur_temp_df.groupby('trade_date')['daily_profit'].mean()
                result_df = result_s.to_frame()
                result_df.columns = ['dailyRet']
                result_df["trade_date"] = list(result_df.index)
                result_df.reset_index(drop=True,inplace=True)
                result_df.loc[len(result_df['trade_date'])-1,"dailyRet"] = result_df.loc[len(result_df['trade_date'])-1,"dailyRet"] - fee # 扣除手续费
                result_df = result_df[["trade_date","dailyRet"]]  
                if group_name not in eachgroup_show:
                    eachgroup_show[group_name] = result_df
                else:
                    eachgroup_show[group_name] = pd.concat([eachgroup_show[group_name],result_df],axis=0) 


            # 第一组 - 最后一组   多空对冲             
            last_group_name = 'group_%s'%(self.groupnum-1)
            first_group_df = eachgroup_show['group_0']
            last_group_df_1 = eachgroup_show[last_group_name]
            last_group_df = last_group_df_1.copy()
            last_group_df.columns = ['trade_date', 'last_dailyRet']
            first_group_df.sort_values(by="trade_date",ascending=True,inplace=True)
            last_group_df.sort_values(by="trade_date", ascending=True, inplace=True)     
            long_short_df = pd.merge(first_group_df,last_group_df,on="trade_date",how="inner")  # trade_date dailyRet final_dailyRet
            long_short_df["ret"] = long_short_df['dailyRet'] - long_short_df['final_dailyRet']
            long_short_df = long_short_df[['trade_date','ret']]
            long_short_df.columns = ['trade_date','dailyRet']            
            eachgroup_show["longshort_hedge"] = long_short_df
            for key in eachgroup_show:
                df_value = eachgroup_show[key]
                df_value.drop_duplicates(subset="trade_date",keep="first",inplace=True)
                eachgroup_show[key] = df_value  
            return eachgroup_show              

    def run(self):
        
        #Step 1: Getting Info
        scores, scoresMap, returns, dates, stockNames, factorNames = self.getData()
    
        # #Step 2: Calculate Weights For Each Factor
        if self.factorWeightMode != 'smart':
            if self.factorWeightMode == 'equal':
                factorWeights = self.calcFactorWeights(self.factorWeightMode, factorNames)
            elif self.factorWeightMode == 'category':
                factorWeights = self.calcFactorWeights(self.factorWeightMode, factorNames, self.factorCategories)
        
        elif self.factorWeightMode == 'smart':
            ICList = []

            for i in range(len(factorNames)):
                ICList.append(self.calcIC(dates, scores[i], returns)[1])

            ICList = pd.DataFrame(ICList).T

            factorWeights = self.calcFactorWeights(self.factorWeightMode, factorNames, HistoricalIC=ICList, smartmode=self.factorWeightModeParams)

        #Step 3: Calculating Equity Scores
        groupedReturn, groupedAggValue = [[] * self.groupnum], [[1] * self.groupnum]

        dailyReturns = pd.DataFrame(returns)
        dailyReturns.columns = stockNames
        dailyReturns.index = dates

        finalRank = {}

        # {'date' : [['000.SZ', '0001.SZ'], [2], [3]], 'date2' : }
        
        for time in range(len(dates)):
            nameList, scoreList = [], []

            for i in range(len(stockNames)):
                name, score = self.calcEquityScore(stockNames[i], factorWeights, scoresMap, time)
                nameList.append(name)
                scoreList.append(score)

            equityGroups = self.rankEquity(nameList, scoreList)

            finalRank[dates[time]] = equityGroups

        for k, v in finalRank.items():
            print(f" for {k}, {len(v)} groups, total = {sum([len(i) for i in v])}")

        return finalRank
        

st = time.process_time()
t0 = time.time()

m = factorModel()
res = m.run()

et = time.process_time()
t1 = time.time()

print(f'CPU Time: {et-st}s')
print(f'Wall Time: {t1-t0}s')
print(f'Wait Time = {(t1-t0) - (et-st)}s')
