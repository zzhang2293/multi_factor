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
from scipy.stats import spearmanr
from app.factor.PortfolioOptimization import PortfolioOpt

class factorModel:

    def __init__(self):
        self.groupnum = 10         # 股票分组数
        self.trade_freq = 'm'      # 交易频率 "m" or "w"b
        self.end = '20230720'      # 因子分析结束日期
        self.start = '20201220' #hardcode this
        self.factor_name_lst = []
        
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

        self.factorWeightMode = 'smart'  #选因子权重的模式
        self.stockWeightMode = 'equal'   #选个股权重的模式
        self.factorCategories = [] #人工给因子分组的话会存在这里
        self.EvalPeriod = 31 # 因子权重优化最长回看周期
        self.minEvalPeriod = 4 # 因子权重优化最短回看周期
        self.benchmark = '000905.SH' # 对比标的
        self.rankLowestFirst = "0" #升序还是倒序，0为升序
        self.userDefinedFactorWeights = [] #用户自定义因子权重存在这里

        self.factorSelectMode = 'manual' # 因子选择模式
        self.factorChoosePeriod = 12 # 因子选择优化回看周期
        self.nFactors = 10 #选前n个因子
        self.equityGroupsInfo = dict() # 按调仓日分组股票名
        
    def getData(self):

        warnings.filterwarnings('ignore')

        if self.factorSelectMode == 'auto':
            self.factor_name_lst = self.allfactorname_lst

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
            start = time.time()
            stock_daily_profit = GetHoldPeriodPriceRet(hold_datelst, df_f2, stk_api)

            self.daily_profit = pd.concat([self.daily_profit, stock_daily_profit], axis=0, ignore_index=True)

            stk_holdret_monthly = StockHoldRet(stock_daily_profit) # monthly profit, use for analysis

            df_f2 = pd.merge(df_f2, stk_holdret_monthly, left_index=True, right_index=True)
        
            
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
                

            
            equity_idx_monthly_equity_returns, monthly_equity_returns, monthly_factor_score, equity_idx_monthly_factor_score = transform_data(all_period_data, self.factor_name_lst)

            factor_names = list(monthly_factor_score.keys())

            return equity_idx_monthly_equity_returns, monthly_equity_returns, monthly_factor_score, equity_idx_monthly_factor_score, self.daily_profit, BenchmarkDailyPct()
    
        return run()

    def calcIC(self, dateList, scoreList, returnList): #docs done
        '''
            函数: calcIC
            -------
            1) 摘要: 此函数用于计算单个因子的每月历史IC值
            2) 函数输入
            - dateList [必须]
                月度日期列表 (list[str])
            - scoreList [必须]
                因子对每月每个股票分数列表 (list[list[float]]), 外层索引为月度, 内层索引为个股
            - returnList [REQUIRED]
                因子对每月每个股票收益列表 (list[list[float]]), 外层索引为月度, 内层索引为个股
            3) 输出: 
            - dateList 
                月度日期列表，和输入一样，用于后续校验 (list[str])
            - ICList
                因子每月历史IC值 (list[float])
        '''
        
        ICList = []

        # 遍历每个月
        for i in dateList:
            scores = scoreList[i]
            returns = returnList[i]
            
            # 计算IC值 (np.corrcoef, [0, 1]是因为返回的是矩阵)
            ICList.append(np.corrcoef(scores, returns)[0,1])

        # 校验
        if len(ICList) != len(dateList):
            raise Exception('Error: ICList Length != dateList Length | ERROR!')

        return dateList, ICList
    
    def chooseFactors(self, IC):
        if IC.shape[0] > self.factorChoosePeriod:
            newIC = IC.iloc[-self.factorChoosePeriod:]
        else:
            newIC = IC

        

        total_map = newIC.sum().abs()

        print(newIC, total_map)

        while True:
            pass

        # chosen_factors = total_map.nlargest(IC.shape[1])

        # indices = [list(newIC.columns).index(factor) for factor in chosen_factors.index]
 
        # return list(chosen_factors.index), indices                

    def calcFactorWeights(self, mode, listOfFactors, listOfCategories = None, HistoricalIC = None, equityScore = None): #docs done

        '''
            函数: calcFactorWeights
            -------
            1) 摘要：此函数用于计算单期各因子的最优权重
            2) 函数输入
            - mode [必须]
                权重计算模式, 可以是equal(等权重), category(人工分组), smart(优化权重) (str)
            - listOfFactors [必须] 
                因子名称序列 (list[str])
            - listOfCategories [非必须, 只用于category模式] 
                人工分类序列 (list[int]) 
                    例: 如果有5个因子, 前三个一组, 后两个一组, 则输入的序列为 [1, 1, 1, 2, 2]
            - HistoricalIC [非必须, 只用于smart模式]
                因子历史IC值序列 (2维 pd.DataFrame, 横轴是因子, 纵轴是时间)
            - equityScore [非必须, 只用于smart模式]
                因子对每个股票的分数序列 (2维 pd.DataFrame, 横轴是因子, 纵轴是时间)
            3) 输出: 
            - weight 
                各因子的权重, 因子顺序和输入的listOfFactors一致 (list[float])
        '''

        if mode not in ['equal', 'category', 'smart']:
            raise Exception('Wrong Category!')

        #等权重
        if mode == 'equal':
            #create a list of size len(listOfFactors) with equal weights summing up to 1
            return [1 / len(listOfFactors)] * len(listOfFactors)
        
        #人工分组
        elif mode == 'category':
            if len(listOfCategories) != len(listOfFactors):
                raise Exception('Error Category Size')

            
            ct = Counter(listOfCategories)
            eachCatWeight = 1 / len(ct.keys())
            
            res = []

            for i in listOfCategories:
                res.append(eachCatWeight / ct[i])

            return res
        
        #优化权重
        elif mode == 'smart':
            if HistoricalIC.shape[1] != len(listOfFactors):
                raise Exception('Error IC Size')
                
            #ic的协方差
            covar = np.cov(HistoricalIC, rowvar=False)
            #每个因子对个股分数的相关性
            corr = np.corrcoef(equityScore, rowvar=False)
            D = np.diag(np.sqrt(np.diag(covar)))

            #IC协方差 = D * IC相关性 * D, D可以是任意矩阵
            #我们因为算IC的协方差 用IC相关性的时候 数据太少 所以我们在这用分数的相关性“强行替换”
            # 所以后面的协方差是 -> D * 分数相关性 * D
            covar = D @ nlg.inv(corr) @ D
            mat = nlg.inv(covar)                 
            weight = mat*np.mat(HistoricalIC.mean()).reshape(len(mat),1)
            weight = np.array(weight.reshape(len(weight),))[0]
            weight = weight.tolist()
                
            return weight
            
    def calcEquityScore(self, equityName, weights, scoresMap, month, indices = None): #docs done

        '''
            函数: calcEquityScore
            -------
            1) 摘要：此函数用于计算单期单个股票的多因子总分
            2) 函数输入
            - equityName [必须]
                个股名称 (str)
            - weights [必须] 
                计算好的因子权重 (list[int])
            - scoresMap [必须] 
                个股各月分数 (dict[dict[list]], 第一层dict的索引是个股名(equityName), 第二层dict的索引是月份名(month))
            - month [必须]
                特定月份名称 (str)
            3) 输出: 
            - (str(个股名称，校验用), float(个股分数))
        '''
        
        if equityName in scoresMap:
            if month in scoresMap[equityName]:
                #用np.dot 计算两个list的向量内积
                if not indices:
                    return (equityName, np.dot(weights, scoresMap[equityName][month]))
                else:
                    return (equityName, np.dot(weights, [scoresMap[equityName][month][i] for i in indices]))
            else:
                #print(f'{equityName} NO FACTOR SCORE DATA for [MONTH {month}]')
                return (None, None)
        else:
            #print(f'{equityName} NO FACTOR SCORE DATA [ALL PERIOD]')
            return (None, None)

    def rankEquity(self, equityNameList, equityScoreList): #docs done

        '''
            函数: rankEquity
            -------
            1) 摘要: 此函数用于对个股进行基于分数的分组
            2) 函数输入
            - equityNameList [必须]
                个股名称列表 (list[str])
            - equityScoreList [必须] 
                个股分数列表 (list[float], 顺序对应equityNameList)
            3) 输出: 
            - list[list[str]], 每个list里面是一个分组, 里面是个股名称, 每组里股票顺序暂无意义
        '''
    
        #变成np会快一点
        equityScoreList = np.array(equityScoreList).astype(float)
        
        #从高到低还是从低到高
        if self.rankLowestFirst == "0":
            sort_index = np.argsort(equityScoreList)[::-1]
        elif self.rankLowestFirst == "1":
            sort_index = np.argsort(equityScoreList)
            
        
        sortedNames = np.array(equityNameList)[sort_index]
        
        #分成n组
        groups = np.array_split(sortedNames, self.groupnum)

        # #return a list of list of equities
        return [list(group) for group in groups]
    
    def calcBasketWeights(self, equityBasket:list, pre_bt_tradedate:str): #docs done
            
        '''
            函数: calcBasketWeights
            -------
            1) 摘要: 此函数用于对单组内个股的权重计算
            2) 函数输入
            - equityBasket [必须]
                组内股票名称 (list[list[str]])
            3) 输出: 
            - list[float], 个股权重
        '''
        #（现阶段暂包含未计算，权当等权）
        if self.stockWeightMode == 'equal':
            res = dict()
            len_stk = len*equityBasket[0]
            for stk in equityBasket[0]:
                res[stk] = 1/len_stk
            return res
        elif self.stockWeightMode == 'smart':
            stk_weight_opt = PortfolioOpt(pre_bt_tradedate, target_list=equityBasket[0], remain_list=equityBasket[1:], api_obj=self.factor_api)
            return stk_weight_opt.PortOptWeight()
            

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
                if self.stockWeightMode == 'smart' and group_name == 'group_0':
                    result_df = self.CalcStkWeight(cur_temp_df=cur_temp_df, current_tradedate=trade_day, pre_tradedate=self.pre_bt_tradedate[idx])
                else:
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

            # dict - groupedProfit
            #     {month1 : {group1:df, group2:df, group3:df}}
            #     每个df有三个col, 叫ts_code(股票代码) profit(日收益) trade_date(交易日)
    def CalcStkWeight(self, cur_temp_df:pd.DataFrame, current_tradedate:str, pre_tradedate:str):
        res_group_info = [item for sub_lst in self.equityGroupsInfo[current_tradedate][1:] for item in sub_lst]
        
        stk_weight_opt = PortfolioOpt(pre_trade_date=pre_tradedate, 
                                      target_list=self.equityGroupsInfo[current_tradedate][0], 
                                      remain_list=res_group_info,#貌似是个list of list 所以加个[0] idk why
                                      api_obj=self.stkapi)
        
        stk_weight_df = stk_weight_opt.PortOptWeight()
        res_return = defaultdict(lambda: 0)
        #res_weight = {}
        #stk weight df长度有可能和targetlist不一样，按照这个决定第一组
        print(stk_weight_df, len(stk_weight_df), len(stk_weight_opt.target_list))
        for row in cur_temp_df.itertuples():
            date = getattr(row, "trade_date")
            ticker = getattr(row, "ts_code")
            profit = getattr(row, "profit_daily")
            if ticker in stk_weight_df:
                res_return[date] += profit * stk_weight_df[ticker]
        
        df = pd.DataFrame(res_return).reset_index(names='dailyRet')
        print(df)
        
        while True:
            pass
        for row in cur_temp_df.itertuples():
            stk_daily = getattr(row, "trade_date")
            stk_code = getattr(row, "ts_code")
            stk_profit = getattr(row, "profit_daily")
            if stk_code in stk_weight_df:
                res_weight[stk_daily] = 0
                res_weight[stk_daily] += stk_profit * stk_weight_df[stk_code]
            else:
                res_weight[stk_daily] += stk_profit * stk_weight_df[stk_code]
    
        res_weight_df = pd.DataFrame(res_weight).reset_index(names='dailyRet')
        return res_weight_df

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

    def find_factor_corr_heatmap(self, Monthly_Factor_Score):
        res = defaultdict(list)

        for i in Monthly_Factor_Score.keys():
            for j in Monthly_Factor_Score[i].keys():
                res[i] += Monthly_Factor_Score[i][j]

        check = defaultdict(int)
        for i in res.keys():
            check[len(res[i])] += 1
        if len(check) != 1:
            print('不同因子数量不同，请检查数据')
            print(check)
            return
        
        df = pd.DataFrame(res)
        data = df.corr()
        return data

    # 辅助函数，用于一次跑多个测试组合
    def calculate(self, Equity_Idx_Monthly_Equity_Returns, Monthly_Equity_Returns, Monthly_Factor_Score, Equity_Idx_Monthly_Factor_Score:dict, Daily_Equity_Returns, benchmark_dailyret): #docs done

        '''
            函数: calculate
            -------
            1) 摘要: 此函数使用上述所有辅助函数计算各项指标
            2) 函数输入
            - Equity_Idx_Monthly_Equity_Returns [必须]
                个股每月的回报率，最外层索引为个股名称
                Dict[Dict[float]]
                {股票1 : {月份1 : 月收益, 月份2 : 月收益, ...}}
            - Monthly_Equity_Returns [必须]
                个股每月的回报率，最外层索引为月度名称
                Dict[list]
                {月份1 : [股票1的月收益, 股票2的月收益, 股票3的月收益, ...]}
            - Monthly_Factor_Score [必须]
                个股每月的因子得分，最外层索引为因子名称
                Dict[Dict[list]]
                {因子1 : {月份1 : [股票1分数, 股票2分数, 股票3分数, ...]}}
            - Equity_Idx_Monthly_Factor_Score [必须]
                个股每月的因子得分，最外层索引为个股名称
                Dict[Dict[list]]
                {股票1 : {月份1 : [因子1分数, 因子2分数, 因子3分数, ...]}}
            - Daily_Equity_Returns [必须]
                个股每日回报率
                pd.DataFrame
                index: 序号（无意义
                column: ts_code(股票代码) profit(单日收益) trade_date(交易日)
            - benchmark_dailyret [必须]
                对应指数每日回报率
                pd.DataFrame
                index: 序号（无意义）
                column: pct_chg(单日收益) trade_date(交易日)

            3) 输出: 
            - combinedIC
                所选因子的共同各月的IC值和累积IC值
                Dict[list]
                {'month': [交易日列表], 'IC': [各月IC值], 'cumulative': [累计IC值]}
            - df_group_net
                各分组每月的净值
                pd.DataFrame
                index = 交易日
                column = ['group_0', 'group_1', 'group2', 'group3', ..., 'longshort_hedge']
            - df_group_alpha
                各分组每月的alpha
                pd.DataFrame
                index = 交易日
                column = ['group_0', 'group_1', 'group2', 'group3', ..., 'longshort_hedge']
            - df_bt_indicator
                各分组各类收益数据合集 (此数据会被展示)
                pd.DataFrame
                index无意义
                column = ["group", "年化收益率","夏普比率","最大回撤"]
            - df_bt_alpha_indicator
                各分组各类alpha数据合集 (此数据会被展示)
                pd.DataFrame
                index无意义
                column = ['group', "年化超额收益率","超额最大回撤","calmar"]
        '''

        try:    
            Daily_Equity_Returns = Daily_Equity_Returns.drop(columns=['trade_data'])
        except:
            pass

        factor_names = list(Monthly_Factor_Score.keys())
        month_names = list(Monthly_Equity_Returns.keys())
        self.month_names = month_names
        stock_names = list(Equity_Idx_Monthly_Factor_Score.keys())
        #minMonths = 2
        groupedProfit = defaultdict(dict)
        totalIC = 0
        combinedIC = {'month': [], 'IC': [], 'cumulative': []}

        #如果要优化，则需要计算各因子IC
        #优化模式下 权重每个月都不一样，会在下面的循环中计算，这里只是计算IC
        if self.factorWeightMode == 'smart':
            
            ICList = []

            for factor in factor_names:
                ICList.append(self.calcIC(month_names, Monthly_Factor_Score[factor], Monthly_Equity_Returns)[1])
            ICList = pd.DataFrame(ICList).T

            ICList.index = month_names
            ICList.columns = factor_names
        
        #每个月循环计算个股收益
        for month in range((self.minEvalPeriod if self.factorWeightMode == 'smart' else 0), len(month_names)):            
            
            nameList, scoreList = [], []

            FactorIndices = None

            #第一步：计算权重
            if self.factorWeightMode == 'smart':
                #拿到之前的IC值（不包含当月）
                currList = ICList.loc[ICList.index[ICList.index < month_names[month]]]

                if self.factorSelectMode == 'auto':
                    factor_names, FactorIndices = self.chooseFactors(currList)

                if currList.shape[0] > self.EvalPeriod:
                    currList = currList.iloc[-self.EvalPeriod:]
                
                #keep only the names we want
                currList = currList[currList.columns.intersection(factor_names)]
                    
                #拿到当月和之前的因子打分（包含当月）
                res = defaultdict(list)
                endMonth = month_names[month]
                if month > self.EvalPeriod:
                    startMonth = month_names[month - self.EvalPeriod]
                else:
                    startMonth = month_names[0]

                for factor in factor_names: # {factor1 : {month1 : [], month2 : []}}
                    for date in Monthly_Factor_Score[factor]:
                        #print(date, int(date) >= int(startMonth), int(date) <= int(endMonth))
                        if int(date) >= int(startMonth) and int(date) <= int(endMonth):
                            res[factor] += Monthly_Factor_Score[factor][date]
                        if int(date) >= int(endMonth):
                            break

                res = pd.DataFrame(res)
                
                #计算权重
                factorWeights = self.calcFactorWeights(self.factorWeightMode, factor_names, HistoricalIC =currList, equityScore=res)
            else: #如果不用优化的话，则无需计算IC
                #全部均权重
                if self.factorWeightMode == 'equal':
                    factorWeights = self.calcFactorWeights(self.factorWeightMode, factor_names)
                #组内均权重
                elif self.factorWeightMode == 'category':
                    factorWeights = self.calcFactorWeights(self.factorWeightMode, factor_names, self.factorCategories)
                #人工决定权重（人工阈值）
                elif self.factorWeightMode == 'customized':
                    factorWeights = self.userDefinedFactorWeights
            
            if len(factorWeights) != len(factor_names):
                print('factorWeights length not equal to factor_names length!')
                continue

            #有权重以后，我们给每个股票算个分，然后把股票名字和分对应存起来
            for name in stock_names:
                if self.factorSelectMode == 'auto':
                    name, score = self.calcEquityScore(name, factorWeights, Equity_Idx_Monthly_Factor_Score, month_names[month], FactorIndices)
                else:
                    name, score = self.calcEquityScore(name, factorWeights, Equity_Idx_Monthly_Factor_Score, month_names[month])
                if name:
                    nameList.append(name)
                    scoreList.append(score)
            
            #根据分数给股票分成n组
            self.equityGroupsInfo[month_names[month]] = self.rankEquity(nameList, scoreList)
            equityGroups = self.rankEquity(nameList, scoreList)
            
            #变化数据结构，方便后面计算 
            '''
            dict - groupedProfit
                {month1 : {group1:df, group2:df, group3:df}}
                每个df有三个col, 叫ts_code(股票代码) profit(日收益) trade_date(交易日)
            '''
            #按月分组，但是这里提取每组每日的收益率
            try:
                daily_returns_this_month = Daily_Equity_Returns[(Daily_Equity_Returns['trade_date'] >= month_names[month]) 
                                    & (Daily_Equity_Returns['trade_date'] < month_names[month+1])]
            except:
                daily_returns_this_month = Daily_Equity_Returns[Daily_Equity_Returns['trade_date'] >= month_names[month]]
            
            for i, group in enumerate(equityGroups):
                # Filter df where 'ts_code' is in the current group
                df_group = daily_returns_this_month[daily_returns_this_month['ts_code'].isin(group)]
                groupedProfit[month_names[month]][f'group_{i}'] = df_group.reset_index(drop=True)  

            #计算我们多因子策略总体的月IC            
            returnArr = []
            for stock in nameList:
                returnArr.append(Equity_Idx_Monthly_Equity_Returns[stock][month_names[month]])

            currIC = np.corrcoef(returnArr, scoreList)[0,1]
            
            totalIC += currIC
            combinedIC['month'].append(month_names[month])
            combinedIC['IC'].append(currIC)
            combinedIC['cumulative'].append(totalIC)
        
        #老代码，主要用groupedProfit和benchmark_dailyret来计算每组的收益率，alpha等
        df_group_net = pd.DataFrame()   # 回测期间分组净值曲线 output
        indicator_lst = []              # 回测期间分组的回测指标
        df_group_alpha = pd.DataFrame()   # 回测期间分组的alpha曲线 new output
        alpha_indicator_lst = []          # 回测期间分组的alpha回测指标
        groupedProfit[list(groupedProfit.keys())[-1]]['group_0'].to_csv('best_stk.csv')
        group_dailyret_dict = self.EachGroupPortRet(groupedProfit)

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

        factor_corr_df = self.find_factor_corr_heatmap(Monthly_Factor_Score)
        factor_corr_df.to_csv('csv_result/factor_corr.csv')
        
        
        return combinedIC, df_group_net, df_group_alpha, df_bt_indicator, df_bt_alpha_indicator
    
    # 辅助函数，用于一次跑单个测试组合
    def run(self): #docs done
        #辅助函数，用于跑一次
        start = time.time()
        Equity_Idx_Monthly_Equity_Returns, Monthly_Equity_Returns, Monthly_Factor_Score, Equity_Idx_Monthly_Factor_Score, Daily_Equity_Returns, benchmark_dailyret = self.getData()
        print(('Finished collecting data, time = ', time.time() - start))
        return self.calculate(Equity_Idx_Monthly_Equity_Returns, Monthly_Equity_Returns, Monthly_Factor_Score, Equity_Idx_Monthly_Factor_Score, Daily_Equity_Returns, benchmark_dailyret)