import math
import time
import warnings
import numpy as np
import pandas as pd
from api.SelectData import SelectFromMongo
from api.SelectFactorData import GetFactorFromMongoDB
import threading
import datetime
import numpy as np
from factorWeights import calcWeights

warnings.filterwarnings('ignore')
def get_val(bt_tradedate:list, hold_period:dict, pre_bt_tradedate:list, 
            universe:list, universe_index:list, factor_name_lst:list, item:int, all_period_data:dict
            , stk_api:SelectFromMongo, factor_api: GetFactorFromMongoDB, delete_list:list):
    if bt_tradedate[item] not in hold_period:
        return 0
    
    pre_day = pre_bt_tradedate[item]
    cur_day = bt_tradedate[item]
    hold_datelst = hold_period[cur_day]
    SetUniverse(universe_index, pre_day)
    df_f2= GetFactorFromDB(universe, factor_name_lst, pre_day, stk_api, factor_api) # df_f2: dataframe, index is ts_code, column is factors, period is monthly
    need_to_delete1 = delst_code_lst(pre_day, stk_api)
    need_to_delete2 = GetSuspendInfo(pre_day, stk_api)
    delete_list.append(need_to_delete1)
    delete_list.append(need_to_delete2)
    stock_daily_profit = GetHoldPeriodPriceRet(hold_datelst, df_f2, stk_api)
    stk_holdret_monthly = StockHoldRet(stock_daily_profit) # monthly profit, use for analysis
    df_f2["profit_month"] = stk_holdret_monthly["profit_month"]
    all_period_data[item] = df_f2
    return 0


#停牌复牌股票
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

def SetUniverse(universe_index:list, data_date:int):
    if universe_index:
        return []
    else:
        temp_uni = []
        for idx in universe_index:
            index_con_df = stk_api.index_weight(idx, [data_date])
            if index_con_df.empty:
                print('select stock in %s' % idx, 'but the index_con_df is empty')
            index_con_lst = list(index_con_df['con_code'])
            temp_uni.extend(index_con_lst)
        temp_uni_0 = list(set(temp_uni))  # 列表股票代码去重
        susp_lst = GetSuspendInfo(data_date, stk_api)  # 某日全市场已经停牌得股票代码列表
        delist = delst_code_lst(data_date, stk_api)    # 近期即将被退市的股票代码信息
        black_list = susp_lst + delist
        temp_uni_0 = [x for x in temp_uni_0 if x not in black_list] # 剔除在黑名单中的股票代码
        return temp_uni_0

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



def GetRet(hold_datelst:list, factor_group_dict:dict):
    s_date = hold_datelst[0]
    e_date = hold_datelst[-1]
    ret_group_dict = {}

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
    print(analysis_3D_list_label)
    print(analysis_2D_monthly_return)
    analysis_2D_monthly_return = np.array(analysis_2D_monthly_return[:3]).T.tolist()
    return analysis_3D_list_label, analysis_2D_monthly_return


class AnalysisMethod():
    def __init__(self):
        self.groupnum = 10         # 股票分组数
        self.trade_freq = 'm'      # 交易频率 "m" or "w"
        self.end = '20230630'      # 因子分析结束日期
        self.start = '20221215'
        self.factor_name_lst = ['Analyst_factor', 'daizhuerjiu']
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
  
    """
        TradeDateDeal: 调仓日和区间持有期
    
    """
    def TradeDateDeal(self):
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






    def MainFunc(self):
        t0 = time.time()
        self.TradeDateDeal()
        #print(self.hold_period)
        all_period_data = {}
        IC_info_lst = []
        thread_list = []
        delete_list = []
        print(self.bt_tradedate)
        for item in range(len(self.bt_tradedate)):
            # process = threading.Thread(target=self.get_val, args=(item, all_period_data, IC_info_lst))
            get_val(self.bt_tradedate, self.hold_period, self.pre_bt_tradedate, 
                    self.universe, self.universe_index, self.factor_name_lst, 
                    item, all_period_data, self.stkapi, self.factor_api, delete_list)
        delete_list = [item for sublist in delete_list for item in sublist]
        for day in all_period_data:
            print(all_period_data[day])
            new_data = all_period_data[day].drop(delete_list, errors='ignore')
            all_period_data[day] = new_data
        for day in all_period_data:
            print(all_period_data[day])
            filtered = all_period_data[day][all_period_data[day].index.isin(list(all_period_data[0].index))]
            all_period_data[day] = filtered
        factor_3D, month_profit = transform_data(all_period_data, self.factor_name_lst)
        
        calcWeights.calcWeights(listOfFactors=self.factor_name_lst, )

if __name__ == "__main__":
    analysis = AnalysisMethod()
    analysis.MainFunc()
