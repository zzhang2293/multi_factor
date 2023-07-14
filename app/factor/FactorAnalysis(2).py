import math
import time
import datetime
import warnings
import numpy as np
import pandas as pd
from .api.SelectData import SelectFromMongo
from .api.SelectFactorData import GetFactorFromMongoDB
import threading
import bisect
import multiprocessing as mp

stk_api = SelectFromMongo()
factor_api = GetFactorFromMongoDB()
warnings.filterwarnings('ignore')    # 过滤掉系统警告日志


class AnalysisMethod():
    def __init__(self):
        self.groupnum = 10             # 股票分组数
        self.trade_freq = "m"          # 交易频率 "m" or "w"
        self.end = "20230626"          # 因子分析结束日期
        self.start = "20201215"        # 因子分析开始日期
        self.factor_name_lst = ["aShareholderZ"]      # 需要分析的因子列表，因子必须要为因子数据库中的因子，若列表包含多因子，则只分析等权因子组合
        self.universe_index = ['000852.SH', '000905.SH', '000300.SH', '399303.SZ'] # 股票池范围，若为空表示全市场
        self.universe = []             # 股票池列表
        self.hold_period = {}          # 历史持有期，字典中的键为调仓日、值为持有期交易日列表（按升序排序的区间交易日）
        self.bt_tradedate = []         # 调仓日
        self.pre_bt_tradedate = []     # 调仓日的前一个交易日
        self.Facor_IsAscending = False # 因子值默认为降序
        self.allfactorname_lst = factor_api.GetAllFactorName()  # 数据库因子名称列表
        self.df_IC = ""
        self.IC_mean = ""
        self.df_group_net = ""
        self.df_bt_indicator = ""
        self.lock = threading.Lock()
    # 获取某日停牌复牌股票信息
    def GetSuspendInfo(self,trade_date, db_api, c=1):
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
    def delst_code_lst(self,date, db_api):
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

                # if temp_code == '600978.SH':
                #     print((d2-d1).days)
                #     print(tmep_date,date)
                if (d2 - d1).days <= 200 and (d2 - d1).days >= -200:
                    result_lst.append(temp_code)
        return result_lst

    # 调仓日和区间持有期
    def TradeDateDeal(self):
        his_trade_df = stk_api.trade_cal(self.start,self.end)
        #print(his_trade_df)
        his_trade_df = his_trade_df[his_trade_df['is_open']==1]
        historyTradeDate = list(his_trade_df['cal_date'])            # 当前交易日列表
        pre_historyTradeDate = list(his_trade_df['pretrade_date'])   # 当前交易日的前一个交易日列表

        bt_TradeDate = []           # 调仓日列表
        pre_bt_TradeDate = []       # 调仓日的上一个交易日

        # 生成调仓日列表
        if self.trade_freq == "m":
            for idx in range(1, len(historyTradeDate)):
                c_dt = historyTradeDate[idx]
                p_dt = historyTradeDate[idx - 1]
                if c_dt[4:6] != p_dt[4:6]:
                    bt_TradeDate.append(c_dt)
                    pre_bt_TradeDate.append(pre_historyTradeDate[idx])

        elif self.trade_freq == "w":
            for idx in range(0, len(historyTradeDate)):
                c_dt = historyTradeDate[idx]
                if datetime.datetime.strptime(c_dt, "%Y%m%d").weekday() + 1 == 1:  # 交易日为周一的日期
                    bt_TradeDate.append(c_dt)
                    pre_bt_TradeDate.append(pre_historyTradeDate[idx])

        self.bt_tradedate = bt_TradeDate            # 升序排序的日期
        self.pre_bt_tradedate = pre_bt_TradeDate

        #time1 = time.time()
        # 生成持有期列表
        for item in range(0,len(bt_TradeDate)):
            s_period = bt_TradeDate[item]
            if item != len(bt_TradeDate)-1:
                e_period = bt_TradeDate[item+1]
            else:
                e_period = historyTradeDate[-1]

            s_idx = historyTradeDate.index(s_period)
            e_idx = historyTradeDate.index(e_period)
            if len(historyTradeDate[s_idx:e_idx+1])>=4:
                self.hold_period[s_period] = historyTradeDate[s_idx:e_idx+1]
        #time2 = time.time()
        #print("code block", time2 - time1)
    # 需要提取数据的股票范围
    def SetUniverse(self, data_date):
        if not self.universe_index:
            self.universe = []
        else:
            temp_uni = []
            for idx in self.universe_index:
                index_con_df = stk_api.index_weight(idx, [data_date])
                if index_con_df.empty:
                    print('select stock in %s' % idx, 'but the index_con_df is empty')
                index_con_lst = list(index_con_df['con_code'])
                temp_uni.extend(index_con_lst)

            temp_uni_0 = list(set(temp_uni))  # 列表股票代码去重

            susp_lst = self.GetSuspendInfo(data_date, stk_api)  # 某日全市场已经停牌得股票代码列表
            delist = self.delst_code_lst(data_date, stk_api)    # 近期即将被退市的股票代码信息
            black_list = susp_lst + delist
            temp_uni_0 = [x for x in temp_uni_0 if x not in black_list] # 剔除在黑名单中的股票代码
            self.universe = temp_uni_0

    # 因子去极值
    def WinSorizeNewMethod(self,df, winsorize_max_num=5):
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

    # 因子标准化
    def StandarDize(self,df):
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

    # 获取某一天股票因子数据
    def GetFactorFromDB(self,data_date):
        '''
        :param data_date: str 调仓日的前一个交易日
        :return: DataFrame 行索引为股票代码，列为因子值 只有一列
        '''
        # 提取因子
        factor_df = factor_api.GetStockFactor(data_date,self.universe,self.factor_name_lst)
        factor_df.fillna(0, inplace=True)

        # 因子预处理
        factor_df = self.StandarDize(self.WinSorizeNewMethod(factor_df))

        if len(self.factor_name_lst)>1:                 # 当因子列表中有多个因子时，将因子等权合成
           factor_df['score'] = factor_df.mean(1)
           factor_df = factor_df[['score']]
        cls_name = list(factor_df.columns)[0]
        factor_df.sort_values(by=cls_name,ascending=self.Facor_IsAscending,inplace=True) # 因子值排序
        return factor_df

    # 因子分组
    def FactorDivGroup(self,df_factor):
        '''
        :param df_factor:  DataFrame 行索引为股票代码，列为因子值（降序）
        :return dict_group dict 键为分组数，如group_1；值为因子DataFrame 行索引为股票代码、唯一列为因子值
        '''
        length = len(df_factor.iloc[:,0])
        each_group_hold = math.floor(length / self.groupnum)  # 每组初始持有股票数

        dict_group = {}
        for idx in range(0,self.groupnum):
            if idx != self.groupnum - 1:
                temp_df = df_factor.iloc[idx*each_group_hold:(idx+1)*each_group_hold,0]
            else:
                temp_df = df_factor.iloc[idx*each_group_hold:,0]
            dict_group["group_%s"%(idx)] = temp_df
        return dict_group

    # 从数据库获取持有期股票每日收益率数据
    def GetHoldPeriodPriceRet(self,hold_datelst,factor_group_dict):
        '''
        :param hold_datelst:      持有期交易日列表（升序）
        :param factor_group_dict: 因子分组字典 键为分组名，值为因子值DataFrame(行索引为股票代码，列为因子值)
        :return: dict 键为分组名，值为股票每天收益率 DataFrame 'trade_date','ts_code','chgPct'
        '''
        s_date = hold_datelst[0]    # 持有的第一个交易日（调仓日）
        e_date = hold_datelst[-1]   # 持有的最后一个交易日（下一个调仓日）
        ret_group_dict = {}

        for key in factor_group_dict:
            stk_lst = list(factor_group_dict[key].index)  # 当前分组对应的股票代码
            his_data = stk_api.MktEqudAdjGet(stk_lst, [], s_date, e_date,['trade_date','ts_code','close','pre_close'])
            his_data.dropna(inplace=True)
            his_data['chgPct'] = his_data['close'] / his_data['pre_close'] - 1
            his_data = his_data[['trade_date','ts_code','chgPct']]
            ret_group_dict[key] = his_data
        return ret_group_dict

    # 持有期分层组合中每只股票的持有收益率计算
    def StockHoldRet(self,stkdailyret_group_dict):
        '''
        :param stkdailyret_group_dict: dict 键为分组名 值为DataFrame 持有期分组的每只股票的每日收益率
        :return: dict 健为分层名，值为DataFrame(行索引为股票代码，唯一列为股票区间累计收益率)
        '''
        stk_holdret_dict = {}     # dict 健为分层名，值为DataFrame(行索引为股票代码，列为股票区间累计收益率)
        for name in stkdailyret_group_dict:
            temp_df = stkdailyret_group_dict[name]  # DataFrame 'trade_date','ts_code','chgPct'
            result_s = temp_df.groupby('ts_code')['chgPct'].sum()
            result_df = result_s.to_frame()
            result_df.columns = ['accuChgPct']
            stk_holdret_dict[name] = result_df
        return stk_holdret_dict

    # IC计算
    def ComputeIC(self,factor_group_dict,stk_holdret_dict):
        '''
        :param factor_group_dict: dict 键为分组名，如group_1；值为因子DataFrame 行索引为股票代码，唯一列为因子值
        :param stk_holdret_dict: dict 健为分组名，值为DataFrame(行索引为股票代码，唯一列为股票区间累计收益率)
        :return: float
        '''
        df_factor = pd.DataFrame()
        df_holdret = pd.DataFrame()
        df_temp = pd.DataFrame()

        for g_name in factor_group_dict:
            df_factor = pd.concat([df_factor,factor_group_dict[g_name]],axis=0)

        for g_name_1 in stk_holdret_dict:
            df_holdret = pd.concat([df_holdret,stk_holdret_dict[g_name_1]],axis=0)

        df_factor.columns = ['factor']
        df_holdret.columns = ['accuret']
        df_temp['factor'] = df_factor['factor']
        df_temp['accuret'] = df_holdret['accuret']
        IC_value = df_temp['factor'].corr(df_temp['accuret'])
        return IC_value

    # 计算持有期每组的组合收益率
    def EachGroupPortRet(self,all_period_data):
        '''
        :param all_period_data: 所有持有期因子分组数据的日收益率序列 dict 键为调仓日 值为dict(键为分组名、值为每组股票对应的日收益率)
        :return: dict 键为分组名 值为DataFrame "trade_date","dailyRet"
        '''
        #遍历持有期并计算分组表现
        eachgroup_show = {}
        # lock_list = []

        for idx in range(0, len(self.bt_tradedate)):
            # thread = threading.Thread(target=self.cal_profit, args=(idx, all_period_data, eachgroup_show))
            # lock_list.append(thread)
            # thread.start()
            cur_trade_day = self.bt_tradedate[idx]  # 当前调仓日

            if cur_trade_day not in all_period_data:
                continue

            cur_each_period = all_period_data[cur_trade_day]  # 当前持有期的分组信息,键为分组名，值为DataFrame
            #print(cur_each_period)
            next_each_period = {}                             # 下一期持有期的分组信息，键为分组名，值为DataFrame
            if idx != len(self.bt_tradedate) - 1 and self.bt_tradedate[idx + 1] in all_period_data:
                next_each_period = all_period_data[self.bt_tradedate[idx + 1]]

            # 遍历分组
            for group_name in cur_each_period:
                
                cur_temp_df = cur_each_period[group_name]      # 当前组 DataFrame 'trade_date','ts_code','chgPct'
                # 计算手续费
                fee = 0
                if next_each_period:
                    next_temp_df = next_each_period[group_name]# 下一期对应得组 DataFrame 'trade_date','ts_code','chgPct'
                    cur_hold = list(cur_temp_df['ts_code'])    # 当前期x组对应的持仓
                    next_hold = list(next_temp_df['ts_code'])  # 下一期对应的x组的持仓
                    cur_stknum = len(cur_hold)
                    next_stknum = len(next_hold)
                    max_holdnum = max(cur_stknum,next_stknum)   # 组合持仓股票数量
                    diff_hold = [x for x in next_hold if x not in cur_hold]
                    fee = len(diff_hold) * 2 * 0.0007 / max_holdnum # 某组换仓手续费计算

                # 计算该组每日组合收益率
                result_s = cur_temp_df.groupby('trade_date')['chgPct'].mean()
                result_df = result_s.to_frame()
                result_df.columns = ['dailyRet']
                result_df["trade_date"] = list(result_df.index)
                result_df.reset_index(drop=True,inplace=True)
                result_df.loc[len(result_df['trade_date'])-1,"dailyRet"] = result_df.loc[len(result_df['trade_date'])-1,"dailyRet"] - fee # 扣除手续费
                result_df = result_df[["trade_date","dailyRet"]]

                if group_name not in eachgroup_show:
                    eachgroup_show[group_name] = result_df
                    # print(eachgroup_show)
                else:
                    eachgroup_show[group_name] = pd.concat([eachgroup_show[group_name],result_df],axis=0)
        # for t in lock_list:
        #     t.join()
        #with open("multi_thread.txt", "w") as f:
        # with open("single_thread.txt", 'w') as f:
        #     print(eachgroup_show, file=f)
        # print("START????????????????????????????????????")
        # for item in eachgroup_show:
        #     for val in eachgroup_show[item].iterrows():        #         print(val)
        # print("end????????????????")
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

        year_ret = round(year_ret,4)
        sharpe_ratio = round(sharpe_ratio, 4)
        max_drawn = round(max_drawn, 4)
        indicator_lst = [group_name,year_ret,sharpe_ratio,max_drawn]
        temp_df = temp_df[["trade_date","net_values"]]
        temp_df.set_index("trade_date",drop=True,inplace=True)
        temp_df.columns = [group_name]
        return temp_df,indicator_lst

    # 每个持有期每组的日收益率序列
    def MainFunc(self):
        t0 = time.time()
        self.TradeDateDeal()                          # 生成调仓日和区间持仓日列表

        # 遍历持有期，对因子分组并从数据库提取日收益率
        all_period_data = {}  # 所有持有期因子分层数据的日收益率序列 dict 键为调仓日 值为dict(键为分组名、值为每组股票对应的日收益率)
        IC_info_lst = []
        time1 = time.time()
        thread_list = []
        for item in range(0,len(self.bt_tradedate)):
            process = threading.Thread(target=self.get_val, args=(item, all_period_data, IC_info_lst))
            thread_list.append(process)
            process.start()
            # if self.bt_tradedate[item] not in self.hold_period:
            #     continue  # 将持有期太少的期数过滤掉
            # else:
            #     pre_day = self.pre_bt_tradedate[item]     # 前一个交易日
            #     cur_day = self.bt_tradedate[item]         # 当前交易日(调仓日)
            #     hold_datelst = self.hold_period[cur_day]  # 当期持有期交易日列表
                
  
            # self.SetUniverse(pre_day)                       # 股票池设置
            # df_f = self.GetFactorFromDB(pre_day)            # 提取调仓日前一个交易的因子值
            # factor_group_dict = self.FactorDivGroup(df_f)   # 当期因子分组 dict 键为分组名，如group_1；值为因子DataFrame 行索引为股票代码，列为因子值
            # # line_420_time = time.time()
            # stkdailyret_group_dict = self.GetHoldPeriodPriceRet(hold_datelst,factor_group_dict)  # dict 键为分组名 值为DataFrame 持有期分组的每只股票的每日收益率
            # # line_422_time = time.time()
            # # print("line 421 time = ", line_422_time - line_420_time)
            # all_period_data[cur_day] = stkdailyret_group_dict
            # stk_holdret_dict = self.StockHoldRet(stkdailyret_group_dict) # 持有期每只股票的区间累计涨跌幅
            # this_period_IC = self.ComputeIC(factor_group_dict,stk_holdret_dict)
            # IC_info_lst.append([cur_day,this_period_IC])
        for t in thread_list:
            t.join()
        # with open("IC_info_list_single.txt", 'w') as f:
        #     print(IC_info_lst, file=f)
        time2 = time.time()
        # IC_info_lst.sort()
        print("code block", time2 - time1)
        # 历史持有期IC值
        self.df_IC = pd.DataFrame(IC_info_lst,index=range(len(IC_info_lst)),columns=["tradeDate","IC"])
        self.df_IC["IC_累计值"] = self.df_IC['IC'].cumsum()
        self.df_IC.set_index("tradeDate",drop=True,inplace=True)  # 每期对应的IC值 output
        self.IC_mean = self.df_IC['IC'].mean()              # IC均值 output
        # print(IC_mean)
        # df_IC.to_csv("./df_IC.csv")

        # 遍历持有期,计算分组的每日收益率序列
        #print(all_period_data)\
        start_time = time.time()

        group_dailyret_dict = self.EachGroupPortRet(all_period_data)  # 键为分组名 值为DataFrame "trade_date","dailyRet" TODO
        with open("dailyret_multi.txt", 'w') as f:
        # with open("dailyret_multi.txt", 'w') as f:
            for item in group_dailyret_dict:
                print(group_dailyret_dict[item], file=f)
        print("group_dailyret diff:", time.time()- start_time)
        # 遍历分组，回测结果保存并展示
        self.df_group_net = pd.DataFrame()   # 回测期间分组净值曲线 output
        indicator_lst = []              # 回测期间分组的回测指标
        for name in group_dailyret_dict:
            df_group_indicator,indi_lst = self.HistoryAccuRetAndIndicator(name,group_dailyret_dict[name])  
            self.df_group_net[name] = df_group_indicator[name]
            indicator_lst.append(indi_lst)
        self.df_bt_indicator = pd.DataFrame(indicator_lst,index = range(len(indicator_lst)),columns=["group","年化收益率","夏普比率","最大回撤"]) # 回测期间分组的回测指标 output
        # with open("df_bt_indicator_multi.txt", 'w') as f:
        with open("df_bt_indicator_single.txt", 'w') as f:

            print(self.df_bt_indicator, file=f)
            
        t1 = time.time()
        print("耗时：",t1-t0)
        print("finish")


    def get_val(self, item:int, all_period_data:dict, IC_info_lst:list):
        if self.bt_tradedate[item] not in self.hold_period:
            return 0
        
        else:
            pre_day = self.pre_bt_tradedate[item]
            cur_day = self.bt_tradedate[item]
            hold_datelst = self.hold_period[cur_day]
        self.SetUniverse(pre_day)
        df_f = self.GetFactorFromDB(pre_day)
        factor_group_dict = self.FactorDivGroup(df_f)
        stkdailyret_group_dict = self.GetHoldPeriodPriceRet(hold_datelst, factor_group_dict)
        all_period_data[cur_day] = stkdailyret_group_dict
        stk_holdret_dict = self.StockHoldRet(stkdailyret_group_dict)
        this_period_IC = self.ComputeIC(factor_group_dict, stk_holdret_dict)
        #bisect.insort(IC_info_lst, [cur_day, this_period_IC], key=lambda x: x[0])
        IC_info_lst.append([cur_day, this_period_IC])
        IC_info_lst.sort(key = lambda x:x[0])
        return 1

    def cal_profit(self, idx:int, all_period_data:dict, eachgroup_show:dict):
        cur_trade_day = self.bt_tradedate[idx]
        if cur_trade_day not in all_period_data:
            return 0
        cur_each_period = all_period_data[cur_trade_day]
        # print(cur_each_period)
        next_each_period = {}
        if idx != len(self.bt_tradedate) - 1 and self.bt_tradedate[idx + 1] in all_period_data:
            next_each_period = all_period_data[self.bt_tradedate[idx + 1]]
        for group_name in cur_each_period:
            cur_temp_df = cur_each_period[group_name]
            fee = 0
            if next_each_period:
                next_temp_df = next_each_period[group_name]
                cur_hold = list(cur_temp_df['ts_code'])    # 当前期x组对应的持仓
                next_hold = list(next_temp_df['ts_code'])  # 下一期对应的x组的持仓
                cur_stknum = len(cur_hold)
                next_stknum = len(next_hold)
                max_holdnum = max(cur_stknum,next_stknum)   # 组合持仓股票数量
                diff_hold = [x for x in next_hold if x not in cur_hold]
                fee = len(diff_hold) * 2 * 0.0007 / max_holdnum # 某组换仓手续费计算             
                            # 计算该组每日组合收益率
            result_s = cur_temp_df.groupby('trade_date')['chgPct'].mean()
            result_df = result_s.to_frame()
            result_df.columns = ['dailyRet']
            result_df["trade_date"] = list(result_df.index)
            result_df.reset_index(drop=True,inplace=True)
            result_df.loc[len(result_df['trade_date'])-1,"dailyRet"] = result_df.loc[len(result_df['trade_date'])-1,"dailyRet"] - fee # 扣除手续费
            result_df = result_df[["trade_date","dailyRet"]]
            # print(result_df)
            # print(group_name)
            # print("")
            if group_name not in eachgroup_show:
                # print("not in show")
                
                eachgroup_show[group_name] = result_df
                # print(type(result_df["trade_date"][0]))
            #print(group_name)
            else:
                # print("in show")
                self.lock.acquire()
                eachgroup_show[group_name] = pd.concat([eachgroup_show[group_name], result_df], axis=0).sort_values(by=["trade_date"])
                print(eachgroup_show[group_name])
                self.lock.release()
        print("finish: ", cur_trade_day)
        return 0

if __name__ == '__main__':
    analysis_obj = AnalysisMethod()

    analysis_obj.MainFunc()