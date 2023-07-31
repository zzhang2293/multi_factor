from app.factor.factorModel import factorModel
import itertools
model = factorModel()
model.start = '20230101'
import pandas as pd
import numpy as np


'''
    获取所有因子排列组合
    param: min: 最小组合数量  max: 最大组合数量

'''
def get_factor_combination_lst(min:int, max:int, path:str):
    df = pd.read_csv(path)
    longshort_hedge_res_dict = {'因子组合': [], '年化收益率': [], '夏普比率': [] ,'最大回撤': []}
    group0_dict = {'因子组合': [], '年化超额收益率': [], '超额最大回撤': [], 'calmar': []}
    factor_lst = list(df.iloc[:,0])
    for num_factor in range(min, max+1):
        for comb in itertools.combinations(factor_lst, num_factor):
            model.factor_name_lst = list(comb)
            _, _, _, df_bt_indicator, df_bt_alpha_indicator = model.run()
            long_short = df_bt_indicator.loc[df_bt_indicator['group'] == 'longshort_hedge']
            long_short = long_short.values.flatten().tolist()
            longshort_hedge_res_dict['因子组合'].append(str(list(comb)))
            longshort_hedge_res_dict['年化收益率'].append(long_short[1])
            longshort_hedge_res_dict['夏普比率'].append(long_short[2])
            longshort_hedge_res_dict['最大回撤'].append(long_short[3])

            group0 = df_bt_alpha_indicator.loc[df_bt_alpha_indicator['group'] == 'group_0']
            group0 = group0.values.flatten().tolist()
            group0_dict['因子组合'].append(str(list(comb)))
            group0_dict['年化超额收益率'].append(group0[1])
            group0_dict['超额最大回撤'].append(group0[2])
            group0_dict['calmar'].append(group0[3])
    long_short_df = pd.DataFrame(longshort_hedge_res_dict)
    long_short_df.sort_values(by='年化收益率', ascending=False)
    group0_df = pd.DataFrame(group0_dict)
    group0_df.sort_values(by='年化超额收益率', ascending=False)
    long_short_df.to_csv('longshort_res.csv')
    group0_df.to_csv('group0.csv')
    


lst = ["udslUCL", 'revQYOY']
s = pd.DataFrame(lst, columns=["factor"])
s.to_csv("factor.csv", index=None)
    

val = get_factor_combination_lst(1,2, 'factor.csv')

from collections import defaultdict
import copy


class factorModelSingleFactor(factorModel):

    def __init__(self):
        self.groupnum = 10         # 股票分组数
        self.trade_freq = 'm'      # 交易频率 "m" or "w"b
        self.end = '20230720'      # 因子分析结束日期
        self.start = '20210420' #hardcode this

    def rankEquity(self, names, scores):
    
        scores = np.array(scores).astype(float)
        sort_index = np.argsort(scores)[::-1]
            
        sortedNames = np.array(names)[sort_index]
        groups = np.array_split(sortedNames, self.groupnum)

        # return a list of list of equities
        return [list(group) for group in groups]
    
    def calc_one_factor(self, stock_names, month_list, Equity_Idx_Monthly_Factor_Score, Daily_Equity_Returns, benchmark_dailyret):

        groupedProfit = defaultdict(dict)

        for month in range(0, len(month_list)):
            nameList, scoreList = [], []
            for name in stock_names:
                nameList.append(name)
                scoreList.append(Equity_Idx_Monthly_Factor_Score[name][month_list[month]][0])

            equityGroups = self.rankEquity(nameList, scoreList)

            try:
                daily_returns_this_month = Daily_Equity_Returns[(Daily_Equity_Returns['trade_date'] >= month_list[month]) 
                                    & (Daily_Equity_Returns['trade_date'] < month_list[month+1])]
            except:
                daily_returns_this_month = Daily_Equity_Returns[Daily_Equity_Returns['trade_date'] >= month_list[month]]
            
            for i, group in enumerate(equityGroups):
                # Filter df where 'ts_code' is in the current group
                df_group = daily_returns_this_month[daily_returns_this_month['ts_code'].isin(group)]
                groupedProfit[month_list[month]][f'group_{i}'] = df_group.reset_index(drop=True)


        df_group_net = pd.DataFrame()   # 回测期间分组净值曲线 output
        indicator_lst = []              # 回测期间分组的回测指标
        df_group_alpha = pd.DataFrame()   # 回测期间分组的alpha曲线 new output
        alpha_indicator_lst = []          # 回测期间分组的alpha回测指标
        
        group_dailyret_dict = self.EachGroupPortRet(groupedProfit)

        for name in group_dailyret_dict:
            # 策略回测指标
            df_group_indicator,indi_lst = self.HistoryAccuRetAndIndicator(name, group_dailyret_dict[name])
            df_group_net[name] = df_group_indicator[name]
            indicator_lst.append(indi_lst)
            
            #策略alpha
            group_alpha, alpha_indi_lst = self.HistoryAlphaAndIndicator(name, group_dailyret_dict[name], benchmark_dailyret)
            df_group_alpha[name] = group_alpha[name]
            alpha_indicator_lst.append(alpha_indi_lst)
            
        df_bt_indicator = pd.DataFrame(indicator_lst,index = range(len(indicator_lst)),columns=["group","年化收益率","夏普比率","最大回撤"]) # 回测期间分组的回测指标 output
        df_bt_alpha_indicator = pd.DataFrame(alpha_indicator_lst,index=range(len(alpha_indicator_lst)),columns=["group","年化超额收益率","超额最大回撤","calmar"])  # 超额评价指标 new output
        
        return df_bt_indicator, df_bt_alpha_indicator
    
    def run(self):

        longshort_hedge_res_dict = {'因子组合': [], '年化收益率': [], '夏普比率': [] ,'最大回撤': []}
        group0_dict = {'因子组合': [], '年化超额收益率': [], '超额最大回撤': [], 'calmar': []}

        _, _, _, Equity_Idx_Monthly_Factor_Score, Daily_Equity_Returns, benchmark_dailyret = self.getData()
        
        df = pd.read_csv("factor.csv",header=None)
        factor_name_list = list(df.iloc[:,0])

        tempKey = Equity_Idx_Monthly_Factor_Score.keys()[0]
        month_list = list(Equity_Idx_Monthly_Factor_Score[tempKey].keys())
        stock_names = list(Equity_Idx_Monthly_Factor_Score.keys())

        for factor in itertools.combinations(factor_name_list, 1):
 
            target_equity_idx_monthly_factor_score = copy.deepcopy(Equity_Idx_Monthly_Factor_Score)
            for stk in target_equity_idx_monthly_factor_score:
                for month in target_equity_idx_monthly_factor_score[stk]:
                    target_equity_idx_monthly_factor_score[stk][month] = [target_equity_idx_monthly_factor_score[stk][month][x] for x in [i for i, value in enumerate(factor_name_list) if value in list(factor)]]

            bt, bt_alpha = self.calc_one_factor(stock_names, month_list, target_equity_idx_monthly_factor_score, Daily_Equity_Returns, benchmark_dailyret)

            long_short = bt.loc[bt['group'] == 'longshort_hedge']
            long_short = long_short.values.flatten().tolist()
            longshort_hedge_res_dict['因子组合'].append(str(list(factor)))
            longshort_hedge_res_dict['年化收益率'].append(long_short[1])
            longshort_hedge_res_dict['夏普比率'].append(long_short[2])
            longshort_hedge_res_dict['最大回撤'].append(long_short[3])

            group0 = bt_alpha.loc[bt_alpha['group'] == 'group_0']
            group0 = group0.values.flatten().tolist()
            group0_dict['因子组合'].append(str(list(factor)))
            group0_dict['年化超额收益率'].append(group0[1])
            group0_dict['超额最大回撤'].append(group0[2])
            group0_dict['calmar'].append(group0[3])

            print(f'Finished for factor [{factor}]')

        long_short_df = pd.DataFrame(longshort_hedge_res_dict)
        long_short_df.sort_values(by='年化收益率', ascending=False)
        group0_df = pd.DataFrame(group0_dict)
        group0_df.sort_values(by='年化超额收益率', ascending=False)
        long_short_df.to_csv('longshort_res.csv')
        group0_df.to_csv('group0.csv')
        
