import time
from app.factor.factorModel import factorModel
import itertools
import copy
model = factorModel()
model.start = '20201212'
model.end = '20230731'
import pandas as pd
import os
import csv
import shutil


'''
    获取所有因子排列组合
    param: min: 最小组合数量  max: 最大组合数量

'''


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
'''
            def calculate(self, Equity_Idx_Monthly_Equity_Returns, Monthly_Equity_Returns, 
            Monthly_Factor_Score, Equity_Idx_Monthly_Factor_Score:dict, 
            Daily_Equity_Returns, benchmark_dailyret):

'''

        
def get_factor_combination_lst(min:int, max:int, path:str):
    header = False
    header_longshort = ['因子组合', '年化收益率', '夏普比率', '最大回撤']
    header_group0 = ['因子组合', '年化超额收益率', '超额最大回撤', 'calmar']
    if not os.path.exists('CombinationResult'):
        os.makedirs('CombinationResult')
    else:
        shutil.rmtree('CombinationResult')
        os.makedirs("CombinationResult")
    df = pd.read_csv(path,header=None)
    longshort_hedge_res_dict = {'因子组合': [], '年化收益率': [], '夏普比率': [] ,'最大回撤': []}
    group0_dict = {'因子组合': [], '年化超额收益率': [], '超额最大回撤': [], 'calmar': []}
    #factor_lst = list(df.iloc[:,0])
    factor_lst = model.allfactorname_lst
    model.factor_name_lst = factor_lst
    Equity_Idx_Monthly_Equity_Returns, Monthly_Equity_Returns, Monthly_Factor_Score, Equity_Idx_Monthly_Factor_Score, Daily_Equity_Returns, benchmark_dailyret = model.getData()
    nums_iter = 0
    start = time.time()
    for num_factor in range(min, max+1):
        for comb in itertools.combinations(factor_lst, num_factor):
            #temp = model.factor_name_lst
            model.factor_name_lst = list(comb)
            target_equity_idx_monthly_factor_score = copy.deepcopy(Equity_Idx_Monthly_Factor_Score)
            for stk in target_equity_idx_monthly_factor_score:
                for month in target_equity_idx_monthly_factor_score[stk]:
                    target_equity_idx_monthly_factor_score[stk][month] = [target_equity_idx_monthly_factor_score[stk][month][x] for x in [i for i, value in enumerate(factor_lst) if value in list(comb)]]

            target_monthly_factor_score = copy.deepcopy(Monthly_Factor_Score)
            target_monthly_factor_score = {key : target_monthly_factor_score[key] for key in list(comb)}

            _, _, _, df_bt_indicator, df_bt_alpha_indicator = model.calculate(Equity_Idx_Monthly_Equity_Returns, Monthly_Equity_Returns, target_monthly_factor_score,
                                                                                target_equity_idx_monthly_factor_score, Daily_Equity_Returns, benchmark_dailyret)
            # write longshort 
            long_short = df_bt_indicator.loc[df_bt_indicator['group'] == 'longshort_hedge']
            long_short = long_short.values.flatten().tolist()
            file = open('CombinationResult/long_short_result.csv', 'a', newline='', encoding='utf-8')
            writer_longshort = csv.DictWriter(file, fieldnames=header_longshort)
            with file:
                if not header:
                    writer_longshort.writeheader()
                writer_longshort.writerow({'因子组合': str(list(comb)), '年化收益率': long_short[1], '夏普比率': long_short[2], '最大回撤': long_short[3]})
                longshort_hedge_res_dict['因子组合'].append(str(list(comb)))
                longshort_hedge_res_dict['年化收益率'].append(long_short[1])
                longshort_hedge_res_dict['夏普比率'].append(long_short[2])
                longshort_hedge_res_dict['最大回撤'].append(long_short[3])
            

            # write the group0 result
            group0 = df_bt_alpha_indicator.loc[df_bt_alpha_indicator['group'] == 'group_0']
            group0 = group0.values.flatten().tolist()
            file = open('CombinationResult/group0.csv', 'a', newline='', encoding='utf-8')
            with file:
                writer_group0 = csv.DictWriter(file, fieldnames=header_group0)
                if not header:
                    writer_group0.writeheader()
                writer_group0.writerow({'因子组合': str(list(comb)), '年化超额收益率': group0[1], '超额最大回撤': group0[2], 'calmar': group0[3]})
                group0_dict['因子组合'].append(str(list(comb)))
                group0_dict['年化超额收益率'].append(group0[1])
                group0_dict['超额最大回撤'].append(group0[2])
                group0_dict['calmar'].append(group0[3])

            nums_iter += 1
            header = True
            print("get combination time:", nums_iter)
    long_short_df = pd.DataFrame(longshort_hedge_res_dict)
    long_short_df.sort_values(by='年化收益率', ascending=False, inplace=True)
    group0_df = pd.DataFrame(group0_dict)
    group0_df.sort_values(by='年化超额收益率', ascending=False, inplace=True)
    long_short_df.to_csv('CombinationResult/long_short_res_sort.csv')
    group0_df.to_csv('CombinationResult/group0_sort.csv')
    print("total_time:", time.time() - start)
# def calculate_indicator(Daily_Equity_Returns:pd.DataFrame, Monthly_Equity_Returns:dict):
#     month_names = list(Monthly_Equity_Returns.keys())
#     try:
#         Daily_Equity_Returns = Daily_Equity_Returns.drop(columns=['trade_data'])
#     except:
#         pass
    
#     for month in range(0, len(month_names))



val = get_factor_combination_lst(1,1, 'factor.csv')
