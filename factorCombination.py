from app.factor.factorModel import factorModel
import itertools
model = factorModel()
model.start = '20230101'
import pandas as pd


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




    

val = get_factor_combination_lst(1,2, 'factor.csv')
