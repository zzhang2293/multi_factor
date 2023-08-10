import math
import datetime
import pandas as pd
import functools


class PortfolioOpt():
    def __init__(self,pre_trade_date:str,target_list:list,remain_list:list,api_obj):
        self.pre_trade_date = pre_trade_date    # 调仓日的前一个交易日
        self.target_list = target_list          # 目标持仓组合
        self.remain_list = remain_list          # 剔除目标持仓组合后的剩余股票组合（排位必须有序）
        self.api_obj = api_obj                  # 数据库连接对象
        self.index_code = '000905.SH'           # 基准指数(组合优化跟踪的指数)
        self.weight_deviation = 0.02            # 某行业权重和相对指数中该行业的偏离
        self.eachstk_maxweight = 0.01           # 单只票在组合中最大权重上限
        self.industry_info_dict = None          # 申万一级行业信息

    # 获取申万一级行业分类的行业指数代码(28个行业)
    def industryIndexCodeWithSW(self):
        df = self.api_obj.SW_classify('L1')
        df = df[['index_code', 'industry_name', 'industry_code']]
        df.columns = ['industryID', 'industryName', 'indexSymbol']
        industry_symbol = list(df['industryID'])
        self.industry_info_dict = dict(zip(df['industryID'], df['industryName']))
        return df, industry_symbol

    # 一组股票按申万行业分类
    def industryClassfy(self, secID_list:list):
        df = self.api_obj.SW_StockInfo(secID_list, self.pre_trade_date)  # 股票分行业数据
        df = df[df['Level'] == 'L1']                                     # 筛选出申万一级行业的标的
        df_null = df[df['out_date'].isnull()]                            # 始终还在行业中的票
        df_not_null = df[df['out_date'].notnull()]                       # 在某个日期被剔除了此行业的票

        # df_not_null_1只保留out_date与self.pre_trade_date最接近的那条数据
        df_not_null_1 = df_not_null[df_not_null['out_date'] > self.pre_trade_date]
        df_not_null_1.sort_values(by='out_date', ascending=True, inplace=True)
        df_not_null_1.drop_duplicates(subset='con_code', keep='first', inplace=True)

        df_null_out = df_null[['con_code', 'index_code']]  # 始终还在行业中的票
        df_not_null_1_out = df_not_null_1[['con_code', 'index_code']]  # 在某个日期被剔除了此行业的票

        # 若某只票即在df_null_out中又在df_not_null_1_out中，那只保留df_null_out中的票
        df_null_lst = list(df_null_out['con_code'])
        df_not_null_1_lst = list(df_not_null_1_out['con_code'])
        repeat_code = [c_idx for c_idx in df_null_lst if c_idx in df_not_null_1_lst]
        if repeat_code:
            df_not_null_1_out = df_not_null_1_out[~df_not_null_1_out['con_code'].isin(repeat_code)]

        out_df = pd.concat([df_null_out, df_not_null_1_out], axis=0)
        out_df = out_df[['con_code', 'index_code']]
        out_df.columns = ["secID", "industryID"]

        stk_ind_dict = dict(zip(out_df['secID'],out_df['industryID']))
        return out_df,stk_ind_dict

    # 退市股票信息(最近30个交易日即将退市的股票列表)
    def delst_code_lst(self,date:str) ->list:
        result_lst = []
        df = self.api_obj.GetStockStatus([], 'D')
        df = df[['ts_code', 'delist_date']]
        temp_df = df[df['delist_date'] >= str(date)]
        temp_df.reset_index(drop=True, inplace=True)

        if not temp_df.empty:
            for idx in temp_df.index:
                temp_code = temp_df.loc[idx, 'ts_code']
                tmep_date = temp_df.loc[idx, 'delist_date']
                d1 = datetime.datetime.strptime(str(date), "%Y%m%d").date()
                d2 = datetime.datetime.strptime(tmep_date, "%Y%m%d").date()
                if (d2 - d1).days <= 30:
                    result_lst.append(temp_code)
        return result_lst

    # 提取某个交易日的指数成分股和权重信息
    def indexConstituentStockWeight(self) -> pd.DataFrame:
        blk_lst = self.delst_code_lst(self.pre_trade_date)
        index_w = self.api_obj.index_weight(self.index_code, [self.pre_trade_date])
        index_w = index_w[~index_w['con_code'].isin(blk_lst)]  # 过滤掉即将退市的股票
        index_w['wght'] = index_w['weight'] / index_w['weight'].sum()  # 指数成份股权重缩放
        index_w_1 = index_w[['con_code', 'wght']]
        index_w_1.columns = ['secID', 'weight']
        return index_w_1

    # 指数成分股所属行业及权重
    def industryWeight(self):
        idx_ct_wght = self.indexConstituentStockWeight()                                   # 指数成分股及相应权重 返回 DataFrame 'secID','weight'

        # 指数成分股所属行业及个股权重
        stock_sw_classfy,stk_temp_d = self.industryClassfy(list(idx_ct_wght['secID']))     # 指数成分股按申万一级行业分类 DataFrame "secID","industryID"
        idx_stk_classfy_wght = idx_ct_wght.merge(stock_sw_classfy, on='secID',how='left')  # 指数成分股的申万行业和权重信息 DataFrame secID weight industryID
        idx_stk_classfy_wght.dropna(inplace=True)
        idx_stk_classfy_wght['weight'] = idx_stk_classfy_wght['weight'] / idx_stk_classfy_wght['weight'].sum()  # 权重缩放 DataFrame secID weight industryID

        # 指数成分股分行业权重和
        result_s = idx_stk_classfy_wght.groupby('industryID')['weight'].sum()
        idx_industry_wght_dict = dict(zip(result_s.index, result_s.values))
        return idx_stk_classfy_wght, idx_industry_wght_dict

    def funcSelectStk(self,df):
        temp_df = df.copy()  # DataFrame secID industryID
        stk_lst = list(temp_df['secID'])  # 该行业入选的股票代码列表
        return stk_lst



    def FromRemainAddStk(self,diff_stknum:int,indu_code:str,other_dict:dict):
        temp_lst = []   # 从剩余组合中挑选的某行业股票列表

        for i in self.remain_list:
            try:
               indu_code_i = other_dict[i]   # 股票对应的行业
            except:
                #print('other_dict %s 无行业分类信息'%i)
                indu_code_i = None

            if indu_code_i == indu_code:
                temp_lst.append(i)
            if len(temp_lst) >= diff_stknum:
                break
        return temp_lst

    def PortOptWeight(self):
        # 指数成分股以及目标组合按行业分类
        idx_stk_clfy_w,idx_industry_wght_dict = self.industryWeight()      # 指数成分股分行业权重和 idx_industry_wght_dict dict 健行业代码 值为行业权重
        target_df, target_dict =  self.industryClassfy(self.target_list)   # 目标组合按照申万一级行业分类  DataFrame "secID", "industryID"
        other_df, other_dict = self.industryClassfy(self.remain_list)      # 剔除目标持仓组合后的剩余股票组合按照申万一级行业分类 DataFrame "secID", "industryID"

        # 组合优化
        result_s = target_df.groupby('industryID').apply(self.funcSelectStk)  # 目标持仓按行业归类 Series index为行业代码 values为股票代码列表
        port_lst = []
        port_wght = []

        for indu_code,indu_wght in idx_industry_wght_dict.items():
            each_indu_stknum = math.floor(indu_wght / self.eachstk_maxweight)    # 指数成分分行业中单只票按权重上限计算，需持有的股票数量
            if indu_code in list(result_s.index):
                re_lst = result_s[indu_code]    # 目标组合按行业持有股票列表
                if len(re_lst) >= each_indu_stknum:
                    re_lst_w = [indu_wght / len(re_lst)] * len(re_lst)   # 股票列表对应的权重
                else:
                    diff_num = each_indu_stknum - len(re_lst)            # 组合中某行业股票不足的数量
                    add_lst = self.FromRemainAddStk(diff_num,indu_code,other_dict)
                    re_lst.extend(add_lst)
                    re_lst_w = [indu_wght / len(re_lst)] * len(re_lst)  # 股票列表对应的权重
            else:
                 re_lst = self.FromRemainAddStk(each_indu_stknum,indu_code,other_dict)
                 re_lst_w = [indu_wght / len(re_lst)] * len(re_lst)  # 股票列表对应的权重

            port_lst.extend(re_lst)
            port_wght.extend(re_lst_w)

        # 权重归一化
        wght_sum = functools.reduce(lambda x, y: x + y, port_wght)
        port_lst_wght = [float(w) / wght_sum for w in port_wght]
        portfolio_dict = dict(zip(port_lst, port_lst_wght))
        return portfolio_dict