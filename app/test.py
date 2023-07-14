        for idx in range(0, len(self.bt_tradedate)):
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
                    # print("eachgroup_show is: ")
                    eachgroup_show[group_name] = result_df
                    # print(eachgroup_show)
                else:
                    eachgroup_show[group_name] = pd.concat([eachgroup_show[group_name],result_df],axis=0)