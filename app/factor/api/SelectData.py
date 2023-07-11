import datetime
import pandas as pd
import numpy as np
from app.factor.api.MongoConnect import MongoConnect

class SelectFromMongo():
    def __init__(self):
        self.db = MongoConnect()
        self.db.Connect("192.168.1.18", "17629", "fww", "fww8888", "admin")

    def GetLastTradeDate(self):
        '''
        获取最新一个交易日
        :return: str 最新交易日
        '''
        Today = datetime.datetime.now()
        NearMonthDay = Today - datetime.timedelta(days=15)

        TradeDateList = self.trade_cal(NearMonthDay.strftime("%Y%m%d"), Today.strftime("%Y%m%d"), 1)
        LastTradeDay = None
        i = -1
        if len(TradeDateList) > 0:
            while True:
                if TradeDateList[i]["is_open"] == 1:
                    LastTradeDay = TradeDateList[i]["cal_date"]
                    break
                else:
                    i -= 1
            return LastTradeDay
        else:
            raise "this Month Not Have trade_calendar , Maybe DataBase is not update"

    def GetStockTradeStatus(self,secID:str):
        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0

        SelectDict = {}
        SelectDict["ts_code"] = secID
        StockStatus = self.db.Select("StockBackSys", "Stock_Status", SelectDict, TitleDict)
        for tInfo in StockStatus:
            if tInfo['ts_code'] == secID:
                if tInfo['list_status'] == "D":
                    return False,tInfo['delist_date']
                else:
                    return True,""

    # 股票状态（基础信息）
    def GetStockStatus(self,secID:list,tStatus):
        # 确定返回字段
        ResultList = []

        TitleDict = {}
        TitleDict["_id"] = 0

        SelectDict = {}
        if len(secID) > 0:
            SelectDict["$or"] = []
            for tID in secID:
                SelectDict["$or"].append({"ts_code": tID})
        SelectDict["list_status"] = tStatus

        StockStatus = self.db.Select("StockBackSys", "Stock_Status", SelectDict, TitleDict)
        for tInfo in StockStatus:
            ResultList.append(tInfo)
        df = pd.DataFrame(ResultList)
        return df

    # 转债状态（基础信息）
    def GetBondInfoStatus(self, secID: list):
        # 确定返回字段
        ResultList = []

        TitleDict = {}
        TitleDict["_id"] = 0

        SelectDict = {}
        if len(secID) > 0:
            SelectDict["$or"] = []
            for tID in secID:
                SelectDict["$or"].append({"ts_code": tID})

        StockStatus = self.db.Select("StockBackSys", "Bond_Status", SelectDict, TitleDict)
        for tInfo in StockStatus:
            ResultList.append(tInfo)
        df = pd.DataFrame(ResultList)
        return df

    def AdjFactorStockData(self,none_price, c_factor, db_new_factor):
        """
        :param none_price:    某股票当日未除权价格
        :param c_factor:      某股票当日tushare复权因子
        :param db_new_factor: 某股票在数据库中最新交易日对应的复权因子
        :return:
        """
        qfq_price = none_price * c_factor / db_new_factor
        return qfq_price

    def fillNaN(self,ResData, calendar_df):
        """
        :param df:          DataFrame价格数据
        :param calendar_df: DataFrame提取数据区间的交易日历
        :return:
        """
        trade_date_df = calendar_df[calendar_df["is_open"] == 1]
        trade_date_df = trade_date_df[["cal_date", "pretrade_date"]]
        trade_date_df.columns = ["trade_date", "pretrade_date"]

        # df.reset_index(drop=True,inplace=True)
        temp_columns = list(ResData.columns)
        out_df = trade_date_df.merge(ResData, on="trade_date", how="left")
        out_df.sort_values(by="trade_date", ascending=True, inplace=True)
        out_df.fillna(method="ffill", inplace=True)

        out_df = out_df[temp_columns]
        out_df.reset_index(drop=True, inplace=True)
        return out_df

    def MktEqudAdjGet_min(self,secID:list,tradeDate:list,beginDate:str,endDate:str,field:list,freq_m = 'min_1'):
        '''
        获取股票前复权行情数据
        :param secID:股票列表
        :param tradeDate:交易日期列表
        :param beginDate:开始日期
        :param endDate:结束日期
        :param field:需要返回的字段
        :param freq_m min_1  min_5 min_15 min_30 min_60
        :return: dataFrame复权后的行情数据
        '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        #获取最新交易日
        tLastTradeDateStr = self.GetLastTradeDate()
        tLastTradeDate = datetime.datetime.strptime(tLastTradeDateStr,"%Y%m%d")

        # 顺序获取
        SelectDict = {}
        SelectDict["$or"] = []
        for tID in secID:
            SelectDict["$or"].append({"ts_code": tID})

        #取股票列表最新复权因子
        TodayAdjDict = {}
        TodaySelectDict = SelectDict.copy()
        TodaySelectDict["trade_date"] = tLastTradeDateStr
        TodayAdjRes = self.db.Select("StockBackSys","Stock_Adj_"+str(tLastTradeDate.year),TodaySelectDict,{"_id":0})
        for tAdj in TodayAdjRes:
            TodayAdjDict[tAdj["ts_code"]] = tAdj["adj_factor"]

        #验证复权因子是否都存在
        LossStockList = []
        for tID in secID:
            if tID in TodayAdjDict:
                continue
            else:
                LossStockList.append(tID)

        #缺失的复权因子单独获取最新
        for tLossID in LossStockList:
            tStatus,tEndDate = self.GetStockTradeStatus(tLossID)
            if tStatus == False:
                tDelistDate = datetime.datetime.strptime(tEndDate,"%Y%m%d")
                tLossResDict = {}
                tLossSelectDict = {}
                tLossSelectDict["ts_code"] = tLossID
                i = 0
                while len(tLossResDict) == 0:
                    tCurDate = tDelistDate - datetime.timedelta(days=i)
                    i += 1
                    CurDateStr = tCurDate.strftime("%Y%m%d")
                    CurYear = tCurDate.year
                    tLossSelectDict["trade_date"] = CurDateStr
                    LossRes = self.db.Select("StockBackSys", "Stock_Adj_" + str(CurYear),tLossSelectDict, {"_id": 0})
                    for tRes in LossRes:
                        tLossResDict[tLossID] = tRes["adj_factor"]
                # 补上缺失
                TodayAdjDict[tLossID] = tLossResDict[tLossID]
            else:
                print("异常点：",tLossID,"上市状态无最新Adj")
                print(tLastTradeDateStr)

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                tDate = datetime.datetime.strptime(tDateStr, "%Y%m%d")
                tYear = tDate.year  # 定位数据库中表名
                SelectDict["trade_date"] = tDateStr

                # 取股票列表当日复权因子
                AdjDict = {}
                adjRes = self.db.Select("StockBackSys","Stock_Adj_"+str(tYear),SelectDict,{"_id":0})
                for adj in adjRes:
                    AdjDict[adj["ts_code"]] = adj["adj_factor"]

                for item_stk in secID:
                    table_name = None
                    if freq_m == 'min_1':
                        table_name = item_stk + '_1min'
                    elif freq_m == 'min_5':
                        table_name = item_stk + '_5min'
                    elif freq_m == 'min_15':
                        table_name = item_stk + '_15min'
                    elif freq_m == 'min_30':
                        table_name = item_stk + '_30min'
                    elif freq_m == 'min_60':
                        table_name = item_stk + '_60min'

                    # 筛选条件
                    SelectDict_temp = {}
                    SelectDict_temp['$or'] = []
                    for dt in tradeDate:
                        SelectDict_temp['$or'].append({'trade_date':dt})

                    res = self.db.Select(freq_m, table_name, SelectDict_temp, TitleDict)
                    for r in res:
                        if "open" in r:
                            r["open"] = self.AdjFactorStockData(r["open"],float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "high" in r:
                            r["high"] = self.AdjFactorStockData(r["high"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "low" in r:
                            r["low"] = self.AdjFactorStockData(r["low"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "close" in r:
                            r["close"] = self.AdjFactorStockData(r["close"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "pre_close" in r:
                            r["pre_close"] = self.AdjFactorStockData(r["pre_close"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                        ResultList.append(r)

        else:
            tbegin = datetime.datetime.strptime(beginDate, "%Y%m%d")
            tend = datetime.datetime.strptime(endDate, "%Y%m%d")

            # 跨表查询复权因子(用于复权因子跨表查找)
            tbeginYear = tbegin.year
            tendYear = tend.year
            if tendYear - tbeginYear > 0:  # 不在同一年
                tCurrentYear = tbeginYear

                AdjDict = {}
                while tCurrentYear <= tendYear:
                    if tCurrentYear == tbeginYear:
                        tThisYearBeginDate = tbegin
                    else:
                        tThisYearBeginDate = datetime.datetime(tCurrentYear, 1, 1)
                    tThisYearBeginDateStr = tThisYearBeginDate.strftime("%Y%m%d")

                    if tCurrentYear == tendYear:
                        tThisYearEndDate = tend
                    else:
                        tThisYearEndDate = datetime.datetime(tCurrentYear, 12, 31)
                    tThisYearEndDateStr = tThisYearEndDate.strftime("%Y%m%d")
                    SelectDict["trade_date"] = {"$gte": tThisYearBeginDateStr, "$lte": tThisYearEndDateStr}

                    # 取股票列表当日复权因子
                    adjRes = self.db.Select("StockBackSys", "Stock_Adj_" + str(tCurrentYear), SelectDict, {})
                    for adj in adjRes:
                        AdjDict[adj["_id"]] = adj["adj_factor"]
                    tCurrentYear += 1
            else:
                SelectDict["trade_date"] = {"$gte": beginDate, "$lte": endDate}
                # 取股票列表当日复权因子
                AdjDict = {}
                adjRes = self.db.Select("StockBackSys", "Stock_Adj_" + str(tbeginYear), SelectDict, {})
                for adj in adjRes:
                    AdjDict[adj["_id"]] = adj["adj_factor"]

            for item_stk in secID:
                table_name = None
                if freq_m == 'min_1':
                    table_name = item_stk + '_1min'
                elif freq_m == 'min_5':
                    table_name = item_stk + '_5min'
                elif freq_m == 'min_15':
                    table_name = item_stk + '_15min'
                elif freq_m == 'min_30':
                    table_name = item_stk + '_30min'
                elif freq_m == 'min_60':
                    table_name = item_stk + '_60min'

                # 筛选条件
                SelectDict_temp = {}
                SelectDict_temp["trade_date"] = {"$gte": beginDate, "$lte": endDate}

                res = self.db.Select(freq_m, table_name , SelectDict_temp, TitleDict)
                for r in res:
                    if "open" in r:
                        r["open"] = self.AdjFactorStockData(r["open"], float(AdjDict[r["ts_code"]+"_"+r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                    if "high" in r:
                        r["high"] = self.AdjFactorStockData(r["high"], float(AdjDict[r["ts_code"]+"_"+r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                    if "low" in r:
                        r["low"] = self.AdjFactorStockData(r["low"], float(AdjDict[r["ts_code"]+"_"+r["trade_date"]]), float(TodayAdjDict[r["ts_code"]]))
                    if "close" in r:
                        r["close"] = self.AdjFactorStockData(r["close"], float(AdjDict[r["ts_code"]+"_"+r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                    if "pre_close" in r:
                        r["pre_close"] = self.AdjFactorStockData(r["pre_close"], float(AdjDict[r["ts_code"]+"_"+r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                    ResultList.append(r)

        df = pd.DataFrame(ResultList)

        if tradeDateNum > 0:
            return df
        else:
            DateList = []
            DateSql = self.trade_cal(beginDate, endDate,1)
            for d in DateSql:
                DateList.append(d)
            DateDF = pd.DataFrame(DateList)

            if df.empty:
                return df
            else:
                return self.fillNaN(df, DateDF)

    def MktEqudAdjGet(self, secID: list, tradeDate: list, beginDate: str, endDate: str, field: list):
        '''
        获取股票前复权行情数据
        :param secID:股票列表
        :param tradeDate:交易日期列表
        :param beginDate:开始日期
        :param endDate:结束日期
        :param field:需要返回的字段
        :return: dataFrame复权后的行情数据
        '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        # 获取最新交易日
        tLastTradeDateStr = self.GetLastTradeDate()
        tLastTradeDate = datetime.datetime.strptime(tLastTradeDateStr, "%Y%m%d")

        # 顺序获取
        SelectDict = {}
        SelectDict["$or"] = []
        for tID in secID:
            SelectDict["$or"].append({"ts_code": tID})

        # 取股票列表最新复权因子
        TodayAdjDict = {}
        TodaySelectDict = SelectDict.copy()
        TodaySelectDict["trade_date"] = tLastTradeDateStr
        TodayAdjRes = self.db.Select("StockBackSys", "Stock_Adj_" + str(tLastTradeDate.year), TodaySelectDict,
                                     {"_id": 0})
        for tAdj in TodayAdjRes:
            TodayAdjDict[tAdj["ts_code"]] = tAdj["adj_factor"]

        # 验证复权因子是否都存在
        LossStockList = []
        for tID in secID:
            if tID in TodayAdjDict:
                continue
            else:
                LossStockList.append(tID)

        # 缺失的复权因子单独获取最新
        for tLossID in LossStockList:
            tStatus, tEndDate = self.GetStockTradeStatus(tLossID)
            if tStatus == False:
                tDelistDate = datetime.datetime.strptime(tEndDate, "%Y%m%d")
                tLossResDict = {}
                tLossSelectDict = {}
                tLossSelectDict["ts_code"] = tLossID
                i = 0
                while len(tLossResDict) == 0:
                    tCurDate = tDelistDate - datetime.timedelta(days=i)
                    i += 1
                    CurDateStr = tCurDate.strftime("%Y%m%d")
                    CurYear = tCurDate.year
                    tLossSelectDict["trade_date"] = CurDateStr
                    LossRes = self.db.Select("StockBackSys", "Stock_Adj_" + str(CurYear), tLossSelectDict, {"_id": 0})
                    for tRes in LossRes:
                        tLossResDict[tLossID] = tRes["adj_factor"]
                # 补上缺失
                TodayAdjDict[tLossID] = tLossResDict[tLossID]
            else:
                print("异常点：", tLossID, "上市状态无最新Adj")
                print(tLastTradeDateStr)

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                tDate = datetime.datetime.strptime(tDateStr, "%Y%m%d")
                tYear = tDate.year  # 定位数据库中表名
                SelectDict["trade_date"] = tDateStr

                # 取股票列表当日复权因子
                AdjDict = {}
                adjRes = self.db.Select("StockBackSys", "Stock_Adj_" + str(tYear), SelectDict, {"_id": 0})
                for adj in adjRes:
                    AdjDict[adj["ts_code"]] = adj["adj_factor"]

                tResList = []
                res = self.db.Select("StockBackSys", "Stock_DayLine_" + str(tYear), SelectDict, TitleDict)
                for r in res:
                    if "open" in r:
                        r["open"] = self.AdjFactorStockData(r["open"], float(AdjDict[r["ts_code"]]),
                                                            float(TodayAdjDict[r["ts_code"]]))
                    if "high" in r:
                        r["high"] = self.AdjFactorStockData(r["high"], float(AdjDict[r["ts_code"]]),
                                                            float(TodayAdjDict[r["ts_code"]]))
                    if "low" in r:
                        r["low"] = self.AdjFactorStockData(r["low"], float(AdjDict[r["ts_code"]]),
                                                           float(TodayAdjDict[r["ts_code"]]))
                    if "close" in r:
                        r["close"] = self.AdjFactorStockData(r["close"], float(AdjDict[r["ts_code"]]),
                                                             float(TodayAdjDict[r["ts_code"]]))
                    # if "pre_close" in r:
                    #     r["pre_close"] = self.AdjFactorStockData(r["pre_close"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                    tResList.append(r)

                i = 1
                while len(tResList) == 0:  # 当日为空，则往前寻找
                    CurDate = tDate - datetime.timedelta(days=i)
                    i += 1
                    CurDateStr = CurDate.strftime("%Y%m%d")
                    CurYear = CurDate.year
                    SelectDict["trade_date"] = CurDateStr

                    # 取股票列表当日复权因子
                    AdjDict = {}
                    adjRes = self.db.Select("StockBackSys", "Stock_Adj_" + str(CurYear), SelectDict, {"_id": 0})
                    for adj in adjRes:
                        AdjDict[adj["ts_code"]] = adj["adj_factor"]

                    res = self.db.Select("StockBackSys", "Stock_DayLine_" + str(CurYear), SelectDict, TitleDict)
                    for r in res:
                        if "open" in r:
                            r["open"] = self.AdjFactorStockData(r["open"], float(AdjDict[r["ts_code"]]),
                                                                float(TodayAdjDict[r["ts_code"]]))
                        if "high" in r:
                            r["high"] = self.AdjFactorStockData(r["high"], float(AdjDict[r["ts_code"]]),
                                                                float(TodayAdjDict[r["ts_code"]]))
                        if "low" in r:
                            r["low"] = self.AdjFactorStockData(r["low"], float(AdjDict[r["ts_code"]]),
                                                               float(TodayAdjDict[r["ts_code"]]))
                        if "close" in r:
                            r["close"] = self.AdjFactorStockData(r["close"], float(AdjDict[r["ts_code"]]),
                                                                 float(TodayAdjDict[r["ts_code"]]))
                        if "pre_close" in r:
                            r["pre_close"] = self.AdjFactorStockData(r["pre_close"], float(AdjDict[r["ts_code"]]),
                                                                     float(TodayAdjDict[r["ts_code"]]))
                        r["trade_date"] = tDateStr
                        tResList.append(r)

                for t in tResList:
                    ResultList.append(t)
        else:
            tbegin = datetime.datetime.strptime(beginDate, "%Y%m%d")
            tend = datetime.datetime.strptime(endDate, "%Y%m%d")

            # 跨表检查
            tbeginYear = tbegin.year
            tendYear = tend.year
            if tendYear - tbeginYear > 0:  # 不在同一年
                tCurrentYear = tbeginYear

                while tCurrentYear <= tendYear:
                    if tCurrentYear == tbeginYear:
                        tThisYearBeginDate = tbegin
                    else:
                        tThisYearBeginDate = datetime.datetime(tCurrentYear, 1, 1)
                    tThisYearBeginDateStr = tThisYearBeginDate.strftime("%Y%m%d")

                    if tCurrentYear == tendYear:
                        tThisYearEndDate = tend
                    else:
                        tThisYearEndDate = datetime.datetime(tCurrentYear, 12, 31)
                    tThisYearEndDateStr = tThisYearEndDate.strftime("%Y%m%d")

                    SelectDict["trade_date"] = {"$gte": tThisYearBeginDateStr, "$lte": tThisYearEndDateStr}

                    # 取股票列表当日复权因子
                    AdjDict = {}
                    adjRes = self.db.Select("StockBackSys", "Stock_Adj_" + str(tCurrentYear), SelectDict, {})
                    for adj in adjRes:
                        AdjDict[adj["_id"]] = adj["adj_factor"]

                    res = self.db.Select("StockBackSys", "Stock_DayLine_" + str(tCurrentYear), SelectDict, TitleDict)
                    for r in res:
                        if "open" in r:
                            r["open"] = self.AdjFactorStockData(r["open"],
                                                                float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),
                                                                float(TodayAdjDict[r["ts_code"]]))
                        if "high" in r:
                            r["high"] = self.AdjFactorStockData(r["high"],
                                                                float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),
                                                                float(TodayAdjDict[r["ts_code"]]))
                        if "low" in r:
                            r["low"] = self.AdjFactorStockData(r["low"],
                                                               float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),
                                                               float(TodayAdjDict[r["ts_code"]]))
                        if "close" in r:
                            r["close"] = self.AdjFactorStockData(r["close"],
                                                                 float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),
                                                                 float(TodayAdjDict[r["ts_code"]]))
                        if "pre_close" in r:
                            r["pre_close"] = self.AdjFactorStockData(r["pre_close"], float(
                                AdjDict[r["ts_code"] + "_" + r["trade_date"]]), float(TodayAdjDict[r["ts_code"]]))
                        ResultList.append(r)
                    tCurrentYear += 1
            else:  # 在同一年
                SelectDict["trade_date"] = {"$gte": beginDate, "$lte": endDate}
                # 取股票列表当日复权因子
                AdjDict = {}
                adjRes = self.db.Select("StockBackSys", "Stock_Adj_" + str(tbeginYear), SelectDict, {})
                for adj in adjRes:
                    AdjDict[adj["_id"]] = adj["adj_factor"]

                res = self.db.Select("StockBackSys", "Stock_DayLine_" + str(tbeginYear), SelectDict, TitleDict)
                for r in res:
                    if "open" in r:
                        r["open"] = self.AdjFactorStockData(r["open"],
                                                            float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),
                                                            float(TodayAdjDict[r["ts_code"]]))
                    if "high" in r:
                        r["high"] = self.AdjFactorStockData(r["high"],
                                                            float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),
                                                            float(TodayAdjDict[r["ts_code"]]))
                    if "low" in r:
                        r["low"] = self.AdjFactorStockData(r["low"],
                                                           float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),
                                                           float(TodayAdjDict[r["ts_code"]]))
                    if "close" in r:
                        r["close"] = self.AdjFactorStockData(r["close"],
                                                             float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),
                                                             float(TodayAdjDict[r["ts_code"]]))
                    if "pre_close" in r:
                        r["pre_close"] = self.AdjFactorStockData(r["pre_close"],
                                                                 float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),
                                                                 float(TodayAdjDict[r["ts_code"]]))
                    ResultList.append(r)

        df = pd.DataFrame(ResultList)

        if tradeDateNum > 0:
            return df
        else:
            DateList = []
            DateSql = self.trade_cal(beginDate, endDate, 1)
            for d in DateSql:
                DateList.append(d)
            DateDF = pd.DataFrame(DateList)
            return self.fillNaN(df, DateDF)

    def MktBonddGet(self,secID:list,tradeDate:list,beginDate:str,endDate:str,field:list):
        '''
        获取转债行情数据
        :param secID:转债代码列表
        :param tradeDate: 日期列表
        :param beginDate: 开始日期
        :param endDate: 结束日期
        :param field: 要返回的字段列表
        :return:dataFrame 转债行情数据
        '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        #确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        #顺序获取
        SelectDict = {}
        SelectDict["$or"] = []
        for tID in secID:
            SelectDict["$or"].append({"ts_code":tID})

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                tDate = datetime.datetime.strptime(tDateStr,"%Y%m%d")
                tYear = tDate.year#定位数据库中表名

                tResList = []
                SelectDict["trade_date"] = tDateStr
                res = self.db.Select("StockBackSys","CB_DayLine_"+str(tYear),SelectDict,TitleDict)
                for r in res:
                    tResList.append(r)

                while len(tResList) == 0:#当日为空，则往前寻找
                    CurDate = tDate - datetime.timedelta(days=1)
                    CurDateStr = CurDate.strftime("%Y%m%d")
                    CurYear = CurDate.year
                    SelectDict["trade_date"] = CurDateStr
                    res = self.db.Select("StockBackSys", "CB_DayLine_" + str(CurYear), SelectDict, TitleDict)
                    for r in res:
                        r["trade_date"] = tDateStr
                        tResList.append(r)
                for t in tResList:
                    ResultList.append(t)
        else:
            tbegin = datetime.datetime.strptime(beginDate,"%Y%m%d")
            tend = datetime.datetime.strptime(endDate,"%Y%m%d")
            #跨表检查
            tbeginYear = tbegin.year
            tendYear = tend.year
            if tendYear - tbeginYear > 0:#不在同一年
                tCurrentYear = tbeginYear

                while tCurrentYear <= tendYear:
                    if tCurrentYear == tbeginYear:
                        tThisYearBeginDate = tbegin
                    else:
                        tThisYearBeginDate = datetime.datetime(tCurrentYear,1,1)
                    tThisYearBeginDateStr  = tThisYearBeginDate.strftime("%Y%m%d")

                    if tCurrentYear == tendYear:
                        tThisYearEndDate = tend
                    else:
                        tThisYearEndDate = datetime.datetime(tCurrentYear,12,31)
                    tThisYearEndDateStr = tThisYearEndDate.strftime("%Y%m%d")

                    SelectDict["trade_date"] = {"$gte":tThisYearBeginDateStr,"$lte":tThisYearEndDateStr}
                    res = self.db.Select("StockBackSys","CB_DayLine_"+str(tCurrentYear),SelectDict,TitleDict)
                    for r in res:
                        ResultList.append(r)
                    tCurrentYear += 1
            else:#在同一年
                SelectDict["trade_date"] = {"$gte":beginDate,"$lte":endDate}
                res = self.db.Select("StockBackSys", "CB_DayLine_" + str(tbeginYear), SelectDict, TitleDict)
                for r in res:
                    ResultList.append(r)
        df = pd.DataFrame(ResultList)
        if tradeDateNum > 0:
            return df
        else:
            DateList = []
            DateSql = self.trade_cal(beginDate,endDate,1)
            for d in DateSql:
                DateList.append(d)
            DateDF = pd.DataFrame(DateList)
            return self.fillNaN(df,DateDF)

    def MktBonddGet_min(self,secID:list,tradeDate:list,beginDate:str,endDate:str,field:list,freq_m = 'min_1'):
        '''
        获取转债行情数据
        :param secID:转债代码列表
        :param tradeDate: 日期列表
        :param beginDate: 开始日期
        :param endDate: 结束日期
        :param field: 要返回的字段列表
        :return:dataFrame 转债行情数据
        '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        #确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                tDate = datetime.datetime.strptime(tDateStr,"%Y%m%d")

                # 确定表名和库名
                for item_stk in secID:
                    table_name = None
                    if freq_m == 'min_1':
                        table_name = item_stk + '_1min'
                    elif freq_m == 'min_5':
                        table_name = item_stk + '_5min'
                    elif freq_m == 'min_15':
                        table_name = item_stk + '_15min'
                    elif freq_m == 'min_30':
                        table_name = item_stk + '_30min'
                    elif freq_m == 'min_60':
                        table_name = item_stk + '_60min'

                    # 筛选条件
                    SelectDict_temp = {}
                    SelectDict_temp['$or'] = []
                    for dt in tradeDate:
                        SelectDict_temp['$or'].append({'trade_date': dt})

                    res = self.db.Select(freq_m, table_name, SelectDict_temp, TitleDict)
                    for r in res:
                        ResultList.append(r)
        else:
            # 确定库名和表名
            for item_stk in secID:
                table_name = None
                if freq_m == 'min_1':
                    table_name = item_stk + '_1min'
                elif freq_m == 'min_5':
                    table_name = item_stk + '_5min'
                elif freq_m == 'min_15':
                    table_name = item_stk + '_15min'
                elif freq_m == 'min_30':
                    table_name = item_stk + '_30min'
                elif freq_m == 'min_60':
                    table_name = item_stk + '_60min'

                # 筛选条件
                SelectDict_temp = {}
                SelectDict_temp["trade_date"] = {"$gte":beginDate,"$lte":endDate}
                res = self.db.Select(freq_m, table_name , SelectDict_temp, TitleDict)
                for r in res:
                    ResultList.append(r)

        df = pd.DataFrame(ResultList)
        if tradeDateNum > 0:
            return df
        else:
            DateList = []
            DateSql = self.trade_cal(beginDate,endDate,1)
            for d in DateSql:
                DateList.append(d)
            DateDF = pd.DataFrame(DateList)
            return self.fillNaN(df,DateDF)

    def MktIndexGet(self,secID:list,tradeDate:list,beginDate:str,endDate:str,field:list):
        '''
        获取指数行情
        :param secID:指数列表
        :param tradeDate: 日期列表
        :param beginDate: 开始日期
        :param endDate: 结束日期
        :param field: 返回字段列表
        :return: dataFrame 指数行情数据
        '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        # 顺序获取
        SelectDict = {}
        SelectDict["$or"] = []
        for tID in secID:
            SelectDict["$or"].append({"ts_code": tID})

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                SelectDict["trade_date"] = tDateStr
                res = self.db.Select("StockBackSys", "Index_DayLine", SelectDict, TitleDict)
                for r in res:
                    ResultList.append(r)
        else:
            SelectDict["trade_date"] = {"$gte": beginDate, "$lte": endDate}
            res = self.db.Select("StockBackSys", "Index_DayLine", SelectDict, TitleDict)
            for r in res:
                ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    def BondDailyInfo(self,secID:list,tradeDate:list,beginDate:str,endDate:str,field:list):
        '''
        获取可转债bondinfo
        :param secID:转债代码列表
        :param tradeDate:交易日列表
        :param beginDate:开始时间
        :param endDate:结束时间
        :param field:需要提取数据的字段，若为空，则返回全部字段
        :return: dataFrame BondInfo数据
        '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        #顺序获取
        SelectDict = {}
        if len(secID) > 0:
            SelectDict["$or"] = []
            for tID in secID:
                SelectDict["$or"].append({"ts_code": tID})

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                SelectDict["Trade_Date"] = str(tDateStr)
                res = self.db.Select("StockBackSys", "CB_BondInfo", SelectDict, TitleDict)
                for r in res:
                    if "ConvPremiumRatio" in r:
                        if r["ConvPremiumRatio"] == '':
                            r["ConvPremiumRatio"] = np.nan
                        else:
                            r["ConvPremiumRatio"] = float(r["ConvPremiumRatio"])
                    if "OutStandingBalance" in r:
                        if r["OutStandingBalance"] == '':
                            r["OutStandingBalance"] = np.nan
                        else:
                            r["OutStandingBalance"] = float(r["OutStandingBalance"])                                            
                    ResultList.append(r)
        else:
            SelectDict["Trade_Date"] = {"$gte": beginDate, "$lte": endDate}
            res = self.db.Select("StockBackSys", "CB_BondInfo", SelectDict, TitleDict)
            for r in res:
                if "ConvPremiumRatio" in r:
                    if r["ConvPremiumRatio"] == '':
                        r["ConvPremiumRatio"] = np.nan
                    else:
                        r["ConvPremiumRatio"] = float(r["ConvPremiumRatio"])
                if "OutStandingBalance" in r:
                    if r["OutStandingBalance"] == '':
                        r["OutStandingBalance"] = np.nan
                    else:
                        r["OutStandingBalance"] = float(r["OutStandingBalance"])                                                                     
                ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    def daily_basic(self,secID:list,tradeDate:list,beginDate:str,endDate:str,field:list):
        '''
        获取股票每日基础指标数据
        :param secID:股票列表
        :param tradeDate:交易日列表
        :param beginDate:开始日期
        :param endDate:结束日期
        :param field:需要返回的字段，为空则全选
        :return: dataFrame 股票每日基础指标数据
        '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        # 顺序获取
        SelectDict = {}
        if len(secID)>0:
            SelectDict["$or"] = []
            for tID in secID:
                SelectDict["$or"].append({"ts_code": tID})

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                tDate = datetime.datetime.strptime(tDateStr, "%Y%m%d")
                tYear = tDate.year  # 定位数据库中表名

                tResList = []
                SelectDict["trade_date"] = tDateStr
                res = self.db.Select("StockBackSys", "Stock_DailyBasic_" + str(tYear), SelectDict, TitleDict)
                for r in res:
                    tResList.append(r)

                while len(tResList) == 0:  # 当日为空，则往前寻找
                    CurDate = tDate - datetime.timedelta(days=1)
                    CurDateStr = CurDate.strftime("%Y%m%d")
                    CurYear = CurDate.year
                    SelectDict["trade_date"] = CurDateStr
                    res = self.db.Select("StockBackSys", "Stock_DailyBasic_" + str(CurYear), SelectDict, TitleDict)
                    for r in res:
                        r["trade_date"] = tDateStr
                        if "close" in r:
                            r["close"] = 0.0
                        if "turnover_rate" in r:
                            r["turnover_rate"] = 0.0
                        if "trunover_rate_f" in r:
                            r["trunover_rate_f"] = 0.0
                        if "volume_ratio" in r:
                            r["volume_ratio"] = 0.0
                        tResList.append(r)
                for t in tResList:
                    ResultList.append(t)
        else:
            #获取区间交易日列表
            TradeDayList,BetweenList = self.GetListTradeDateBetween(beginDate,endDate)

            tResList = []
            tbegin = datetime.datetime.strptime(beginDate, "%Y%m%d")
            tend = datetime.datetime.strptime(endDate, "%Y%m%d")
            # 跨表检查
            tbeginYear = tbegin.year
            tendYear = tend.year
            if tendYear - tbeginYear > 0:  # 不在同一年
                tCurrentYear = tbeginYear

                while tCurrentYear <= tendYear:
                    if tCurrentYear == tbeginYear:
                        tThisYearBeginDate = tbegin
                    else:
                        tThisYearBeginDate = datetime.datetime(tCurrentYear, 1, 1)
                    tThisYearBeginDateStr = tThisYearBeginDate.strftime("%Y%m%d")

                    if tCurrentYear == tendYear:
                        tThisYearEndDate = tend
                    else:
                        tThisYearEndDate = datetime.datetime(tCurrentYear, 12, 31)
                    tThisYearEndDateStr = tThisYearEndDate.strftime("%Y%m%d")

                    SelectDict["trade_date"] = {"$gte": tThisYearBeginDateStr, "$lte": tThisYearEndDateStr}
                    res = self.db.Select("StockBackSys", "Stock_DailyBasic_" + str(tCurrentYear), SelectDict, TitleDict)
                    for r in res:
                        tResList.append(r)
                    tCurrentYear += 1
            else:  # 在同一年
                SelectDict["trade_date"] = {"$gte": beginDate, "$lte": endDate}
                res = self.db.Select("StockBackSys", "Stock_DailyBasic_" + str(tbeginYear), SelectDict, TitleDict)
                for r in res:
                    tResList.append(r)

            #检查交易日是否有缺失
            LastDay = TradeDayList[0]
            LossDateList = []
            for tDay in TradeDayList:
                HaveData = False
                for xDict in tResList:
                    if xDict["trade_date"] == tDay:
                        HaveData = True
                        break
                if HaveData == False:
                    LossDateList.append((LastDay,tDay))
                else:
                    LastDay = tDay

            #缺失填充
            for tLastDay,tLossDay in LossDateList:
                for xDict in tResList:
                    if xDict["trade_date"] == tLastDay:
                        tLastDict = xDict.copy()
                        tLastDict["trade_date"] = tLossDay
                        if "close" in tLastDict:
                            tLastDict["close"] = 0.0
                        if "turnover_rate" in tLastDict:
                            tLastDict["turnover_rate"] = 0.0
                        if "trunover_rate_f" in tLastDict:
                            tLastDict["trunover_rate_f"] = 0.0
                        if "volume_ratio" in tLastDict:
                            tLastDict["volume_ratio"] = 0.0
                        ResultList.append(tLastDict)
                        break

            for t in tResList:
                ResultList.append(t)

        df = pd.DataFrame(ResultList)
        if "trade_date" in field:                         
            df.sort_values(by="trade_date",ascending=True,inplace=True)
        return df

    def trade_cal(self,start_date:str,end_date:str,ResType=0):
        '''
        获取交易日历
        :param start_date:
        :param end_date:
        :param ResType:返回数据类型，默认为返回dataFrame类型，为1返回日历信息List
        :return:
        '''
        TitleDict = {}
        TitleDict["_id"] = 0

        SelectDict = {}
        SelectDict["cal_date"] = {"$gte":start_date,"$lte":end_date}

        ResultList = []
        res = self.db.Select("StockBackSys", "trade_calendar", SelectDict, TitleDict)
        for r in res:
            ResultList.append(r)
        if ResType == 1:
            return ResultList
        else:
            df = pd.DataFrame(ResultList)
            return df

    def GetListTradeDateBetween(self,start_date:str,end_date:str):
        '''
        获取指定区间的交易日和自然日列表
        :param start_date:
        :param end_date:
        :return 交易日list，自然日list
        '''
        TitleDict = {}
        TitleDict["_id"] = 0

        SelectDict = {}
        SelectDict["cal_date"] = {"$gte": start_date, "$lte": end_date}

        TradeDayList = []#交易日列表
        ResultList = []#自然日列表
        res = self.db.Select("StockBackSys", "trade_calendar", SelectDict, TitleDict)
        for r in res:
            if r["is_open"] == 1:
                TradeDayList.append(r["cal_date"])
            ResultList.append(r["cal_date"])
        return TradeDayList,ResultList

    def cb_call(self,secID:list,tradeDate:list,field:list):
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        # 顺序获取
        SelectDict = {}
        if len(secID) > 0:
            SelectDict["$or"] = []
            for tID in secID:
                SelectDict["$or"].append({"ts_code": tID})

        for tDateStr in tradeDate:
            SelectDict["ann_date"] = {"$lte":tDateStr}
            res = self.db.Select("StockBackSys", "CB_Call", SelectDict, TitleDict)
            for r in res:
                ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    def index_weight(self,IndexCode:str,tradeDate:list):
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0

        # 顺序获取
        SelectDict = {}
        if IndexCode != "":
            SelectDict["index_code"] = IndexCode

        for tDateStr in tradeDate:
            tDate = datetime.datetime.strptime(tDateStr, "%Y%m%d")
            CurrentYear = tDate.year  # 定位数据库中表名
            SelectDict["trade_date"] = str(tDateStr)
            res = self.db.Select("StockBackSys", "Index_Weight_" + str(CurrentYear), SelectDict, TitleDict)
            for r in res:
                ResultList.append(r)

            i = 1
            while len(ResultList) == 0:  # 当日为空，则往前寻找
                CurDate = tDate - datetime.timedelta(days=i)
                i += 1
                CurDateStr = CurDate.strftime("%Y%m%d")
                CurYear = CurDate.year
                SelectDict["trade_date"] = CurDateStr
                res = self.db.Select("StockBackSys", "Index_Weight_" + str(CurYear), SelectDict, TitleDict)
                for r in res:
                    ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    def SW_classify(self,Level:str):
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0

        SelectDict = {}
        SelectDict["level"] = str(Level)
        res = self.db.Select("StockBackSys", "SW_Classify", SelectDict, TitleDict)
        for r in res:
            ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    def SW_Member(self,SWCode:str):
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0

        SelectDict = {}
        SelectDict["index_code"] = str(SWCode)
        res = self.db.Select("StockBackSys", "SW_Member", SelectDict, TitleDict)
        for r in res:
            ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    def SW_StockInfo(self,secID:list,TradeDate:str):
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0

        # 顺序获取
        SelectDict = {}
        SelectDict["$or"] = []
        for tID in secID:
            SelectDict["$or"].append({"con_code": tID})

        SelectDict["in_date"] = {"$lte": TradeDate}
        res = self.db.Select("StockBackSys", "SW_Member", SelectDict, TitleDict)
        for r in res:
            ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    def Stock_Suspend(self,secID:list,TradeDate:str):
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0

        # 顺序获取
        SelectDict = {}
        if len(secID)>0:
            SelectDict["$or"] = []
            for tID in secID:
                SelectDict["$or"].append({"ts_code": tID})

        SelectDict["trade_date"] = TradeDate
        res = self.db.Select("StockBackSys", "Stock_Suspend", SelectDict, TitleDict)
        for r in res:
            ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    def SW_Daily(self,secID:list,tradeDate:list,beginDate:str,endDate:str,field:list):
        '''
        获取指数行情
        :param secID:指数列表
        :param tradeDate: 日期列表
        :param beginDate: 开始日期
        :param endDate: 结束日期
        :param field: 返回字段列表
        :return: dataFrame 指数行情数据
        '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        # 顺序获取
        SelectDict = {}
        if len(secID) > 0:
            SelectDict["$or"] = []
            for tID in secID:
                SelectDict["$or"].append({"IndexID": tID})

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                SelectDict["tradeDate"] = tDateStr
                res = self.db.Select("StockBackSys", "SW_L1_Daily", SelectDict, TitleDict)
                for r in res:
                    ResultList.append(r)
        else:
            SelectDict["tradeDate"] = {"$gte": beginDate, "$lte": endDate}
            res = self.db.Select("StockBackSys", "SW_L1_Daily", SelectDict, TitleDict)
            for r in res:
                ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    def Fund_Daily(self,secID:list,tradeDate:list,beginDate:str,endDate:str,field:list):
        tradeDateNum = len(tradeDate)
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        # 获取最新交易日
        tLastTradeDateStr = self.GetLastTradeDate()
        tLastTradeDate = datetime.datetime.strptime(tLastTradeDateStr, "%Y%m%d")

        # 顺序获取
        SelectDict = {}
        SelectDict["$or"] = []
        for tID in secID:
            SelectDict["$or"].append({"ts_code": tID})

        # 取基金列表最新复权因子
        TodayAdjDict = {}
        TodaySelectDict = SelectDict.copy()
        TodaySelectDict["trade_date"] = tLastTradeDateStr
        TodayAdjRes = self.db.Select("StockBackSys", "Fund_Adj_" + str(tLastTradeDate.year), TodaySelectDict,{"_id": 0})
        for tAdj in TodayAdjRes:
            TodayAdjDict[tAdj["ts_code"]] = tAdj["adj_factor"]

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                tDate = datetime.datetime.strptime(tDateStr, "%Y%m%d")
                tYear = tDate.year  # 定位数据库中表名
                SelectDict["trade_date"] = tDateStr

                # 取基金列表当日复权因子
                AdjDict = {}
                adjRes = self.db.Select("StockBackSys", "Fund_Adj_" + str(tYear), SelectDict, {"_id": 0})
                for adj in adjRes:
                    AdjDict[adj["ts_code"]] = adj["adj_factor"]

                tResList = []
                res = self.db.Select("StockBackSys", "Fund_Daily_" + str(tYear), SelectDict, TitleDict)
                for r in res:
                    if r["ts_code"] in AdjDict:
                        if "open" in r:
                            r["open"] = self.AdjFactorStockData(r["open"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "high" in r:
                            r["high"] = self.AdjFactorStockData(r["high"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "low" in r:
                            r["low"] = self.AdjFactorStockData(r["low"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "close" in r:
                            r["close"] = self.AdjFactorStockData(r["close"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "pre_close" in r:
                            r["pre_close"] = self.AdjFactorStockData(r["pre_close"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                    tResList.append(r)
                i = 1
                while len(tResList) == 0:  # 当日为空，则往前寻找
                    CurDate = tDate - datetime.timedelta(days=i)
                    i += 1
                    CurDateStr = CurDate.strftime("%Y%m%d")
                    CurYear = CurDate.year
                    SelectDict["trade_date"] = CurDateStr

                    #获取当日复权因子
                    AdjDict = {}
                    adjRes = self.db.Select("StockBackSys", "Fund_Adj_" + str(CurYear), SelectDict, {"_id": 0})
                    for adj in adjRes:
                        AdjDict[adj["ts_code"]] = adj["adj_factor"]

                    #获取
                    res = self.db.Select("StockBackSys", "Fund_Daily_" + str(CurYear), SelectDict, TitleDict)
                    for r in res:
                        if r["ts_code"] in AdjDict:#指数基金无复权因子，不复权
                            if "open" in r:
                                r["open"] = self.AdjFactorStockData(r["open"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                            if "high" in r:
                                r["high"] = self.AdjFactorStockData(r["high"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                            if "low" in r:
                                r["low"] = self.AdjFactorStockData(r["low"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                            if "close" in r:
                                r["close"] = self.AdjFactorStockData(r["close"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                            if "pre_close" in r:
                                r["pre_close"] = self.AdjFactorStockData(r["pre_close"], float(AdjDict[r["ts_code"]]),float(TodayAdjDict[r["ts_code"]]))
                        r["trade_date"] = tDateStr
                        tResList.append(r)

                for t in tResList:
                    ResultList.append(t)
        else:
            tbegin = datetime.datetime.strptime(beginDate, "%Y%m%d")
            tend = datetime.datetime.strptime(endDate, "%Y%m%d")

            # 跨表检查
            tbeginYear = tbegin.year
            tendYear = tend.year
            if tendYear - tbeginYear > 0:  # 不在同一年
                tCurrentYear = tbeginYear

                while tCurrentYear <= tendYear:
                    if tCurrentYear == tbeginYear:
                        tThisYearBeginDate = tbegin
                    else:
                        tThisYearBeginDate = datetime.datetime(tCurrentYear, 1, 1)
                    tThisYearBeginDateStr = tThisYearBeginDate.strftime("%Y%m%d")

                    if tCurrentYear == tendYear:
                        tThisYearEndDate = tend
                    else:
                        tThisYearEndDate = datetime.datetime(tCurrentYear, 12, 31)
                    tThisYearEndDateStr = tThisYearEndDate.strftime("%Y%m%d")

                    SelectDict["trade_date"] = {"$gte": tThisYearBeginDateStr, "$lte": tThisYearEndDateStr}

                    # 取股票列表当日复权因子
                    AdjDict = {}
                    adjRes = self.db.Select("StockBackSys", "Fund_Adj_" + str(tCurrentYear), SelectDict, {})
                    for adj in adjRes:
                        AdjDict[adj["_id"]] = adj["adj_factor"]

                    res = self.db.Select("StockBackSys", "Fund_Daily_" + str(tCurrentYear), SelectDict, TitleDict)
                    for r in res:
                        if r["ts_code"] in AdjDict:  # 指数基金无复权因子，不复权
                            if "open" in r:
                                r["open"] = self.AdjFactorStockData(r["open"],float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                            if "high" in r:
                                r["high"] = self.AdjFactorStockData(r["high"],float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                            if "low" in r:
                                r["low"] = self.AdjFactorStockData(r["low"],float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                            if "close" in r:
                                r["close"] = self.AdjFactorStockData(r["close"],float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                            if "pre_close" in r:
                                r["pre_close"] = self.AdjFactorStockData(r["pre_close"], float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]), float(TodayAdjDict[r["ts_code"]]))
                        ResultList.append(r)
                    tCurrentYear += 1
            else:  # 在同一年
                SelectDict["trade_date"] = {"$gte": beginDate, "$lte": endDate}
                # 取股票列表当日复权因子
                AdjDict = {}
                adjRes = self.db.Select("StockBackSys", "Fund_Adj_" + str(tbeginYear), SelectDict, {})
                for adj in adjRes:
                    AdjDict[adj["_id"]] = adj["adj_factor"]

                res = self.db.Select("StockBackSys", "Fund_Daily_" + str(tbeginYear), SelectDict, TitleDict)
                for r in res:
                    if r["ts_code"] in AdjDict:  # 指数基金无复权因子，不复权
                        if "open" in r:
                            r["open"] = self.AdjFactorStockData(r["open"],float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "high" in r:
                            r["high"] = self.AdjFactorStockData(r["high"],float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "low" in r:
                            r["low"] = self.AdjFactorStockData(r["low"],float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "close" in r:
                            r["close"] = self.AdjFactorStockData(r["close"],float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                        if "pre_close" in r:
                            r["pre_close"] = self.AdjFactorStockData(r["pre_close"],float(AdjDict[r["ts_code"] + "_" + r["trade_date"]]),float(TodayAdjDict[r["ts_code"]]))
                    ResultList.append(r)

        df = pd.DataFrame(ResultList)

        if tradeDateNum > 0:
            return df
        else:
            DateList = []
            DateSql = self.trade_cal(beginDate, endDate, 1)
            for d in DateSql:
                DateList.append(d)
            DateDF = pd.DataFrame(DateList)
            return self.fillNaN(df, DateDF)

    def StockAnalysisData(self,secID:list,tradeDate:list,beginDate:str,endDate:str,field:list):
    
        '''
       获取指数行情
       :param secID:指数列表
       :param tradeDate: 日期列表
       :param beginDate: 开始日期
       :param endDate: 结束日期
       :param field: 返回字段列表
       :return: dataFrame 指数行情数据
       '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        # 确定返回字段
        TitleDict = {}         # 返回字段字典
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        # 顺序获取
        SelectDict = {}        # 筛选条件字典
        if len(secID) > 0:
            SelectDict["$or"] = []
            for tID in secID:
                SelectDict["$or"].append({"secID": tID})

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                tYear = tDateStr[0:4]
                SelectDict["tradeDate"] = int(tDateStr)
                res = self.db.Select("StockBackSys", "AnalysisData_"+tYear, SelectDict, TitleDict)
                for r in res:
                    ResultList.append(r)
        else:
            tbegin = datetime.datetime.strptime(beginDate, "%Y%m%d")
            tend = datetime.datetime.strptime(endDate, "%Y%m%d")

            # 跨表检查
            tbeginYear = tbegin.year
            tendYear = tend.year

            # 跨年跨表查询
            if tendYear - tbeginYear > 0:  # 不在同一年
                tCurrentYear = tbeginYear

                while tCurrentYear <= tendYear:
                    if tCurrentYear == tbeginYear:
                        tThisYearBeginDate = tbegin
                    else:
                        tThisYearBeginDate = datetime.datetime(tCurrentYear, 1, 1)
                    tThisYearBeginDateStr = tThisYearBeginDate.strftime("%Y%m%d")

                    if tCurrentYear == tendYear:
                        tThisYearEndDate = tend
                    else:
                        tThisYearEndDate = datetime.datetime(tCurrentYear, 12, 31)
                    tThisYearEndDateStr = tThisYearEndDate.strftime("%Y%m%d")

                    SelectDict["tradeDate"] = {"$gte": int(tThisYearBeginDateStr), "$lte": int(tThisYearEndDateStr)}


                    res = self.db.Select("StockBackSys", "AnalysisData_" + str(tCurrentYear), SelectDict, TitleDict)
                    for r in res:
                        ResultList.append(r)
                    tCurrentYear += 1
            # 同一年单表查询
            else:
                SelectDict["tradeDate"] = {"$gte": int(beginDate), "$lte": int(endDate)}

                res = self.db.Select("StockBackSys", "AnalysisData_" + str(tbeginYear), SelectDict, TitleDict)
                for r in res:
                    ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    def StockBakDaily(self,tradeDate:list,field:list):
        ResultList = []

        # 确定返回字段
        TitleDict = {}
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        # 顺序获取
        SelectDict = {}

        for tDateStr in tradeDate:
            SelectDict["trade_date"] = tDateStr
            res = self.db.Select("StockBackSys", "stock_bak_daily_"+tDateStr[0:4], SelectDict, TitleDict)
            for r in res:
                ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df
    
    # 北向数据
    def NorthCapitalData(self,secID:list,tradeDate:list,beginDate:str,endDate:str,field:list):
        '''
       :param secID:股票代码列表
       :param tradeDate: 日期列表
       :param beginDate: 开始日期
       :param endDate: 结束日期
       :param field: 返回字段列表
       :return: dataFrame 数据
       '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        # 确定返回字段
        TitleDict = {}  # 返回字段字典
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        # 顺序获取
        SelectDict = {}  # 筛选条件字典
        if len(secID) > 0:
            SelectDict["$or"] = []
            for tID in secID:
                SelectDict["$or"].append({"ts_code": tID})

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                tYear = tDateStr[0:4]
                SelectDict["trade_date"] = str(tDateStr)
                res = self.db.Select("StockBackSys", "HK_Hold_" + tYear, SelectDict, TitleDict)
                for r in res:
                    ResultList.append(r)
        else:
            tbegin = datetime.datetime.strptime(beginDate, "%Y%m%d")
            tend = datetime.datetime.strptime(endDate, "%Y%m%d")

            # 跨表检查
            tbeginYear = tbegin.year
            tendYear = tend.year

            # 跨年跨表查询
            if tendYear - tbeginYear > 0:  # 不在同一年
                tCurrentYear = tbeginYear

                while tCurrentYear <= tendYear:
                    if tCurrentYear == tbeginYear:
                        tThisYearBeginDate = tbegin
                    else:
                        tThisYearBeginDate = datetime.datetime(tCurrentYear, 1, 1)
                    tThisYearBeginDateStr = tThisYearBeginDate.strftime("%Y%m%d")

                    if tCurrentYear == tendYear:
                        tThisYearEndDate = tend
                    else:
                        tThisYearEndDate = datetime.datetime(tCurrentYear, 12, 31)
                    tThisYearEndDateStr = tThisYearEndDate.strftime("%Y%m%d")

                    SelectDict["trade_date"] = {"$gte": int(tThisYearBeginDateStr), "$lte": int(tThisYearEndDateStr)}

                    res = self.db.Select("StockBackSys", "HK_Hold_" + str(tCurrentYear), SelectDict, TitleDict)
                    for r in res:
                        ResultList.append(r)
                    tCurrentYear += 1
            # 同一年单表查询
            else:
                SelectDict["trade_date"] = {"$gte": str(beginDate), "$lte": str(endDate)}

                res = self.db.Select("StockBackSys", "HK_Hold_" + str(tbeginYear), SelectDict, TitleDict)
                for r in res:
                    ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df

    # 小型价值股基本面数据
    def ShortValueData(self, secID: list, tradeDate: list, beginDate: str, endDate: str, field: list):

        '''
       :param secID:股票列表
       :param tradeDate: 日期列表
       :param beginDate: 开始日期
       :param endDate: 结束日期
       :param field: 返回字段列表
       :return: dataFrame 指数行情数据
       '''
        tradeDateNum = len(tradeDate)
        ResultList = []

        # 确定返回字段
        TitleDict = {}  # 返回字段字典
        TitleDict["_id"] = 0
        for tKey in field:
            TitleDict[tKey] = 1

        # 顺序获取
        SelectDict = {}  # 筛选条件字典
        if len(secID) > 0:
            SelectDict["$or"] = []
            for tID in secID:
                SelectDict["$or"].append({"secID": tID})

        if tradeDateNum > 0:
            for tDateStr in tradeDate:
                tYear = tDateStr[0:4]
                SelectDict["tradeDate"] = str(tDateStr)
                res = self.db.Select("StockBackSys", "ShortMktValueData_" + tYear, SelectDict, TitleDict)
                for r in res:
                    ResultList.append(r)
        else:
            tbegin = datetime.datetime.strptime(beginDate, "%Y%m%d")
            tend = datetime.datetime.strptime(endDate, "%Y%m%d")

            # 跨表检查
            tbeginYear = tbegin.year
            tendYear = tend.year

            # 跨年跨表查询
            if tendYear - tbeginYear > 0:  # 不在同一年
                tCurrentYear = tbeginYear

                while tCurrentYear <= tendYear:
                    if tCurrentYear == tbeginYear:
                        tThisYearBeginDate = tbegin
                    else:
                        tThisYearBeginDate = datetime.datetime(tCurrentYear, 1, 1)
                    tThisYearBeginDateStr = tThisYearBeginDate.strftime("%Y%m%d")

                    if tCurrentYear == tendYear:
                        tThisYearEndDate = tend
                    else:
                        tThisYearEndDate = datetime.datetime(tCurrentYear, 12, 31)
                    tThisYearEndDateStr = tThisYearEndDate.strftime("%Y%m%d")

                    SelectDict["tradeDate"] = {"$gte": str(tThisYearBeginDateStr), "$lte": str(tThisYearEndDateStr)}

                    res = self.db.Select("StockBackSys", "ShortMktValueData_" + str(tCurrentYear), SelectDict, TitleDict)
                    for r in res:
                        ResultList.append(r)
                    tCurrentYear += 1
            # 同一年单表查询
            else:
                SelectDict["tradeDate"] = {"$gte": str(beginDate), "$lte": str(endDate)}

                res = self.db.Select("StockBackSys", "ShortMktValueData_" + str(tbeginYear), SelectDict, TitleDict)
                for r in res:
                    ResultList.append(r)

        df = pd.DataFrame(ResultList)
        return df


# if __name__ == '__main__':
#     obj = SelectFromMongo()
#     TEMP_DF = obj.GetBondInfoStatus([])
#     TEMP_DF.to_csv('./temp.csv',index=None)