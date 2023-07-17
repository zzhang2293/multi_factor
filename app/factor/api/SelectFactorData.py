import time
import datetime
import pandas as pd
import numpy as np
from .MongoConnect import MongoConnect

class GetFactorFromMongoDB():
    def __init__(self):
        self.db = MongoConnect()
        self.db.Connect("192.168.1.18", "17629", "fww", "fww8888", "admin")
        self.my_dbname = "StockMutiFactor"

    def func01(self,df):
        '''
        :param df: DataFrame CollectionName FactorName
        :return:
        '''
        temp_lst = list(df['FactorName'])
        return temp_lst

    # 查询因子对应的数据表名
    def GetFactorCollectionName(self, factorname_lst:list):
        '''
        :param factorname_lst: list 因子名称列表,可空，若为空则返回表中所有的数据
        :return: return_dict_0 dict 数据表对应的因子名称
                 return_dict dict 因子对应的数据表名
        '''
        return_dict_0 = {}
        return_dict = {}
        ResultList = []

        if factorname_lst:
            TitleDict = {}
            TitleDict["_id"] = 0

            SelectDict = {}
            SelectDict["$or"] = []
            for idx in factorname_lst:
                SelectDict["$or"].append({"FactorName": idx})
        else:
            TitleDict = {}
            TitleDict["_id"] = 0
            SelectDict = {}

        result_name = self.db.Select(self.my_dbname, "FactorWithCollectionName", SelectDict, TitleDict)

        for tInfo in result_name:
            ResultList.append(tInfo)

        df = pd.DataFrame(ResultList)
        if not df.empty:
            result_s = df.groupby("CollectionName").apply(self.func01)
            for item in result_s.index:
                return_dict_0[item] = result_s[item]                         # 数据表对应的因子名称
            return_dict = dict(zip(df['FactorName'], df['CollectionName']))  # 因子对应的数据表名
        return return_dict_0,return_dict

    # 查询数据库中因子表对应的因子名称
    def GetAllFactorName(self):
        TitleDict = {}
        TitleDict["_id"] = 0
        SelectDict = {}
        result_name = self.db.Select(self.my_dbname, "FactorWithCollectionName", SelectDict, TitleDict)

        ResultList = []
        for tInfo in result_name:
            ResultList.append(tInfo)

        df = pd.DataFrame(ResultList)
        factorname_lst = []
        if not df.empty:
            factorname_lst = list(df['FactorName'])
        return factorname_lst


    # 根据因子名称查询全市场股票因子值数据
    def GetStockFactor(self,data_date:str,stk_lst:list,factorname_lst:list):
        '''
        :param data_date:      获取因子数据的日期，必填且不能为空
        :param stk_lst:        获取因子值的股票列表,可空，当为空时返回数据库中该因子所有股票的因子数据,这里最好不为空
        :param factorname_lst: 因子名称列表,若为空则返回所有因子的数据，这里最好不要写空
        :return:
        '''
        # 需要提取的因子按照表分类
        collect_facname_d,facname_collect_d = self.GetFactorCollectionName(factorname_lst)

        result_df = pd.DataFrame()

        # 遍历查询因子表
        for collect in collect_facname_d:                  # 遍历因子表
            ResultList = []
            temp_facname = collect_facname_d[collect]      # 当前表对应的因子名称字段

            # 确定需要的返回字段（这里也需要返回_id）
            TitleDict = {}
            TitleDict["_id"] = 1
            for tKey in temp_facname:
                TitleDict[tKey] = 1  #因子名称

            # 筛选条件
            SelectDict = {}
            SelectDict["_id.trade_date"] = data_date
            if stk_lst:
                SelectDict["$or"] = []
                for tID in stk_lst:
                    SelectDict["$or"].append({"_id.ts_code": tID})

            # 查询数据库
            res = self.db.Select(self.my_dbname,collect,SelectDict,TitleDict)

            for r in res:  # 每个r代表表中的每一行，这里只是将这一行以字典的形式表示：键为表字段，值为该行对应字段的值
                # r["trade_date"] = r["_id"]["trade_date"]
                r["ts_code"] = r["_id"]["ts_code"]
                del (r["_id"])      # 删除_id字段
                ResultList.append(r)

            # 整理返回结果
            if ResultList:
                df = pd.DataFrame(ResultList)
                df.set_index("ts_code",drop=True,inplace=True)
                if result_df.empty:
                    result_df  = df.copy()
                else:
                    for name in df.columns:
                        result_df[name] = df[name]
        return result_df