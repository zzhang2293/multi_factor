from io import BytesIO
from django.template import loader
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.http import FileResponse
import json
import csv
import pandas as pd
from app.factor.factorModel import factorModel
import zipfile
import os

analysisMethod = factorModel()
scope = analysisMethod.universe_index

'''
    result that will show on web page
    these global variables are used to convert to csv file and send to front
    data structure: 
    combinedIC(dict): ['month': [list of month], 'IC': [list of ic value], 'cumulative': [list of cumulative ic]]

    df_group_net(DataFrame): [index = trade_date, column = ['group_0', 'group_1', 'group2', 'group3', 'longshort_hedge']]

    df_group_alpha(DataFrame): same as df_group_net, 

    df_bt_indicator(DataFrame): [index = [1,2,3,4,5 ...], column = ["group", "年化收益率","夏普比率","最大回撤"]]

    df_bt_alpha)indicator(DataFrame) : [index = [1,2,3,4,5 ....], column = ['group', "年化超额收益率","超额最大回撤","calmar"]]

'''

def members(request):
    template = loader.get_template('myfirst.html')
    print("get info")
    return HttpResponse(template.render())


@csrf_exempt
def get_factors(request):
    allFactor = analysisMethod.allfactorname_lst
    return JsonResponse(allFactor, safe=False)

@csrf_exempt
def collect_data(requests):

    print("START COLLECTING DATA")
    if requests.method == "POST":
        data = json.loads(requests.body)
        if (data["group"] != ""):
            analysisMethod.groupnum = int(data["group"])
        if data["freq"] != "":
            analysisMethod.trade_freq = data["freq"]
        factor = data["factor"]
        factor_dict = data["factor_val"]
        if len(data["scope"]) > 0:
            analysisMethod.universe_index = data["scope"]
        # modify the format of startdate and enddate
        if data["startDate"] != "":
            startDateModified = data["startDate"].split("T")[0]
            startDateModified = startDateModified.replace("-", "")
            analysisMethod.start = startDateModified
        if data["endDate"] != "":
            endDateModified = data["endDate"].split("T")[0]
            endDateModified = endDateModified.replace("-", "")
            analysisMethod.end = endDateModified 
        if data['benchmark'] != '':
            analysisMethod.benchmark = data['benchmark']
        if data['factor_weight_mode'] != '':
            analysisMethod.factorWeightMode = data['factor_weight_mode']
    
        factor_list = []
        if len(factor) > 0:
            for val in factor:
                factor_list.append(factor_dict[val]["label"])
            analysisMethod.factor_name_lst = factor_list
        if len(data['rankLowestFirst']) > 0:
            analysisMethod.rankLowestFirst = data['rankLowestFirst']
        if len(data['factor_weight']) > 0:
            analysisMethod.userDefinedFactorWeights = [float(i) for i in data['factor_weight'].split(" ")]
        if data['EvalPeriod'] != '':
            analysisMethod.EvalPeriod = int(data['EvalPeriod'])
        if data['minEvalPeriod'] != '':
            analysisMethod.minEvalPeriod = int(data['minEvalPeriod'])
        if data['stockWeightMode'] != '':
            analysisMethod.stockWeightMode = data['stockWeightMode']
        print("factor_name_lst: ", analysisMethod.factor_name_lst)
        print("start: ", analysisMethod.start)
        print("end: ", analysisMethod.end)
        print("trade_freq: ", analysisMethod.trade_freq)
        print("groupnum: ", analysisMethod.groupnum)
        print("universe_index: ", analysisMethod.universe_index)
        print("rankLowestFirst: ", analysisMethod.rankLowestFirst)
        print("benchmark: ", analysisMethod.benchmark)
        print("category mode: ", analysisMethod.factorWeightMode)
        print("factor_weight: ", analysisMethod.userDefinedFactorWeights)
        print('EvalPeriod', analysisMethod.EvalPeriod)
        print('minEvalPeriod', analysisMethod.minEvalPeriod)
        print('stockWeightMode', analysisMethod.stockWeightMode)

        
        combinedIC, df_group_net,df_group_alpha, df_bt_indicator, df_bt_alpha_indicator = analysisMethod.run()
        save_csv(combinedIC, df_group_net, df_bt_indicator, df_bt_alpha_indicator)
        group_data = df_group_net.to_dict()
        group_data_alpha = df_group_alpha.to_dict()
        group_data["years"] = list(df_group_net.index)
        table_data_trans = []
        table_data_trans2 = []
        for index, row in df_bt_indicator.iterrows():
            item = {}
            item["group"] = row["group"]
            item["year_rate"] = row["年化收益率"]
            item["rate"] = row["夏普比率"]
            item["max"] = row["最大回撤"]
            table_data_trans.append(item)
        for index, row in df_bt_alpha_indicator.iterrows():
            item = {}
            item["group"] = row["group"]
            item["year_rate"] = row["年化超额收益率"]
            item["max_drawdown"] = row["超额最大回撤"]
            item["calmar"] = row['calmar']
            table_data_trans2.append(item)
        IC_analysis = {}
        IC_analysis["month"] = list(combinedIC['month'])
        IC_analysis["IC"] = list(combinedIC["IC"])
        IC_analysis["cumulative"] = list(combinedIC["cumulative"])
        res = {}
        res["group"] = group_data
        res['group_alpha'] = group_data_alpha
        res["indicator"] = table_data_trans
        res['indicator_alpha'] = table_data_trans2
        res["IC_val"] = IC_analysis
        result = res
        res_json = json.dumps(res)
        #analysisMethod.universe_index = scope
        

        return HttpResponse(res_json)
    


@csrf_exempt
def get_csv_output(request):

    
    return FileResponse(open('csv_result/packed_files.ZIP', 'rb'), as_attachment=True, filename="packed_files.zip")

def save_csv(combinedIC, df_group_net, df_bt_indicator, df_bt_alpha_indicator):
    IC_Dataframe = pd.DataFrame(combinedIC)
    if not os.path.exists('csv_result'):
        os.makedirs('csv_result')
    IC_Dataframe.to_csv("csv_result/IC_result.csv")
    df_group_net.to_csv('csv_result/grouped_month_ret.csv')
    df_bt_indicator.to_csv('csv_result/indicator.csv')
    df_bt_alpha_indicator.to_csv('csv_result/alpha_indicator.csv')
    with zipfile.ZipFile('csv_result/packed_files.ZIP', 'w') as zipped:
        zipped.write('csv_result/IC_result.csv')
        zipped.write('csv_result/grouped_month_ret.csv')
        zipped.write('csv_result/indicator.csv')
        zipped.write('csv_result/alpha_indicator.csv')