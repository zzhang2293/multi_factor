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
    results that will show on web page
    these global variables are used to convert to csv file and send to front
    data structure: 
    combinedIC(dict): ['month': [list of month], 'IC': [list of ic value], 'cumulative': [list of cumulative ic]]

    df_group_net(DataFrame): [index = trade_date, column = ['group_0', 'group_1', 'group2', 'group3', 'longshort_hedge']]

    df_group_alpha(DataFrame): same as df_group_net, 

    df_bt_indicator(DataFrame): [index = [1,2,3,4,5 ...], column = ["group", "年化收益率","夏普比率","最大回撤"]]

    df_bt_alpha)indicator(DataFrame) : [index = [1,2,3,4,5 ....], column = ['group', "年化超额收益率","超额最大回撤","calmar"]]

'''
'''
    这个文件是 view 层， 用来接收 request 并交给后端处理，和发送处理好的文件到html页面。所有接收和发送都是这个文件来负责
'''


'''
    发送静态页面analysis_frontend.html 到浏览器
'''
def members(request):
    template = loader.get_template('analysis_frontend.html')
    return HttpResponse(template.render())


'''
    异步响应，将所有因子显示到穿梭框里，这个是刚加载完页面后，页面自动发送请求到这个函数里
'''
@csrf_exempt
def get_factors(request):
    allFactor = analysisMethod.allfactorname_lst
    return JsonResponse(allFactor, safe=False)


'''
    获取所有数据，前端提交一个form 表单
'''

@csrf_exempt
def collect_data(requests):

    print("START COLLECTING DATA")
    if requests.method == "POST":
        data = json.loads(requests.body)
        if (data["group"] != ""):
            analysisMethod.groupnum = int(data["group"])  # 多少组数据
        if data["freq"] != "":
            analysisMethod.trade_freq = data["freq"]  #频率
        if len(data["scope"]) > 0:         # 股票池范围
            analysisMethod.universe_index = data["scope"]
      
        if data["startDate"] != "":     #开始日期
            startDateModified = data["startDate"].split("T")[0]
            startDateModified = startDateModified.replace("-", "")
            analysisMethod.start = startDateModified
        if data["endDate"] != "":    #结束日期
            endDateModified = data["endDate"].split("T")[0]
            endDateModified = endDateModified.replace("-", "")
            analysisMethod.end = endDateModified 
        if data['benchmark'] != '':    #benchmark
            analysisMethod.benchmark = data['benchmark']
        if data['factor_weight_mode'] != '':    #计算权重的方式，自定义 智能和等权
            analysisMethod.factorWeightMode = data['factor_weight_mode']
    
        factor = data["factor"]
        factor_dict = data["factor_val"]    # factor：索引列表，告诉factor_dict都选了哪些因子
        factor_list = []
        if len(factor) > 0:
            for val in factor:    
                factor_list.append(factor_dict[val]["label"])
            analysisMethod.factor_name_lst = factor_list

        if len(data['rankLowestFirst']) > 0:  #获取完分组收益率后如何排序
            analysisMethod.rankLowestFirst = data['rankLowestFirst']
        if len(data['factor_weight']) > 0:  #自定义因子权重
            analysisMethod.userDefinedFactorWeights = [float(i) for i in data['factor_weight'].split(" ")]
        if data['EvalPeriod'] != '':    #因子权重智能模式下向前回看多久（最大）
            analysisMethod.EvalPeriod = int(data['EvalPeriod'])
        if data['minEvalPeriod'] != '':   #因子权重智能模式下向前回看多久（最小）
            analysisMethod.minEvalPeriod = int(data['minEvalPeriod'])
        if data['stockWeightMode'] != '':
            analysisMethod.stockWeightMode = data['stockWeightMode']  #设置分组后股票权重
        if data['factorSelectMode'] != '':
            analysisMethod.factorSelectMode = data['factorSelectMode']  #选因子模式（智能或者手动选）
        if data['factorChoosePeriod'] != '':
            analysisMethod.factorChoosePeriod = data['factorChoosePeriod'] #智能选择因子时回看周期
        if data['nFactors'] != '':
            analysisMethod.nFactors = data['nFactors']  #选多少因子
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
        print('factorSelectMode: ', analysisMethod.factorSelectMode)
        print('nFactors: ', analysisMethod.nFactors)
        print('factorChoosePeriod: ', analysisMethod.factorChoosePeriod)

        #开始运行
        combinedIC, df_group_net,df_group_alpha, df_bt_indicator, df_bt_alpha_indicator = analysisMethod.run()
        #保存到csv，用于后面如果有下载文件需求直接从后台拿文件
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
        #打包json传到前端
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