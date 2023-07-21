from django.template import loader
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from app.factor.factorModel import factorModel



analysisMethod = factorModel()
scope = analysisMethod.universe_index


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
        factor_list = []
        if len(factor) > 0:
            for val in factor:
                factor_list.append(factor_dict[val]["label"])
            analysisMethod.factor_name_lst = factor_list
        if len(data['rankLowestFirst']) > 0:
            analysisMethod.rankLowestFirst = data['rankLowestFirst']
        print(analysisMethod.factor_name_lst)
        print(analysisMethod.start)
        print(analysisMethod.end)
        print(analysisMethod.trade_freq)
        print(analysisMethod.groupnum)
        print(analysisMethod.universe_index)
        print(analysisMethod.rankLowestFirst)
        combinedIC, df_group_net,df_group_alpha, df_bt_indicator, df_bt_alpha_indicator = analysisMethod.run()
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
        res_json = json.dumps(res)
        #analysisMethod.universe_index = scope
        

        return HttpResponse(res_json)