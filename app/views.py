from django.template import loader
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from .factor.FactorAnalysis2 import AnalysisMethod
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime
import pandas as pd


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.switch_backend('agg')
analysisMethod = AnalysisMethod()
scope = analysisMethod.universe_index
#name_list = analysisMethod.factor_name_lst

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

    print("STAET COLLECTING DATA")
    group_graph_name = ""
    file_name = ""
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
        print(analysisMethod.factor_name_lst)
        print(analysisMethod.start)
        print(analysisMethod.end)
        print(analysisMethod.trade_freq)
        print(analysisMethod.groupnum)
        print(analysisMethod.universe_index)
        analysisMethod.MainFunc()
        # print(analysisMethod.df_IC)
    group_data = analysisMethod.df_group_net.to_dict()
    group_data["years"] = list(analysisMethod.df_group_net.index)
    table_data_trans = []
    for index, row in analysisMethod.df_bt_indicator.iterrows():
        item = {}
        item["group"] = row["group"]
        item["year_rate"] = row["年化收益率"]
        item["rate"] = row["夏普比率"]
        item["max"] = row["最大回撤"]
        table_data_trans.append(item)
    IC_analysis = {}
    IC_analysis["year"] = list(analysisMethod.df_IC.index)
    IC_analysis["IC"] = list(analysisMethod.df_IC["IC"])
    IC_analysis["cumulative"] = list(analysisMethod.df_IC["IC_累计值"])
    res = {}
    res["group"] = group_data
    res["indicator"] = table_data_trans
    res["IC_val"] = IC_analysis
    res_json = json.dumps(res)
    analysisMethod.universe_index = scope

    return HttpResponse(res_json)


# def drawIC(x, y, y1):
#     dir_path = "static"
#     pattern = "result_*.png"
#     if len(glob.glob(os.path.join(dir_path, pattern))) > 0:
#         filepath = glob.glob(os.path.join(dir_path, pattern))[0]
#         os.remove(filepath)
#     y_pos = np.arange(len(x))
#     #print(y_pos)
#     plt.bar(y_pos, y, align='center', alpha=0.5)
#     # plt.xticks(rotation=90)
#     ax = plt.gca()
#     ax.set_xticklabels([])
#     ax.set_xticks(y_pos)
#     ax.spines["bottom"].set_position(("data", 0))
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_visible(False)
#     ax.set_ylim([min(y), max(y)])
#     label_offset = 0.5
#     for item, (x_position, y_position) in zip(x, enumerate(y1)):
#         label_y = -label_offset
#         ax.text(x_position, label_y, item, ha="center", va="top")
#     ax.text(0.5, -0.05, "IC Analysis", ha="center", va="top", transform=ax.transAxes)

#     ax2 = ax.twinx()
#     ax2.plot(y_pos, y1, color="red", marker="o")
#     ax2.set_xticklabels([])
#     ax2.set_xticks(y_pos)
#     ax2.spines["bottom"].set_visible(False)
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#     ax2.spines["left"].set_visible(False)
#     ax2.set_ylim([0, max(y1)])
#     ax2.set_yticks(np.arange(min(y1), max(y1), 0.1))
#     ax.set_ylabel("IC", loc="top")
#     # # ax.yaxis.set_label_coords(1.1, 0.5)
#     ax2.set_ylabel("IC_Cumulative", loc="top")
#     plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
#     ax.set_xticklabels(x, rotation=90)
#     ax2.set_xticklabels(x, rotation=90)
#     current_time = datetime.datetime.now()
#     time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
#     _name = f"result_{time_str}.png"
#     plt.savefig("static/" + _name)

#     print("finish")
#     plt.clf()
#     return _name

# def draw_group(groups):
#     dir_path = "static"
#     pattern = "result2_*.png"
#     if len(glob.glob(os.path.join(dir_path, pattern))) > 0:
#         filepath = glob.glob(os.path.join(dir_path, pattern))[0]
#         os.remove(filepath)

#     x_ticks = []
#     ax = plt.gca()
#     ind = list(groups.index)
#     for val in ind:
#         x_ticks.append(int(val))
#     y_pos = np.arange(len(x_ticks))
#     column_names = groups.columns.tolist()
#     for column in column_names:
#         print(groups[column])
#         if column == 'longshort_hedge':
#             plt.plot(y_pos, list(groups[column]), linestyle='--', label=column, linewidth=1, color='red')
#         else:
#             plt.plot(y_pos, list(groups[column]), label=column, linewidth=1)
        
#     ax.set_xticklabels([])
    
#     # ax.set_yticklabels([])
#     if len(y_pos > 20):
#         divide = int(len(y_pos) / 20) + 1
#         ax.set_xticks(y_pos[::divide])
#         ax.set_xticklabels(x_ticks[::divide], rotation=90)
#         # ax.set_yticks(y_pos[::5])
#         # ax.set_yticklabels(list(groups["longshort_hedge"])[::5])
#     else:
#         ax.set_xticks(y_pos)
#         ax.set_xticklabels(x_ticks, rotation=90)
#         # ax.set_yticklabels(list(groups["longshort_hedge"]))    
#     plt.xlabel(u"日期")
#     plt.ylabel(u"净值")
#     plt.title(u"因子十分组及多空对冲净值走势")
#     plt.legend()
#     #plt.xticks(x_ticks, x_ticks, rotation="vertical") 
#     current_time = datetime.datetime.now()
#     time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
#     _name = f"result2_{time_str}.png"
#     plt.savefig("static/" + _name)
#     plt.clf()
#     return _name