from app.factor.factorModel import factorModel
from collections import defaultdict
import pandas as pd
import seaborn as sns

lst = ['Analyst_factor', 'NegMktValue', 'technology_factor', 'tps_sps', 'momentumn_factor', 'avgwght_momentum', 'seven_f', 'udslDWL', 'udslUCL', 'udsl', 'aShareholderZ', 'taEntropy', 'corrVP', 'apbSkew', 'sude', 'sudrev', 'lpnpQ', 'npQYOY', 'npYTDYOY', 'npTTMQOQ', 'npTTMYOY', 'revQYOY', 'revYTDYOY', 'revTTMQOQ', 'revTTMYOY', 'rrocQ', 'ocfa', 'roeQ', 'roeTTM', 'roeQYOYD', 'roeTTMQOQD', 'roeTTMYOYD', 'dtop', 'divPaidRatio', 'etopQ', 'stopQ', 'detopQ', 'dstopQD', 'aiSude', 'conSude', 'aiSudrev', 'conSudrev', 'aiNpYOY', 'aiRevYOY', 'conNpYOY', 'conRevYOY', 'aiEtop', 'aiEtopZ90', 'aiEtopZ180', 'conDaPE20', 'conDaPE40', 'conDaPE60', 'conDaPS20', 'conDaPS40', 'conDaPS60', 'astDa12Etop', 'astRankUppct', 'astProfitUppct', 'gsa', 'hkHoldRatioAll', 'hkHoldRatioB', 'hkHoldRatioC', 'fundT10Count', 'fundT10WeightMean', 'fundT10WeightMax', 'fundT10NegValuePct', 'naiveWeightChgAsym', 'fundT10ChgWeight', 'fundT10ChgValueRatio', 'pReportDate', 'pReportDiff', 'hkHoldVolChgB20', 'hkHoldVolChgC20', 'hkHoldVolChgAll20', 'hkHoldVolChgB60', 'hkHoldVolChgC60', 'hkHoldVolChgAll60', 'hkHoldVolChgB120', 'hkHoldVolChgC120', 'hkHoldVolChgAll120', 'aiDaNp30', 'aiDaNp60', 'aiDaNp90', 'aiDaRev30', 'aiDaRev60', 'aiDaRev90', 'aiDaPE30', 'aiDaPE60', 'aiDaPE90', 'aiDaPS30', 'aiDaPS60', 'aiDaPS90', 'astRptSentiW', 'astRptSentiZ180', 'astRptSentiZ365', 'astRptSentiZ730', 'sumIPC1Y', 'sumRelatedCorp1Y', 'sumExclPatent1Y', 'sumReviewDays1Y', 'maxRelatedCorp1Y', 'bcvp05M20D', 'ocvp05M20D', 'corrVPL05M20D', 'upp01M20D', 'ddp01M20D', 'voll01M20D', 'daizhuerjiu', 'FlowerHidInForest', 'rideinboatonwater', 'caomujiebing', 'decay_panic', 'volatility_enhance_panic', 'primitive_panic', 'flyintofire', 'modify_amplitude', 'month_jump', 'UTR', 'new_RPV', 'ubl', 'EP_d', 'EPDS', 'TotalMktValue', 'fuzziness_corr', 'fuzziness_amount_r', 'fin_adj_fpdiff', 'cloudOpenFogDisppear', 'ff3R220', 'ff3SpMom20', 'ff3SpVol20', 'ff3SysMom20', 'ff3SysVol20', 'rmVol20', 'rmVol60', 'rmVol120', 'trVol20', 'trVol60', 'trVol120', 'trVoV', 'mintvalQua20D', 'mintvalSkew20D', 'mintvalMts20D', 'mintvalMte20D', 'sectvalKurt20D', 'ovalMbsr20D', 'gmmMean1m20D', 'gmmDmean1m20D']

def find_factor_corr_heatmap(name_list, graph = False):
    model = factorModel()
    model.factor_name_lst = name_list
    _, _, Monthly_Factor_Score, _, _, _ = model.getData()
    factorNames = Monthly_Factor_Score.keys()
    factorCount = len(factorNames)

    res = defaultdict(list)

    for i in Monthly_Factor_Score.keys():
        for j in Monthly_Factor_Score[i].keys():
            res[i] += Monthly_Factor_Score[i][j]

    check = defaultdict(int)
    for i in res.keys():
        check[len(res[i])] += 1
    if len(check) != 1:
        print('不同因子数量不同，请检查数据')
        print(check)
        return
    
    df = pd.DataFrame(res)
    data = df.corr()
    return data