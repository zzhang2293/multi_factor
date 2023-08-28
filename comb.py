import numpy as np
import pandas as pd

def worker(factorComb, res, shared_data, lock):
    Monthly_Factor_Score = shared_data['Monthly_Factor_Score']
    Monthly_Equity_Returns = shared_data['Monthly_Equity_Returns']

    currIC = 0
    for month in list(Monthly_Equity_Returns.keys()):
        scores = np.array([])
        returns = Monthly_Equity_Returns[month]
        for factor in factorComb:
            if scores.size == 0:
                scores = np.array(Monthly_Factor_Score[factor][month])
            else:
                scores = np.add(scores, Monthly_Factor_Score[factor][month])
        corr = np.corrcoef(scores, returns)[0,1]
        currIC += corr
    lock.acquire()
    try:
        res.append(list(factorComb) + [currIC])
        print(list(factorComb) + [currIC])
    finally:
        lock.release()

if __name__ == "__main__":

    chooseAmount = 7

    import multiprocessing as mp
    from app.factor.factorModel import factorModel
    import itertools

    m = factorModel()
    m.factor_name_lst = m.allfactorname_lst
    print("getting data")
    _, Monthly_Equity_Returns, Monthly_Factor_Score, _, _, _ = m.getData()
    print(f'Got data')

    factors = m.factor_name_lst
    months = list(Monthly_Equity_Returns.keys())

    manager = mp.Manager()
    shared_data = manager.dict()
    shared_data['Monthly_Equity_Returns'] = Monthly_Equity_Returns
    shared_data['Monthly_Factor_Score'] = Monthly_Factor_Score
    res = manager.list()

    pool = mp.Pool(processes=mp.cpu_count())
    lock = manager.Lock()

    for factorComb in itertools.combinations(factors, chooseAmount):
        pool.apply_async(worker, args=(factorComb, res, shared_data, lock))

    pool.close()
    pool.join()

    result_df = pd.DataFrame(list(res), columns=['F1','F2','F3','F4','F5','F6','F7','IC'])
    result_df.to_csv(f'factorChoose{chooseAmount}.csv')
