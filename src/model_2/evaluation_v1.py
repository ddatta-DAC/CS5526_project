import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
import matplotlib.pyplot as plt


def precision_recall_curve(
        sorted_id_score_dict,
        anomaly_id_list
):
    recall = 0
    correct = 0
    recall_vals = []
    precision_vals = []

    count = len(sorted_id_score_dict)
    cur_count = 0
    # Assumption is that the lowest likelihood events are anomalous
    for id, score in sorted_id_score_dict.items():
        flag = False
        if id in anomaly_id_list:
            flag = True
        cur_count += 1

        if flag is True:
            recall+=1
            correct += 1
        p = correct/cur_count
        r = recall/cur_count
        precision_vals.append(p)
        recall_vals.append(r)


    # x Axis : Recall
    # Y Axis : Precision

    return recall_vals, precision_vals

def performance_by_score(
        sorted_id_score_dict,
        anomaly_id_list
    ):

    N = len(sorted_id_score_dict)
    # increase by 0.5 % from 0.5 to 5
    _dict = {}
    for n in np.arange(0.5,5.5,0.5):
        c = int (N * n)
        _list_1 = list(sorted_id_score_dict.keys())[:c]
        res = len(set(_list_1).intersection(set(anomaly_id_list)))
        res = len(res)/c
        _dict[n] = res

    x = list(_dict.keys())
    y = list(_dict.values())
    return x,y
