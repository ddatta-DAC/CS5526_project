import numpy as np
import yaml
import pandas as pd
import sklearn
from pprint import pprint
import glob
import os
import math
from sklearn import preprocessing
from scipy.stats import rv_discrete
import pickle
from sklearn.metrics import mutual_info_score
import itertools
import time

try:
    import ad_tree_v1
except:
    from . import ad_tree_v1

import operator

# ------------------------- #
# Based on
# Detecting patterns of anomalies
# https://dl.acm.org/citation.cfm?id=1714140
# ------------------------- #
_author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "1.0"
# ------------------------- #

CONFIG_FILE = 'config_1.yaml'

with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

SAVE_DIR = config['SAVE_DIR']
_DIR = config['_DIR']
OP_DIR = config['OP_DIR']
DATA_DIR = config['DATA_DIR']
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
SAVE_DIR = os.path.join(SAVE_DIR, _DIR)

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

if not os.path.exists(OP_DIR):
    os.mkdir(OP_DIR)
OP_DIR = os.path.join(OP_DIR, _DIR)

if not os.path.exists(OP_DIR):
    os.mkdir(OP_DIR)

DATA_FILE = os.path.join(DATA_DIR, _DIR, _DIR + '_x.pkl')
with open(DATA_FILE, 'rb') as fh:
    DATA_X = pickle.load(fh)
ID_LIST_FILE = os.path.join(DATA_DIR, _DIR, _DIR + '_idList.pkl')

with open(ID_LIST_FILE, 'rb') as fh:
    ID_LIST = pickle.load(fh)

print(DATA_X.shape)


# ----------------------------------- #
def calc_MI(x, y):
    custom = False
    if custom:
        x_vals = list(set(x))
        y_vals = list(set(y))
        k = len(x_vals) + len(y_vals)
        N = len(x)
        p_x = {_x: (x_vals.count(_x) / N) for _x in x_vals}
        p_y = {_y: (y_vals.count(_y) / N) for _y in y_vals}

        mi = 0
        xy = np.transpose(np.vstack([x, y]))
        df = pd.DataFrame(data=xy, columns=['x', 'y'])
        df = df.groupby(['x', 'y']).size().reset_index(name='counts')

        N = len(df)
        df['_mi'] = 0

        def set_val(row, N):
            _p_xy = row['counts'] / N
            _p_x = p_x[row['x']]
            _p_y = p_y[row['y']]
            r = _p_xy * math.log(_p_xy / (_p_x * _p_y), k)
            return r

        df['_mi'] = df.apply(set_val, axis=1, args=(N,))
        mi = np.sum(list(df['_mi']))
        # for i, row in df.iterrows():
        #     # calculate p_xy
        #     _p_xy = row['counts']/N
        #     _p_x = p_x[row['x']]
        #     _p_y = p_y[row['y']]
        #     mi += _p_xy * math.log(_p_xy / (_p_x * _p_y), k)

    # for now use this
    mi = mutual_info_score(x, y)
    return mi


# ----------------------------------- #


# Algorithm thresholds
MI_THRESHOLD = 0.1
ALPHA = 0.1


# Get arity of each domain
def get_domain_arity():
    f = os.path.join(DATA_DIR, _DIR, 'domain_dims.pkl')
    with open(f, 'rb') as fh:
        dd = pickle.load(fh)
    return list(dd.values())


# --------------- #
def get_MI_attrSetPair(data_x, s1, s2, obj_adtree):

    def _join(row, indices):
        r = '_'.join([str(row[i]) for i in indices])
        return r

    _idx = list(s1)
    _idx.extend(s2)
    _atr = list(s1)
    _atr.extend(s2)
    print(_atr)

    _tmp_df = pd.DataFrame(data=DATA_X).sample(frac=0.25)
    _tmp_df = _tmp_df[_atr]

    _tmp_df['x'] = None
    _tmp_df['y'] = None
    _tmp_df['x'] = _tmp_df.apply(
        _join,
        axis=1,
        args=(s1,)
    )
    _tmp_df['y'] = _tmp_df.apply(
        _join,
        axis=1,
        args=(s2,)
    )
    mi = calc_MI(_tmp_df['x'], _tmp_df['y'])
    return mi

    # MI = Sum ( P_(x)(y) log( P_(x)(y)/ P_(x)*P_(y) )


# get sets of attributes for computing r-value
# input attribute indices 0 ... m-1
# Returns sets of attributes of size k

def get_attribute_sets(
        attribute_list,
        obj_adtree,
        k=2
):
    global SAVE_DIR
    use_mi = True
    # check if file present in save dir
    op_file_name = 'set_pairs_' + str(k) + '.pkl'
    op_file_path = os.path.join(SAVE_DIR, op_file_name)

    if os.path.exists(op_file_path):
        with open(op_file_path, 'rb') as fh:
            set_pairs = pickle.load(fh)
        return set_pairs

    # -------------------------------------- #

    # We can attribute sets till size k
    # Add in size 1
    sets = list(itertools.combinations(attribute_list, 1))

    for _k in range(2, k + 1):
        _tmp = list(itertools.combinations(attribute_list, _k))
        sets.extend(_tmp)

    # check if 2 sets have MI > 0.1 and are mutually exclusive
    set_pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            s1 = sets[i]
            s2 = sets[j]
            print(s1, s2)
            # mutual exclusivity test
            m_e = (len(set(s1).intersection(s2)) == 0)

            if m_e is False:
                continue
            # -- Ignore MI for now -- #
            # MI
            if use_mi is False:
                mi_flag = True
            else:
                mi = get_MI_attrSetPair(DATA_X, s1, s2, obj_adtree)
                if mi > 0.1:
                    mi_flag = True

            if mi_flag is True:
                set_pairs.append((s1, s2))

    _dict = {e[0]: e[1] for e in enumerate(set_pairs, 0)}
    set_pairs = _dict
    # Save

    with open(op_file_path, 'wb') as fh:
        pickle.dump(set_pairs, fh, pickle.HIGHEST_PROTOCOL)

    return set_pairs


def get_count(obj_adtree, domains, vals):
    _dict = {k: v for k, v in zip(domains, vals)}
    res = obj_adtree.get_count(_dict)
    return res


def get_r_value(record, obj_adtree, set_pairs, N):
    _r_dict = {}
    for k, v in set_pairs.items():
        _vals = []
        _domains = []
        for _d in v[0]:
            _domains.append(_d)
            _vals.append(record[_d])

        P_at = get_count(obj_adtree, _domains, _vals)
        P_at = P_at / N
        # print(P_at)

        _vals_1 = []
        _domains_1 = []
        for _d in v[1]:
            _domains_1.append(_d)
            _vals_1.append(record[_d])

        P_bt = get_count(obj_adtree, _domains_1, _vals_1)
        P_bt = P_bt / N
        # print(P_bt)

        _vals.extend(_vals_1)
        _domains.extend(_domains_1)

        P_ab = get_count(obj_adtree, _domains, _vals) / N
        r = (P_ab) / (P_at * P_bt)
        _r_dict[k] = r
    # heuristic
    sorted_r = list(sorted(_r_dict.items(), key=operator.itemgetter(1)))
    # print(sorted_r)

    score = 1
    U = set()
    threshold = ALPHA

    for i in range(len(sorted_r)):
        _r = sorted_r[i][1]
        tmp = set_pairs[sorted_r[i][0]]
        _attr = [item for sublist in tmp for item in sublist]

        if _r > threshold:
            break
        if len(U.intersection(set(_attr))) == 0:
            U = U.union(set(_attr))
            score *= _r
    print(score)
    return score


def main():
    N = DATA_X.shape[0]
    obj_ADTree = ad_tree_v1.ADT()
    obj_ADTree.setup(DATA_X)

    attribute_list = list(range(DATA_X.shape[1]))
    print('Attribute list', attribute_list)

    attribute_set_pairs = get_attribute_sets(
        attribute_list,
        obj_ADTree,
        k=2
    )

    print(attribute_set_pairs)
    print(' Number of attribute set pairs ', len(attribute_set_pairs))

    id_list = ID_LIST['all']
    result_dict = {}
    for _id, record in zip(id_list, DATA_X):
        r = get_r_value(record, obj_ADTree, attribute_set_pairs, N)
        result_dict[id] = r

    # save file
    SAVE_FILE_OP = '_'.join(['result_alg_1_', _DIR, str(time.time().split('.'[0]))]) + '.pkl'
    SAVE_FILE_OP_PATH = os.path.join(DATA_DIR, _DIR, SAVE_FILE_OP)
    with open(SAVE_FILE_OP_PATH, 'wb') as fh:
        pickle.dump(result_dict, fh, pickle.HIGHEST_PROTOCOL)

# ------------------------------------------ #

main()

# ------------------------------------------ #
# def _join(row, indices):
#     r = '_'.join([str(row[i]) for i in indices])
#     return r
#
# _idx = list(s1)
# _idx.extend(s2)
#
# _tmp_df = pd.DataFrame(data=DATA_X)
# _tmp_df['x'] = None
# _tmp_df['y'] = None
#
# _tmp_df['x'] = _tmp_df.apply(
#     _join,
#     axis=1,
#     args=(s1,)
# )
# _tmp_df['y'] = _tmp_df.apply(
#     _join,
#     axis=1,
#     args=(s2,)
# )
# mi = calc_MI(_tmp_df['x'], _tmp_df['y'])
# if mi < 0.1:
#     mi_flag = False
# else:
#     mi_flag = True
