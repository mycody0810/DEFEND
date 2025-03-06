#!/usr/bin/env python
# coding: utf-8

import pandas as pd
def concat_raw_unsw_nb15():
    # Read the original UNSW-NB15 dataset with 2.5 million rows
    columns = pd.read_csv("../input/unsw-nb15/UNSW-NB15_features.csv", encoding="ISO-8859-1").loc[:, "Name"]
    df = pd.DataFrame(data=None)
    # for i in range(1, 5):
    for i in range(1, 5):
        tem_df = pd.read_csv("../input/unsw-nb15/UNSW-NB15_{}.csv".format(i), low_memory=False)
        tem_df.columns = columns
        df = pd.concat([df, tem_df], axis=0, ignore_index=True)
    cate_qualify(data=df)
    # To align with txxxing-set.csv, delete the following fields (but the rate field is missing)
    df.drop(columns=["srcip", "sport", "dstip", "dsport", "Stime", "Ltime"], inplace=True)
    # Handle anomalies in the ct_ftp_cmd feature
    df["ct_ftp_cmd"] = df["ct_ftp_cmd"].apply(lambda x: 0 if x == ' ' else int(x)).astype(int)
    # Handle missing values
    column_fillna(df, 'is_ftp_login', 0)
    column_fillna(df, 'ct_flw_http_mthd', 0)
    df.to_csv("../input/us_features/feature_2.csv", index=False)
    return df

def concat_merge_unsw_nb15_25w():
    training = pd.read_csv("../input/us_features/training_data_2c_feature2.csv")
    testing = pd.read_csv("../input/us_features/testing_data_2c_feature2.csv")
    df = pd.concat([training, testing], axis=0, ignore_index=True)
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
    drop_cols = df.columns[:44]
    df = df.drop(columns=drop_cols, axis=1)
    column_map = {}
    for col in df.columns:
        if col.endswith("_c1"):
            column_map[col] = col[:-3]
    df.rename(columns=column_map, inplace=True)
    cate_qualify(data=df)
    df["ct_ftp_cmd"] = df["ct_ftp_cmd"].apply(lambda x: 0 if x == ' ' else int(x)).astype(int)
    column_fillna(df, 'is_ftp_login', 0)
    column_fillna(df, 'ct_flw_http_mthd', 0)
    df.to_csv("../input/us_features/feature_7.csv", index=False)
    return df

def get_txxing_set_clss_analysis():
    from analysis.dataset_analysis import category_class_analysis
    data_test = pd.read_csv("../input/unsw-nb15/UNSW_NB15_testing-set.csv")
    data_train = pd.read_csv("../input/unsw-nb15/UNSW_NB15_training-set.csv")
    data = pd.concat([data_test, data_train], axis=0)
    cate, _ = category_class_analysis(data, data.shape[0])
    return cate
def cate_qualify(data):
    column_fillna(data, 'attack_cat', 'Normal')
    data["attack_cat"] = data["attack_cat"].apply(lambda x: x.strip())
    data["attack_cat"] = data["attack_cat"].apply(lambda x: x.title())
    data["attack_cat"] = data["attack_cat"].replace("Backdoors", "Backdoor")
    data["attack_cat"] = data["attack_cat"].replace("Dos", "DoS")

def column_fillna(data, col, value):
    col_data = data.loc[:, col]
    df_col = col_data.fillna(value)
    data.loc[:, col] = df_col

# concat_merge_unsw_nb15_25w()
