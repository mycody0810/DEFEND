import file_io
from tqdm import tqdm
import data_cover
import sys
import pandas
import time
import numpy as np
import os
from pymongo import MongoClient
import col_name 
# Collection
from pymongo.collection import Collection
from pymongo.database import Database
from scipy.stats import norm
from pymongo import cursor
from scipy.stats import entropy
from collections import Counter
import pickle
import copy
import random


sys_path = sys.path[0]
if not os.path.isdir(sys_path):
    sys_path = os.path.dirname(sys_path) 
os.chdir(sys_path)


def get_block_data_mongo(db_name, table_name, time_list):
    client = MongoClient('mongodb://127.0.0.1:27017/')
    db = client[db_name]    
    raw_data = db[table_name]
    
    for time_t in time_list:
        pipeline_dict_list = [
            {
                "$match": 
                    {
                        "timestamp": {
                            "$gte": time_t[0],
                            "$lt": time_t[1]
                        }
                    }
            },
            {
                "$out": {
                    "db": db_name + "_block",
                    "coll": table_name + "_" + str(time_t[0]) + "_" + str(time_t[1])
                }
            }
            
        ]
        
        print(pipeline_dict_list)
        
        raw_data.aggregate(pipeline_dict_list)
        # exit()
    
    return 



def vaule_to_hash(value):
    
    if value is None:
        value = 0
    
    if isinstance(value, float):
        if value == -1 or value == 0:
            sum_hash = 0
        else:
            sum_hash = value
    elif isinstance(value, int):
        if value == -1 or value == 0:
            sum_hash = 0
        else:
            sum_hash = value
    elif isinstance(value, str):
        if len(value) == 0:
            sum_hash = 0
        else:
            byte_array = bytearray(data_cover.str_to_bytes(str(value).upper()))
            sum_hash = np.array(list(byte_array)).mean()
    else:
        byte_array = bytearray(data_cover.str_to_bytes(str(value).upper()))
        sum_hash = np.array(list(byte_array)).mean()
    return sum_hash




def get_give_the_optimal_class_1_feature(merge_1_category_testing_training_feature_db,
                                         out_db):
    # client = MongoClient('mongodb://127.0.0.1:27017/')
    # # paper5> db.merge_1_category_testing_features
    # merge_1_category_testing_training_feature_db = \
    #     client["paper5"]["merge_1_category_testing_features"]
        
    # # merge_1_category_training_features_db = \
    # #     client["paper5"]["merge_1_category_training_features"]
    # # use paper5
    # # db.give_the_optimal_class_1_features.drop()
    # out_db = client["paper5"]["give_the_optimal_class_1_features_testing"]
    # out_db = client["paper5"]["give_the_optimal_class_1_features_training"]
    
        
    
    
    # testing_id_max = 82332
    training_id_max = 175341
    
    # c1_col_name_list = col_name.get_merge_1c_col_name()
    
    tqdm_t = tqdm(range(1, training_id_max+1))
    
    map_list = col_name.get_merge_1c_col_name_map()
    
    for id_t in tqdm_t:
        query_dict = {
            "id": id_t
        }
        testing_data = merge_1_category_testing_training_feature_db.find(query_dict, {"_id": 0})
        testing_data_cnt = merge_1_category_testing_training_feature_db.count_documents(query_dict)
        if testing_data_cnt == 1:
            out_db.insert_many(list(testing_data))
            continue
        # print(id_t, testing_data_cnt)
        
        
        min_num = -1
        min_data = None
        
        testing_data_list = []
        for data_line in testing_data:
            testing_data_list = testing_data_list + [data_line]
        # 随机乱序
        random.shuffle(testing_data_list)
        
        # 在众多特征中，取与目标特征距离距离最小的的那一条特征。
        # 当testing_data_list乱序时，如果所有当前特征与目标特征的都无差异，那么则相当于随机选了一个特征。
        
        for data_line in testing_data_list:
            # print(data_line)
            
            
            if data_line["attack_cat"] == "Normal":
                data_line["attack_cat"] = ""
            if data_line["attack_cat"].upper().count("BACKDOOR"):
                data_line["attack_cat"] = "Backdoor"
                
            if data_line["attack_cat_c1"] == "Normal":
                data_line["attack_cat_c1"] = ""
            if data_line["attack_cat_c1"].upper().count("BACKDOOR"):
                data_line["attack_cat_c1"] = "Backdoor"
                
            sum_t = 0
            for col_name_t in map_list:
                # print(col_name_t)
                
                col_name_t_1 = col_name_t[0]
                col_name_t_2 = col_name_t[1]
                
                val_1 = data_line[col_name_t_1]
                val_2 = data_line[col_name_t_2]
                
                
                
                val_1_t =vaule_to_hash(val_1)
                val_2_t = vaule_to_hash(val_2)
                
                if val_1_t == 0:
                    continue
                
                d_val_t = abs(val_1_t - val_2_t)
                # max_t = max(val_1_t, val_2_t)
                d_val_t2 = (d_val_t / val_1_t)
                # print(d_val_t2)
                if d_val_t2 > 1:
                    d_val_t2 = 1
                # print(d_val_t2)
                sum_t += d_val_t2
                
                        
            d_val = sum_t
                
            if min_data is None:
                min_data = data_line
                min_num = d_val
            else:
                if d_val < min_num:
                    min_data = data_line
                    min_num = d_val
            
            # print(min_num)
            if min_num == 0:
                
                break
        
        if min_data == None:
            out_db.insert_many(list(testing_data))
            continue
        out_db.insert_one(min_data)
        
        
        
    tqdm_t.close()
        



def main():
    
    client = MongoClient('mongodb://127.0.0.1:27017/')
    # paper5> db.merge_1_category_testing_features

        
    # merge_1_category_training_features_db = \
    #     client["paper5"]["merge_1_category_training_features"]
    # use paper5
    # db.give_the_optimal_class_1_features.drop()
    
    merge_1_category_testing_training_feature_db = \
        client["paper5"]["merge_1_category_testing_features"]
    out_db = client["paper5"]["give_the_optimal_class_1_features_testing"]
    get_give_the_optimal_class_1_feature(merge_1_category_testing_training_feature_db,
                                         out_db)
    
    merge_1_category_testing_training_feature_db = \
        client["paper5"]["merge_1_category_training_features"]
    out_db = client["paper5"]["give_the_optimal_class_1_features_training"]
    get_give_the_optimal_class_1_feature(merge_1_category_testing_training_feature_db,
                                         out_db)
    
    return










if __name__ == "__main__":
    
    main()