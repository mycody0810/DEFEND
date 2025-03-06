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


sys_path = sys.path[0]
if not os.path.isdir(sys_path):
    sys_path = os.path.dirname(sys_path) 
os.chdir(sys_path)

def get_default_dict():
    default_dict = \
        {
            # "srcip": None, 
            # "sport": None,
            # "dstip": None,
            # "dsport": None,
            "proto": None,
            "state": None,
            "dur": None,
            "sbytes": None,
            "dbytes": None,
            "sttl": None,
            "dttl": None,
            "sloss": None,
            "dloss": None,
            "service": None,
            "Sload": None,
            "Dload": None,
            "Spkts": None,
            "Dpkts": None,
            "swin": None,
            "dwin": None,
            "stcpb": None,
            "dtcpb": None,
            "smeansz": None,
            "dmeansz": None,
            "trans_depth": None,
            "res_bdy_len": None,
            "Sjit": None,
            "Djit": None,
            # "Stime": None,
            # "Ltime": None,
            "Sintpkt": None,
            "Dintpkt": None,
            "tcprtt": None,
            "synack": None,
            "ackdat": None,
            "is_sm_ips_ports": None,
            "ct_state_ttl": None,
            "ct_flw_http_mthd": None,
            "is_ftp_login": None,
            "ct_ftp_cmd": None,
            "ct_srv_src": None,
            "ct_srv_dst": None,
            "ct_dst_ltm": None,
            "ct_src_ltm": None,
            "ct_src_dport_ltm": None,
            "ct_dst_sport_ltm": None,
            "ct_dst_src_ltm": None,
            "attack_cat": None,
            "Label": None,
        }
    return default_dict
    
def get_default_str_list():
    str_list = \
        [
            "srcip",
            "dstip",
            "proto",
            "state",
            "service",
            "attack_cat",
        ]
    return str_list

def get_default_float_list():
    float_list = \
        [
            "dur",
            "Sload",
            "Dload",
            "Sintpkt",
            "Dintpkt",
        ]
    return float_list
                


# {'proto': 1, 'state': 1, 'dur': 1, 'sbytes': 1, 'dbytes': 1, 'sttl': 1, 'dttl': 1, 'sloss': 1, 'dloss': 1, 'service': 1, 'Sload': 1, 'Dload': 1, 'Spkts': 1, 'Dpkts': 1, 'swin': 1, 'dwin': 1, 'stcpb': 1, 'dtcpb': 1, 'smeansz': 1, 'dmeansz': 1, 'trans_depth': 1, 'res_bdy_len': 1, 'Sjit': 1, 'Djit': 1, 'tcprtt': 1, 'synack': 1, 'ackdat': 1, 'is_sm_ips_ports': 1, 'ct_state_ttl': 1, 'ct_flw_http_mthd': 1, 'is_ftp_login': 1, 'ct_ftp_cmd': 1, 'ct_srv_src': 1, 'ct_srv_dst': 1, 'ct_dst_ltm': 1, 'ct_src_ltm': 1, 'ct_src_dport_ltm': 1, 'ct_dst_sport_ltm': 1, 'ct_dst_src_ltm': 1, 'attack_cat': 1, 'Label': 1}
def main_do(csv_path, raw_db, out_db):
    
    
    # client = MongoClient('mongodb://127.0.0.1:27017/')
    # # out_db = client["paper0"]["matching_UNSW_NB15_training_test"]
    # out_db = client["paper0"]["matching_UNSW_NB15_testing"]
    # raw_db = client["paper2"]["data1008"]

    
    # csv_path = "C:\\work_2014\\论文\\data\\UNSW-NB15\\csv_features\\UNSW_NB15_testing-set.csv"
    # csv_path = "C:\\work_2014\\论文\\data\\UNSW-NB15\\csv_features\\UNSW_NB15_training-set.csv"
    
    df_1 = pandas.read_csv(csv_path)
    
    tqdm_t = tqdm(total=len(df_1))
    
    for index, row in df_1.iterrows():
        tqdm_t.update(1)
        # if index % 1000 == 0:
        #     print(index)
        default_dict = get_default_dict()
        default_str_list = get_default_str_list()
        default_float_list = get_default_float_list()
        for key in default_dict:
            
            key_t = key.lower()
            if key_t == "smeansz":
                key_t = "smean"
                
            if key_t == "dmeansz":
                key_t = "dmean"
                
            if key_t == "res_bdy_len":
                key_t = "response_body_len"
                
            if key_t == "sintpkt":
                key_t = "sinpkt"
            
            if key_t == "dintpkt":
                key_t = "dinpkt"
            
            
            if key in default_str_list:
                default_dict[key] = {"$in": [str(row[key_t]).upper().replace(" ", ""),
                                            str(row[key_t]).lower().replace(" ", "")]}
            else:
                default_dict[key] = row[key_t]
                
                
            if key in default_float_list:
                default_dict[key] = float(default_dict[key])
                
            for key_t in default_dict.keys():
                Value_t = default_dict[key_t]
                if isinstance(Value_t, int):
                    if Value_t == 0:
                        default_dict[key_t] = {"$in": [0, -1]}
        
                
        
        
        if str(default_dict["attack_cat"]).count("NORMAL") > 0:
            default_dict["attack_cat"] = ""
            
        if str(default_dict["attack_cat"]).count("BACKDOOR") > 0:
            default_dict["attack_cat"] = {"$in": ["BACKDOOR", "BACKDOORS"]}
            
        # print(default_dict)
        # print("%f" % default_dict["dur"])
        
        raw_data_count = raw_db.count_documents(default_dict)
        
        if raw_data_count == 0:
            print("----------------")
            print("not found")
            print(default_dict)
            print("dur", "%f" % default_dict["dur"])
            print(row)
            print(index)

            # exit()
            # continue
        raw_data_list = raw_db.find(default_dict)
        
        row_dict = row.to_dict()
        
        id_str_list = []
        
        for raw_data in raw_data_list:
            id_str_list.append((raw_data["_id"]))
            
        row_dict["id_str_list"] = id_str_list
        row_dict["id_str_list_len"] = len(id_str_list)
        out_db.insert_one(row_dict)
        # print(raw_data_list)
    tqdm_t.close()
        
        
    
    return





def main():
    
    
    client = MongoClient('mongodb://127.0.0.1:27017/')
    # out_db = client["paper0"]["matching_UNSW_NB15_training_test"]
    
    raw_db = client["paper2"]["data1008"]


    csv_path = "C:\\work_2014\\论文\\data\\UNSW-NB15\\csv_features\\UNSW_NB15_testing-set.csv"
    out_db = client["paper0"]["matching_UNSW_NB15_testing"]
    main_do(csv_path, raw_db, out_db)
    
    csv_path = "C:\\work_2014\\论文\\data\\UNSW-NB15\\csv_features\\UNSW_NB15_training-set.csv"
    out_db = client["paper0"]["matching_UNSW_NB15_training"]
    main_do(csv_path, output_table_name="matching_UNSW_NB15_training")
    
    
    return





if __name__ == "__main__":
    
    main()