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


sys_path = sys.path[0]
if not os.path.isdir(sys_path):
    sys_path = os.path.dirname(sys_path) 
os.chdir(sys_path)


def calculate_entropy(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

"""
srcip	nominal
sport	integer
dstip	nominal
dsport	integer
proto	nominal
state	nominal
dur	Float
sbytes	Integer
dbytes	Integer
sttl	Integer
dttl	Integer
sloss	Integer
dloss	Integer
service	nominal
Sload	Float
Dload	Float
Spkts	integer
Dpkts	integer
swin	integer
dwin	integer
stcpb	integer
dtcpb	integer
smeansz	integer
dmeansz	integer
trans_depth	integer
res_bdy_len	integer
Sjit	Float
Djit	Float
Stime	Timestamp
Ltime	Timestamp
Sintpkt	Float
Dintpkt	Float
tcprtt	Float
synack	Float
ackdat	Float
is_sm_ips_ports	Binary
ct_state_ttl	Integer
ct_flw_http_mthd	Integer
is_ftp_login	Binary
ct_ftp_cmd	integer
ct_srv_src	integer
ct_srv_dst	integer
ct_dst_ltm	integer
ct_src_ltm	integer
ct_src_dport_ltm	integer
ct_dst_sport_ltm	integer
ct_dst_src_ltm	integer
attack_cat	nominal
Label	binary
"""


def main():
    print(sys.argv)
    # pcap_path = "C:\\work_2014\\论文\\z202408\\code2\\test_pcap"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        
    mongoDB_client = MongoClient("mongodb://127.0.0.1:27017/")
    paper_c = mongoDB_client["paper2"]["data1008"]
    has_done = mongoDB_client["paper2"]["data1008_hasdone"]
    
    csv_path = "C:\\work_2014\\论文\\data\\UNSW-NB15\\csv_features"
    
    csv_file_name = "new_UNSW-NB15_"
    
    col_name_list = col_name.get_col_name_1008()
    # col_name_list = col_name_list[0:48]
    
    # info_list = []
    
    for i in range(0, 10):
        csv_file_path = os.path.join(csv_path, csv_file_name + str(i) + ".csv")
        if os.path.exists(csv_file_path) == False:
            continue
        print(csv_file_path)
        # df = pandas.read_csv(csv_file_path)
        csv_text = file_io.read_txt_file(csv_file_path)
        csv_text = csv_text.replace("\r", "")
        line_list = csv_text.split("\n")
        tqdm_t = tqdm(total=line_list.__len__(), desc='Reading csv files', ncols=100)
        
        for line_t in line_list:
            tqdm_t.update(1)
            line_t = line_t.split(",")
            if len(line_t) != 49:
                continue
            dict_info = {}
            dict_info[col_name_list[0]] = str(line_t[0])
            if line_t[1] == "-" or line_t[1] == "":
                line_t[1] = -1
            dict_info[col_name_list[1]] = int(line_t[1])
            dict_info[col_name_list[2]] = str(line_t[2])
            if line_t[3] == "":
                line_t[3] = -1
            dict_info[col_name_list[3]] = int(line_t[3])
            dict_info[col_name_list[4]] = str(line_t[4])
            dict_info[col_name_list[5]] = str(line_t[5])
            dict_info[col_name_list[6]] = float(line_t[6])
            dict_info[col_name_list[7]] = int(line_t[7])
            dict_info[col_name_list[8]] = int(line_t[8])
            dict_info[col_name_list[9]] = int(line_t[9])
            dict_info[col_name_list[10]] = int(line_t[10])
            dict_info[col_name_list[11]] = int(line_t[11])
            dict_info[col_name_list[12]] = int(line_t[12])
            dict_info[col_name_list[13]] = str(line_t[13])
            dict_info[col_name_list[14]] = float(line_t[14])
            dict_info[col_name_list[15]] = float(line_t[15])
            dict_info[col_name_list[16]] = int(line_t[16])
            dict_info[col_name_list[17]] = int(line_t[17])
            dict_info[col_name_list[18]] = int(line_t[18])
            dict_info[col_name_list[19]] = int(line_t[19])
            dict_info[col_name_list[20]] = int(line_t[20])
            dict_info[col_name_list[21]] = int(line_t[21])
            dict_info[col_name_list[22]] = int(line_t[22])
            dict_info[col_name_list[23]] = int(line_t[23])
            dict_info[col_name_list[24]] = int(line_t[24])
            dict_info[col_name_list[25]] = int(line_t[25])
            dict_info[col_name_list[26]] = float(line_t[26])
            dict_info[col_name_list[27]] = float(line_t[27])
            dict_info[col_name_list[28]] = float(line_t[28])
            dict_info[col_name_list[29]] = float(line_t[29])
            dict_info[col_name_list[30]] = float(line_t[30])
            dict_info[col_name_list[31]] = float(line_t[31])
            dict_info[col_name_list[32]] = float(line_t[32])
            dict_info[col_name_list[33]] = float(line_t[33])
            dict_info[col_name_list[34]] = float(line_t[34])
            dict_info[col_name_list[35]] = int(line_t[35])
            dict_info[col_name_list[36]] = int(line_t[36])
            if line_t[37] == "":
                line_t[37] = -1
            dict_info[col_name_list[37]] = int(line_t[37])
            if line_t[38] == "":
                line_t[38] = -1
            dict_info[col_name_list[38]] = int(line_t[38])
            if line_t[39] == "":
                line_t[39] = -1
            dict_info[col_name_list[39]] = int(line_t[39])
            dict_info[col_name_list[40]] = int(line_t[40])
            dict_info[col_name_list[41]] = int(line_t[41])
            dict_info[col_name_list[42]] = int(line_t[42])
            dict_info[col_name_list[43]] = int(line_t[43])
            dict_info[col_name_list[44]] = int(line_t[44])
            dict_info[col_name_list[45]] = int(line_t[45])
            dict_info[col_name_list[46]] = int(line_t[46])
            dict_info[col_name_list[47]] = str(line_t[47])
            dict_info[col_name_list[48]] = int(line_t[48])
            
            paper_c.insert_one(dict_info)
            
            
        
            # info_list = info_list + [dict_info]
        # print(dict_info)
        tqdm_t.close()
    # file_io.write_obj_file(info_list, "info_list.obj")

       
    return







if __name__ == "__main__":
    
    main()