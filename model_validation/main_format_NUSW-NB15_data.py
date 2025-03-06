import file_io
from tqdm import tqdm
import data_cover
import sys
import pandas
import time
import numpy as np
import os
from pymongo import MongoClient


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


def main():
    print(sys.argv)
    # pcap_path = "C:\\work_2014\\论文\\z202408\\code2\\test_pcap"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    csv_path = "C:\\work_2014\\论文\\data\\UNSW-NB15\\csv_features"
    
    csv_file_name = "UNSW-NB15_"
    
    for i in range(0, 10):
        csv_file_path = os.path.join(csv_path, csv_file_name + str(i) + ".csv")
        if os.path.exists(csv_file_path) == False:
            continue
        print(csv_file_path)
        # df = pandas.read_csv(csv_file_path)
        csv_text = file_io.read_txt_file(csv_file_path)
        csv_text = csv_text.replace("\r", "")
        line_list = csv_text.split("\n")
        
        for index_t, line_t in enumerate(line_list):
            # print(line_t)
            if line_t == "":
                continue
            if line_t.count(",") != 48:
                print("error")
                print("line_t.count(\",\") != 48")
                print(line_t)
                print(csv_file_path)
                print(index_t + 1)
                exit()
            
            value_list = line_t.split(",")
            
            
            
            for v_index, value_t in enumerate(value_list):
                if value_t == "":
                    continue
                if value_t == None:
                    continue
                if bytes(value_t, encoding="utf8")[0] >= 128:
                    print("error")
                    print("bytes(line_t[0])[0] >= 128")
                    print(line_t)
                    print(csv_file_path)
                    print(index_t + 1)
                    exit()
                if value_t == " ":
                    value_list[v_index] = ""
                
                value_t = value_t.replace(" ", "")
                if value_t.startswith("0x"):
                    value_t = str(int(value_t, 16))
                value_list[v_index] = value_t
                
                if value_t == "":
                    continue
                if bytes(value_t, encoding="utf8")[0] >= 65 and bytes(value_t, encoding="utf8")[0] <= 122:
                    value_t = value_t.upper()
                    value_list[v_index] = value_t
                    
            if value_list[3] == "-":
                value_list[3] = ""
                
            value_text = ",".join(value_list)
            line_list[index_t] = value_text
        csv_text = "\n".join(line_list)
        csv_file_path_new = os.path.join(csv_path, "new_" + csv_file_name + str(i) + ".csv")
        file_io.write_txt_file(csv_file_path_new, csv_text)
       
    return







if __name__ == "__main__":
    
    main()