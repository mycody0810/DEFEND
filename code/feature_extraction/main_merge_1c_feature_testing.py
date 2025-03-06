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
from datetime import datetime, timedelta
import time
import traceback

sys_path = sys.path[0]
if not os.path.isdir(sys_path):
    sys_path = os.path.dirname(sys_path) 
os.chdir(sys_path)

"""
paper5> db.output_2_category_features.findOne()
{
  _id: ObjectId('670f45be7c8223d642e66465'),
  srcip: '59.166.0.0',
  sport: 1390,
  dstip: '149.171.126.6',
  dsport: 53,
  proto: 'UDP',
  state: 'CON',
  dur: 0.001055,
  sbytes: 132,
  dbytes: 164,
  sttl: 31,
  dttl: 29,
  sloss: 0,
  dloss: 0,
  service: 'DNS',
  Sload: 500473.9375,
  Dload: 621800.9375,
  Spkts: 2,
  Dpkts: 2,
  swin: 0,
  dwin: 0,
  stcpb: 0,
  dtcpb: 0,
  smeansz: 66,
  dmeansz: 82,
  trans_depth: 0,
  res_bdy_len: 0,
  Sjit: 0,
  Djit: 0,
  Stime: 1421927414,
  Ltime: 1421927414,
  Sintpkt: 0.017,
  Dintpkt: 0.013,
  tcprtt: 0,
  synack: 0,
  ackdat: 0,
  is_sm_ips_ports: 0,
  ct_state_ttl: 0,
  ct_flw_http_mthd: 0,
  is_ftp_login: 0,
  ct_ftp_cmd: 0,
  ct_srv_src: 3,
  ct_srv_dst: 7,
  ct_dst_ltm: 1,
  ct_src_ltm: 3,
  ct_src_dport_ltm: 1,
  ct_dst_sport_ltm: 1,
  ct_dst_src_ltm: 1,
  attack_cat: '',
  Label: 0,
  _id_c1_1: ObjectId('670f2d28a9648c9b50dc7606'),
  mediacy_pkt_info_set_id_c1_1: ObjectId('670f199ccf069a029e6cb219'),
  NUSW_group_id_str_c1_1: ObjectId('670dd99acf069a029eec7acf'),
  NUSW_id_list_c1_1: [ ObjectId('6704dda013c0f134fde1140d') ],
  srcip_c1_1: '59.166.0.0',
  dstip_c1_1: '149.171.126.6',
  sport_c1_1: 1390,
  dsport_c1_1: 53,
  Ethernet_proto_c1_1: 2048,
  ip_proto_c1_1: 17,
  count_c1_1: 2,
  start_time_c1_1: 1421927414.321562,
  end_time_c1_1: 1421927414.321579,
  tcp_flags_c1_1: 0,
  layer_1_payload_size_mean_c1_1: 66,
  layer_1_payload_size_std_c1_1: 0,
  layer_1_payload_size_max_c1_1: 66,
  layer_1_payload_size_min_c1_1: 66,
  layer_1_payload_size_entropy_c1_1: 0,
  layer_2_payload_size_mean_c1_1: 46,
  layer_2_payload_size_std_c1_1: 0,
  layer_2_payload_size_max_c1_1: 46,
  layer_2_payload_size_min_c1_1: 46,
  layer_2_payload_size_entropy_c1_1: 0,
  layer_3_payload_size_mean_c1_1: 38,
  layer_3_payload_size_std_c1_1: 0,
  layer_3_payload_size_max_c1_1: 38,
  layer_3_payload_size_min_c1_1: 38,
  layer_3_payload_size_entropy_c1_1: 0,
  arp_op_mean_c1_1: null,
  arp_op_std_c1_1: null,
  arp_op_max_c1_1: null,
  arp_op_min_c1_1: null,
  arp_op_entropy_c1_1: null,
  icmp_code_mean_c1_1: null,
  icmp_code_std_c1_1: null,
  icmp_code_max_c1_1: null,
  icmp_code_min_c1_1: null,
  icmp_code_entropy_c1_1: null,
  icmp_type_mean_c1_1: null,
  icmp_type_std_c1_1: null,
  icmp_type_max_c1_1: null,
  icmp_type_min_c1_1: null,
  icmp_type_entropy_c1_1: null,
  igmp_type_mean_c1_1: null,
  igmp_type_std_c1_1: null,
  igmp_type_max_c1_1: null,
  igmp_type_min_c1_1: null,
  igmp_type_entropy_c1_1: null,
  ip_off_mean_c1_1: 16384,
  ip_off_std_c1_1: 0,
  ip_off_max_c1_1: 16384,
  ip_off_min_c1_1: 16384,
  ip_off_entropy_c1_1: 0,
  ip_ttl_mean_c1_1: 31.5,
  ip_ttl_std_c1_1: 0.5,
  ip_ttl_max_c1_1: 32,
  ip_ttl_min_c1_1: 31,
  ip_ttl_entropy_c1_1: 1,
  tcp_ack_mean_c1_1: null,
  tcp_ack_std_c1_1: null,
  tcp_ack_max_c1_1: null,
  tcp_ack_min_c1_1: null,
  tcp_ack_entropy_c1_1: null,
  tcp_flags_mean_c1_1: null,
  tcp_flags_std_c1_1: null,
  tcp_flags_max_c1_1: null,
  tcp_flags_min_c1_1: null,
  tcp_flags_entropy_c1_1: null,
  tcp_seq_mean_c1_1: null,
  tcp_seq_std_c1_1: null,
  tcp_seq_max_c1_1: null,
  tcp_seq_min_c1_1: null,
  tcp_seq_entropy_c1_1: null,
  tcp_window_mean_c1_1: null,
  tcp_window_std_c1_1: null,
  tcp_window_max_c1_1: null,
  tcp_window_min_c1_1: null,
  tcp_window_entropy_c1_1: null,
  core_payload_bytes_mean_c1_1: 38,
  core_payload_bytes_std_c1_1: 0,
  core_payload_bytes_max_c1_1: 38,
  core_payload_bytes_min_c1_1: 38,
  core_payload_bytes_entropy_c1_1: 0,
  core_payload_mean_mean_c1_1: 54.526315789473685,
  core_payload_mean_std_c1_1: 0,
  core_payload_mean_max_c1_1: 54.526315789473685,
  core_payload_mean_min_c1_1: 54.526315789473685,
  core_payload_mean_entropy_c1_1: 0,
  core_payload_std_mean_c1_1: 55.35138321007614,
  core_payload_std_std_c1_1: 0,
  core_payload_std_max_c1_1: 55.35138321007614,
  core_payload_std_min_c1_1: 55.35138321007614,
  core_payload_std_entropy_c1_1: 0,
  core_payload_max_mean_c1_1: 203,
  core_payload_max_std_c1_1: 0,
  core_payload_max_max_c1_1: 203,
  core_payload_max_min_c1_1: 203,
  core_payload_max_entropy_c1_1: 0,
  core_payload_min_mean_c1_1: 0,
  core_payload_min_std_c1_1: 0,
  core_payload_min_max_c1_1: 0,
  core_payload_min_min_c1_1: 0,
  core_payload_min_entropy_c1_1: 0,
  core_payload_entropy_mean_c1_1: 3.7728288869959457,
  core_payload_entropy_std_c1_1: 0,
  core_payload_entropy_max_c1_1: 3.7728288869959457,
  core_payload_entropy_min_c1_1: 3.7728288869959457,
  core_payload_entropy_entropy_c1_1: 0,
  _id_c1_2: ObjectId('670f2d28a9648c9b50dc7607'),
  mediacy_pkt_info_set_id_c1_2: ObjectId('670f199ccf069a029e6cb21a'),
  NUSW_group_id_str_c1_2: ObjectId('670dd99acf069a029eec7acf'),
  NUSW_id_list_c1_2: [ ObjectId('6704dda013c0f134fde1140d') ],
  srcip_c1_2: '149.171.126.6',
  dstip_c1_2: '59.166.0.0',
  sport_c1_2: 53,
  dsport_c1_2: 1390,
  Ethernet_proto_c1_2: 2048,
  ip_proto_c1_2: 17,
  count_c1_2: 2,
  start_time_c1_2: 1421927414.322604,
  end_time_c1_2: 1421927414.322617,
  tcp_flags_c1_2: 0,
  layer_1_payload_size_mean_c1_2: 82,
  layer_1_payload_size_std_c1_2: 0,
  layer_1_payload_size_max_c1_2: 82,
  layer_1_payload_size_min_c1_2: 82,
  layer_1_payload_size_entropy_c1_2: 0,
  layer_2_payload_size_mean_c1_2: 62,
  layer_2_payload_size_std_c1_2: 0,
  layer_2_payload_size_max_c1_2: 62,
  layer_2_payload_size_min_c1_2: 62,
  layer_2_payload_size_entropy_c1_2: 0,
  layer_3_payload_size_mean_c1_2: 54,
  layer_3_payload_size_std_c1_2: 0,
  layer_3_payload_size_max_c1_2: 54,
  layer_3_payload_size_min_c1_2: 54,
  layer_3_payload_size_entropy_c1_2: 0,
  arp_op_mean_c1_2: null,
  arp_op_std_c1_2: null,
  arp_op_max_c1_2: null,
  arp_op_min_c1_2: null,
  arp_op_entropy_c1_2: null,
  icmp_code_mean_c1_2: null,
  icmp_code_std_c1_2: null,
  icmp_code_max_c1_2: null,
  icmp_code_min_c1_2: null,
  icmp_code_entropy_c1_2: null,
  icmp_type_mean_c1_2: null,
  icmp_type_std_c1_2: null,
  icmp_type_max_c1_2: null,
  icmp_type_min_c1_2: null,
  icmp_type_entropy_c1_2: null,
  igmp_type_mean_c1_2: null,
  igmp_type_std_c1_2: null,
  igmp_type_max_c1_2: null,
  igmp_type_min_c1_2: null,
  igmp_type_entropy_c1_2: null,
  ip_off_mean_c1_2: 16384,
  ip_off_std_c1_2: 0,
  ip_off_max_c1_2: 16384,
  ip_off_min_c1_2: 16384,
  ip_off_entropy_c1_2: 0,
  ip_ttl_mean_c1_2: 29.5,
  ip_ttl_std_c1_2: 0.5,
  ip_ttl_max_c1_2: 30,
  ip_ttl_min_c1_2: 29,
  ip_ttl_entropy_c1_2: 1,
  tcp_ack_mean_c1_2: null,
  tcp_ack_std_c1_2: null,
  tcp_ack_max_c1_2: null,
  tcp_ack_min_c1_2: null,
  tcp_ack_entropy_c1_2: null,
  tcp_flags_mean_c1_2: null,
  tcp_flags_std_c1_2: null,
  tcp_flags_max_c1_2: null,
  tcp_flags_min_c1_2: null,
  tcp_flags_entropy_c1_2: null,
  tcp_seq_mean_c1_2: null,
  tcp_seq_std_c1_2: null,
  tcp_seq_max_c1_2: null,
  tcp_seq_min_c1_2: null,
  tcp_seq_entropy_c1_2: null,
  tcp_window_mean_c1_2: null,
  tcp_window_std_c1_2: null,
  tcp_window_max_c1_2: null,
  tcp_window_min_c1_2: null,
  tcp_window_entropy_c1_2: null,
  core_payload_bytes_mean_c1_2: 54,
  core_payload_bytes_std_c1_2: 0,
  core_payload_bytes_max_c1_2: 54,
  core_payload_bytes_min_c1_2: 54,
  core_payload_bytes_entropy_c1_2: 0,
  core_payload_mean_mean_c1_2: 56.48148148148148,
  core_payload_mean_std_c1_2: 0,
  core_payload_mean_max_c1_2: 56.48148148148148,
  core_payload_mean_min_c1_2: 56.48148148148148,
  core_payload_mean_entropy_c1_2: 0,
  core_payload_std_mean_c1_2: 60.57249734691686,
  core_payload_std_std_c1_2: 0,
  core_payload_std_max_c1_2: 60.57249734691686,
  core_payload_std_min_c1_2: 60.57249734691686,
  core_payload_std_entropy_c1_2: 0,
  core_payload_max_mean_c1_2: 203,
  core_payload_max_std_c1_2: 0,
  core_payload_max_max_c1_2: 203,
  core_payload_max_min_c1_2: 203,
  core_payload_max_entropy_c1_2: 0,
  core_payload_min_mean_c1_2: 0,
  core_payload_min_std_c1_2: 0,
  core_payload_min_max_c1_2: 0,
  core_payload_min_min_c1_2: 0,
  core_payload_min_entropy_c1_2: 0,
  core_payload_entropy_mean_c1_2: 4.197236873673568,
  core_payload_entropy_std_c1_2: 0,
  core_payload_entropy_max_c1_2: 4.197236873673568,
  core_payload_entropy_min_c1_2: 4.197236873673568,
  core_payload_entropy_entropy_c1_2: 0
}
"""

"""
paper0> db.matching_UNSW_NB15_testing.findOne()
{
  _id: ObjectId('67108025d28b19a39717d186'),
  id: 1,
  dur: 0.000011,
  proto: 'udp',
  service: '-',
  state: 'INT',
  spkts: 2,
  dpkts: 0,
  sbytes: 496,
  dbytes: 0,
  rate: 90909.0902,
  sttl: 254,
  dttl: 0,
  sload: 180363632,
  dload: 0,
  sloss: 0,
  dloss: 0,
  sinpkt: 0.011,
  dinpkt: 0,
  sjit: 0,
  djit: 0,
  swin: 0,
  stcpb: 0,
  dtcpb: 0,
  dwin: 0,
  tcprtt: 0,
  synack: 0,
  ackdat: 0,
  smean: 248,
  dmean: 0,
  trans_depth: 0,
  response_body_len: 0,
  ct_srv_src: 2,
  ct_state_ttl: 2,
  ct_dst_ltm: 1,
  ct_src_dport_ltm: 1,
  ct_dst_sport_ltm: 1,
  ct_dst_src_ltm: 2,
  is_ftp_login: 0,
  ct_ftp_cmd: 0,
  ct_flw_http_mthd: 0,
  ct_src_ltm: 1,
  ct_srv_dst: 2,
  is_sm_ips_ports: 0,
  attack_cat: 'Normal',
  label: 0,
  id_str_list: [ ObjectId('6704e1af13c0f134fd07cc70') ],
  id_str_list_len: 1
}
"""

def merge_1c_feature():
    mongoDB_client = MongoClient("mongodb://127.0.0.1:27017/")
    UNSW_NB15_testing_table = mongoDB_client["paper0"]["matching_UNSW_NB15_testing"]
    UNSW_NB15_training_table = mongoDB_client["paper0"]["matching_UNSW_NB15_training"]
    
    UNSW_NB15_raw_table = mongoDB_client["paper2"]["data1008"]
    _2_category_features_table = mongoDB_client["paper5"]["output_2_category_features"]
    
    # output_table = mongoDB_client["paper5"]["merge_2_category_features"]
    output_table = mongoDB_client["paper5"]["merge_2_category_training_features"]
    
    
    data_len = UNSW_NB15_training_table.count_documents({})
    tqdm_t = tqdm(UNSW_NB15_training_table.find(), total=data_len)
    Index_t = 0
    for row in tqdm_t:
        id_str_list = row["id_str_list"]
        id_str_list_len = row["id_str_list_len"]
        
        id_str_t = row["_id"]
        
        del row["_id"]
        

        for id_str in id_str_list:
            
            _2_category_data = _2_category_features_table.find({"NUSW_id_list_c1_1": id_str})
            for _2_category_row in _2_category_data:
                del _2_category_row["_id"]
                
                _2_category_row = col_name.updata_dict_col(_2_category_row, "_c2")
                
                out_dict = {}
                out_dict = col_name.merge_dict(out_dict, row)
                out_dict = col_name.merge_dict(out_dict, _2_category_row)
                # del out_dict["_id"]
                # print(out_dict)
                # out_dict["index_t"] = Index_t
                # Index_t += 1
                out_dict["id_str"] = id_str_t
                output_table.insert_one(out_dict)
                # exit()

    tqdm_t.close()

    


def merge_1c_feature_2(mongoDB_client,
                       UNSW_NB15_training_testing_table, 
                       _2_category_features_table,
                       output_table,
                       has_done):
    # mongoDB_client = MongoClient("mongodb://127.0.0.1:27017/")
    # use paper0
    # db.matching_UNSW_NB15_testing.findOne()
    # db.matching_UNSW_NB15_testing.aggregate([ { $unwind: "$id_str_list" }, { $project: {_id:0, original_id: "$_id",  id_str_list: 1 } },{$out: "map_index"}] )
    # UNSW_NB15_training_testing_table = mongoDB_client["paper0"]["matching_UNSW_NB15_training"]
    # UNSW_NB15_testing_table = mongoDB_client["paper0"]["matching_UNSW_NB15_testing"]
    # paper2> db.data1008.findOne()
    # _2_category_features_table = mongoDB_client["paper2"]["data1008"]
    # output_table = mongoDB_client["paper5"]["merge_1_category_testing_features"]
    
    # _2_category_features_table = mongoDB_client["paper5"]["output_2_category_features"]
    # output_table = mongoDB_client["paper5"]["merge_2_category_training_features"]
    # output_table = mongoDB_client["paper5"]["merge_1_category_training_features"]
    
    # Added table to track processed records
    # has_done = mongoDB_client["paper5"]["has_done_merge_1c_"]

    data_len = UNSW_NB15_training_testing_table.count_documents({})
    tqdm_t = tqdm(total=data_len, ncols=100)

    # Start a session
    session = mongoDB_client.start_session()
    row_count = None
    try:
        # Initialize timestamp for session refresh
        refresh_timestamp = datetime.now()

        # Use a cursor with no timeout and a reasonable batch size for large datasets
        with UNSW_NB15_training_testing_table.find({}, no_cursor_timeout=True).batch_size(5000) as cursor:
            for row in cursor:
                tqdm_t.update()
                row_count = row
                id_str_list = row["id_str_list"]
                id_str_t = row["_id"]
                
                # Skip if already processed
                if has_done.find_one({"id_str": id_str_t}):
                    continue

                # Remove _id from the current row to avoid conflict during insertion
                row["id_raw"] = row["id"]
                del row["_id"]

                dict_list = []
                # Process each id_str in the list
                for id_str in id_str_list:
                    _2_category_data = _2_category_features_table.find({"_id": id_str})
                    # _2_category_data = _2_category_features_table.find({"NUSW_id_list_c1_1": id_str})
                    
                    
                    for _2_category_row in _2_category_data:
                        # del _2_category_row["_id"]

                        # Update column names in _2_category_row and merge with the current row
                        # _2_category_row = col_name.updata_dict_col(_2_category_row, "_c2")
                        _2_category_row = col_name.updata_dict_col(_2_category_row, "_c1")
                        
                        out_dict = {}
                        out_dict = col_name.merge_dict(out_dict, row)
                        out_dict = col_name.merge_dict(out_dict, _2_category_row)
                        # out_dict["id_str"] = id_str_t

                        dict_list = dict_list + [out_dict]
                        # Insert the merged document into the output table
                        # output_table.insert_one(out_dict)

                # Mark the current record as done
                output_table.insert_many(dict_list)
                has_done.insert_one({"id_str": id_str_t})

                # Refresh the session every 5 minutes to avoid timeout
                if (datetime.now() - refresh_timestamp) > timedelta(minutes=5):
                    print("Refreshing session...")
                    mongoDB_client.admin.command("refreshSessions", [session.session_id])
                    refresh_timestamp = datetime.now()

                # Sleep to avoid overloading the server
                time.sleep(0.1)

    except Exception as e:
        print(f"Error encountered: {str(e)}")
        traceback.print_exc()
        print(row_count)

    finally:
        # Ensure the cursor and session are properly closed
        tqdm_t.close()
        session.end_session()
    
    
def main():
    
    mongoDB_client = MongoClient("mongodb://127.0.0.1:27017/")
    _2_category_features_table = mongoDB_client["paper2"]["data1008"]
    has_done = mongoDB_client["paper5"]["has_done_merge_1c_"]
    
    # use paper0
    # db.matching_UNSW_NB15_testing.findOne()
    # db.matching_UNSW_NB15_testing.aggregate([ { $unwind: "$id_str_list" }, { $project: {_id:0, original_id: "$_id",  id_str_list: 1 } },{$out: "map_index"}] )
    
    
    
    
    
    UNSW_NB15_training_testing_table = mongoDB_client["paper0"]["matching_UNSW_NB15_training"]
    output_table = mongoDB_client["paper5"]["merge_1_category_training_features"]
    merge_1c_feature_2(mongoDB_client,
                       UNSW_NB15_training_testing_table, 
                       _2_category_features_table,
                       output_table, 
                       has_done)
    

    UNSW_NB15_training_testing_table = mongoDB_client["paper0"]["matching_UNSW_NB15_testing"]   
    output_table = mongoDB_client["paper5"]["merge_1_category_testing_features"]
    
    merge_1c_feature_2(mongoDB_client,
                       UNSW_NB15_training_testing_table, 
                       _2_category_features_table,
                       output_table, 
                       has_done)
    
    
    
    
    return



if __name__ == "__main__":
    # cd /data1/code_me/code2
    # vim main_merge_1c_feature.py
    # python main_merge_1c_feature.py
    # gg
    # .,$d
    # use paper5
    # db.merge_2_category_features.findOne()
    # db.merge_2_category_features.drop()
    # crontab -e
    # 43 13 * * * /data1/code_me/code2/solo_run_6.sh
    # db.merge_2_category_training_features
    # [ObjectId('6704df5e13c0f134fdf1ab45'), ObjectId('6704df5e13c0f134fdf1ab46')]
    main()