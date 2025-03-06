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
from bson.objectid import ObjectId


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


def make_mediacy_group_5_tuples_time(raw_NB15_data, mediacy_group_5_tuples_time_name):
    pipeline = [
            {
                "$group": {
                    "_id": {
                        "srcip": "$srcip",
                        "dstip": "$dstip",
                        "proto": "$proto",
                        "sport": "$sport",
                        "dsport": "$dsport",
                        "start_time": "$Stime",
                        "end_time": "$Ltime",
                        
                    },
                    "id_list": {"$push": "$_id"},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "srcip": "$_id.srcip",
                    "dstip": "$_id.dstip",
                    "proto": "$_id.proto",
                    "sport": "$_id.sport",
                    "dsport": "$_id.dsport",
                    "start_time": "$_id.start_time",
                    "end_time": "$_id.end_time", 
                    "id_list": 1,                 
                }
            },
            {
                "$merge": mediacy_group_5_tuples_time_name 
            },
            
        ]
    raw_NB15_data.aggregate(pipeline)

    return
    
    
    
def make_mediacy_pkt_info_set(raw_pcap_data, mediacy_pkt_info_set_name,
                              srcip, dstip, ip_proto, Ethernet_proto, 
                              sport, dsport, start_time, end_time,
                              id_list, 
                              id_str, ):
    
    
    and_list = [
                    {"src_ip": srcip},
                    {"dst_ip": dstip},
                    {"timestamp": {"$gte": start_time - 1, "$lte": end_time + 1}}
                ]
    and2_list = [
                    {"src_ip": dstip},
                    {"dst_ip": srcip},
                    {"timestamp": {"$gte": start_time - 1, "$lte": end_time + 1}}
                ]
    
    
    
    pipeline = [
            {
                "$match": 
                    { 
                        "$or": [
                            {"$and": and_list},
                            {"$and": and2_list}
                        ]
                    }
            },
            {
                "$group": {
                    "_id": {
                        "srcip": "$src_ip",
                        "dstip": "$dst_ip",
                        "sport": "$src_port",
                        "dsport": "$dst_port",
                        "Ethernet_proto": "$Ethernet_proto",
                        "ip_proto": "$ip_proto",
                    },
                    "count": {"$sum": 1},
                    "start_time": {"$min": "$timestamp"},
                    "end_time": {"$max": "$timestamp"},
                    "timestamp_list": {"$push": "$timestamp"},
                    "layer_1_payload_size_list": {"$push": "$layer_1_payload_size"},
                    "layer_2_payload_size_list": {"$push": "$layer_2_payload_size"},
                    "layer_3_payload_size_list": {"$push": "$layer_3_payload_size"},
                    "arp_op_list": {"$push": "$arp_op"},
                    "icmp_code_list": {"$push": "$icmp_code"},
                    "icmp_type_list": {"$push": "$icmp_type"},
                    "igmp_type_list": {"$push": "$igmp_type"},
                    "ip_off_list": {"$push": "$ip_off"},
                    "ip_ttl_list": {"$push": "$ip_ttl"},
                    "tcp_ack_list": {"$push": "$tcp_ack"},
                    "tcp_flags_list": {"$push": "$tcp_flags"},
                    "tcp_seq_list": {"$push": "$tcp_seq"},
                    "tcp_window_list": {"$push": "$tcp_window"},
                    "core_payload_bytes_list": {"$push": "$core_payload_len"},
                    "core_payload_mean_list": {"$push": "$core_payload_mean"},
                    "core_payload_std_list": {"$push": "$core_payload_std"},
                    "core_payload_max_list": {"$push": "$core_payload_max"},
                    "core_payload_min_list": {"$push": "$core_payload_min"},
                    "core_payload_entropy_list": {"$push": "$core_payload_entropy"},
                }
            },
                
            {
                "$project": {
                    "_id": 0,
                    "srcip": "$_id.srcip",
                    "dstip": "$_id.dstip",
                    "proto": "$_id.proto",
                    "sport": "$_id.sport",
                    "dsport": "$_id.dsport",
                    "Ethernet_proto": "$_id.Ethernet_proto",
                    "ip_proto": "$_id.ip_proto",
                    "count": 1,
                    "start_time": 1,
                    "end_time": 1,
                    "timestamp_list": 1,
                    "layer_1_payload_size_list": 1,
                    "layer_2_payload_size_list": 1,
                    "layer_3_payload_size_list": 1,
                    "arp_op_list": 1,
                    "icmp_code_list": 1,
                    "icmp_type_list": 1,
                    "igmp_type_list": 1,
                    "ip_off_list": 1,
                    "ip_ttl_list": 1,
                    "tcp_ack_list": 1,
                    "tcp_flags_list": 1,
                    "tcp_seq_list": 1,
                    "tcp_window_list": 1,
                    "core_payload_bytes_list": 1,
                    "core_payload_mean_list": 1,
                    "core_payload_std_list": 1,
                    "core_payload_max_list": 1,
                    "core_payload_min_list": 1,
                    "core_payload_entropy_list": 1,
                    "NUSW_id_list": id_list,
                    "NUSW_group_id_str": id_str,
                           
                }
            },
            {
                "$merge": mediacy_pkt_info_set_name 
            },
                
        ]
    raw_pcap_data.aggregate(pipeline)
    
    # print(id_str)
    # exit()
    
    return



def calculate_statistics(x_list:list):
    if x_list is None:
        return None, None, None, None, None
    if len(x_list) == 0:
        return None, None, None, None, None
    
    if x_list[0] is None:
        return None, None, None, None, None
    
    x_list = np.array(x_list)
    x_mean = float(np.mean(x_list))
    x_std = float(np.std(x_list))
    x_max = float(np.max(x_list))
    x_min = float(np.min(x_list))
    x_entropy = float(calculate_entropy(x_list))
    
    return x_mean, x_std, x_max, x_min, x_entropy


def calculate_aggregate_features(aggregate_features_table:Collection, mediacy_pkt_info_set_db:Collection):
    
    data_len = mediacy_pkt_info_set_db.count_documents({})
    data_list = mediacy_pkt_info_set_db.find()
    
    for data_t in tqdm(data_list, total=data_len):
        aggregate_features_dict = {}
        srcip = data_t["srcip"]
        dstip = data_t["dstip"]
        # proto = data_t["proto"]
        sport = data_t["sport"]
        dsport = data_t["dsport"]
        Ethernet_proto = data_t["Ethernet_proto"]
        ip_proto = data_t["ip_proto"]
        
        NUSW_group_id_str = data_t["NUSW_group_id_str"]
        NUSW_id_list = data_t["NUSW_id_list"]
        mediacy_pkt_info_set_id = data_t["_id"]
        
        if aggregate_features_table.count_documents({"mediacy_pkt_info_set_id": mediacy_pkt_info_set_id}) != 0:
            print("already exist")
            # exit()
            continue
        
        # print(NUSW_group_id_str)
        # exit()
        
        
        count = data_t["count"]
        start_time = data_t["start_time"]
        end_time = data_t["end_time"]
        
        timestamp_list = data_t["timestamp_list"]
        layer_1_payload_size_list = data_t["layer_1_payload_size_list"]
        layer_2_payload_size_list = data_t["layer_2_payload_size_list"]
        layer_3_payload_size_list = data_t["layer_3_payload_size_list"]
        arp_op_list = data_t["arp_op_list"]
        icmp_code_list = data_t["icmp_code_list"]
        icmp_type_list = data_t["icmp_type_list"]
        igmp_type_list = data_t["igmp_type_list"]
        ip_off_list = data_t["ip_off_list"]
        ip_ttl_list = data_t["ip_ttl_list"]
        tcp_ack_list = data_t["tcp_ack_list"]
        tcp_flags_list = data_t["tcp_flags_list"]
        tcp_seq_list = data_t["tcp_seq_list"]
        tcp_window_list = data_t["tcp_window_list"]
        core_payload_bytes_list = data_t["core_payload_bytes_list"]
        core_payload_mean_list = data_t["core_payload_mean_list"]
        core_payload_std_list = data_t["core_payload_std_list"]
        core_payload_max_list = data_t["core_payload_max_list"]
        core_payload_min_list = data_t["core_payload_min_list"]
        core_payload_entropy_list = data_t["core_payload_entropy_list"]
        
        layer_1_payload_size_mean, layer_1_payload_size_std, layer_1_payload_size_max, layer_1_payload_size_min, layer_1_payload_size_entropy = \
            calculate_statistics(layer_1_payload_size_list)
        layer_2_payload_size_mean, layer_2_payload_size_std, layer_2_payload_size_max, layer_2_payload_size_min, layer_2_payload_size_entropy = \
            calculate_statistics(layer_2_payload_size_list)
        layer_3_payload_size_mean, layer_3_payload_size_std, layer_3_payload_size_max, layer_3_payload_size_min, layer_3_payload_size_entropy = \
            calculate_statistics(layer_3_payload_size_list)
        arp_op_mean, arp_op_std, arp_op_max, arp_op_min, arp_op_entropy = \
            calculate_statistics(arp_op_list)
        icmp_code_mean, icmp_code_std, icmp_code_max, icmp_code_min, icmp_code_entropy = \
            calculate_statistics(icmp_code_list)
        icmp_type_mean, icmp_type_std, icmp_type_max, icmp_type_min, icmp_type_entropy = \
            calculate_statistics(icmp_type_list)
        igmp_type_mean, igmp_type_std, igmp_type_max, igmp_type_min, igmp_type_entropy = \
            calculate_statistics(igmp_type_list)
        ip_off_mean, ip_off_std, ip_off_max, ip_off_min, ip_off_entropy = \
            calculate_statistics(ip_off_list)
        ip_ttl_mean, ip_ttl_std, ip_ttl_max, ip_ttl_min, ip_ttl_entropy = \
            calculate_statistics(ip_ttl_list)
        tcp_ack_mean, tcp_ack_std, tcp_ack_max, tcp_ack_min, tcp_ack_entropy = \
            calculate_statistics(tcp_ack_list)
        tcp_flags_mean, tcp_flags_std, tcp_flags_max, tcp_flags_min, tcp_flags_entropy = \
            calculate_statistics(tcp_flags_list)
        tcp_seq_mean, tcp_seq_std, tcp_seq_max, tcp_seq_min, tcp_seq_entropy = \
            calculate_statistics(tcp_seq_list)
        tcp_window_mean, tcp_window_std, tcp_window_max, tcp_window_min, tcp_window_entropy = \
            calculate_statistics(tcp_window_list)
        core_payload_bytes_mean, core_payload_bytes_std, core_payload_bytes_max, core_payload_bytes_min, core_payload_bytes_entropy = \
            calculate_statistics(core_payload_bytes_list)
        core_payload_mean_mean, core_payload_mean_std, core_payload_mean_max, core_payload_mean_min, core_payload_mean_entropy = \
            calculate_statistics(core_payload_mean_list)
        core_payload_std_mean, core_payload_std_std, core_payload_std_max, core_payload_std_min, core_payload_std_entropy = \
            calculate_statistics(core_payload_std_list)
        core_payload_max_mean, core_payload_max_std, core_payload_max_max, core_payload_max_min, core_payload_max_entropy = \
            calculate_statistics(core_payload_max_list)
        core_payload_min_mean, core_payload_min_std, core_payload_min_max, core_payload_min_min, core_payload_min_entropy = \
            calculate_statistics(core_payload_min_list)
        core_payload_entropy_mean, core_payload_entropy_std, core_payload_entropy_max, core_payload_entropy_min, core_payload_entropy_entropy = \
            calculate_statistics(core_payload_entropy_list)
            
            
        
        aggregate_TCP_Flags = None
        if tcp_flags_list is not None:
            aggregate_TCP_Flags = 0
            for tcp_flags in tcp_flags_list:
                
                if tcp_flags is None:
                    # print("Error: tcp_flags is None")
                    # exit()
                    continue
                
                aggregate_TCP_Flags = tcp_flags | aggregate_TCP_Flags
                aggregate_TCP_Flags = aggregate_TCP_Flags & 0xff
        
        aggregate_features_dict["mediacy_pkt_info_set_id"] = mediacy_pkt_info_set_id
        aggregate_features_dict["NUSW_group_id_str"] = NUSW_group_id_str
        aggregate_features_dict["NUSW_id_list"] = NUSW_id_list
        
        aggregate_features_dict["srcip"] = srcip
        aggregate_features_dict["dstip"] = dstip
        # aggregate_features_dict["proto"] = proto
        aggregate_features_dict["sport"] = sport
        aggregate_features_dict["dsport"] = dsport
        aggregate_features_dict["Ethernet_proto"] = Ethernet_proto
        aggregate_features_dict["ip_proto"] = ip_proto
        aggregate_features_dict["count"] = count
        aggregate_features_dict["start_time"] = start_time
        aggregate_features_dict["end_time"] = end_time
        aggregate_features_dict["tcp_flags"] = aggregate_TCP_Flags
        aggregate_features_dict["layer_1_payload_size_mean"] = layer_1_payload_size_mean
        aggregate_features_dict["layer_1_payload_size_std"] = layer_1_payload_size_std
        aggregate_features_dict["layer_1_payload_size_max"] = layer_1_payload_size_max
        aggregate_features_dict["layer_1_payload_size_min"] = layer_1_payload_size_min
        aggregate_features_dict["layer_1_payload_size_entropy"] = layer_1_payload_size_entropy
        aggregate_features_dict["layer_2_payload_size_mean"] = layer_2_payload_size_mean
        aggregate_features_dict["layer_2_payload_size_std"] = layer_2_payload_size_std
        aggregate_features_dict["layer_2_payload_size_max"] = layer_2_payload_size_max
        aggregate_features_dict["layer_2_payload_size_min"] = layer_2_payload_size_min
        aggregate_features_dict["layer_2_payload_size_entropy"] = layer_2_payload_size_entropy
        aggregate_features_dict["layer_3_payload_size_mean"] = layer_3_payload_size_mean
        aggregate_features_dict["layer_3_payload_size_std"] = layer_3_payload_size_std
        aggregate_features_dict["layer_3_payload_size_max"] = layer_3_payload_size_max
        aggregate_features_dict["layer_3_payload_size_min"] = layer_3_payload_size_min
        aggregate_features_dict["layer_3_payload_size_entropy"] = layer_3_payload_size_entropy
        aggregate_features_dict["arp_op_mean"] = arp_op_mean
        aggregate_features_dict["arp_op_std"] = arp_op_std
        aggregate_features_dict["arp_op_max"] = arp_op_max
        aggregate_features_dict["arp_op_min"] = arp_op_min
        aggregate_features_dict["arp_op_entropy"] = arp_op_entropy
        aggregate_features_dict["icmp_code_mean"] = icmp_code_mean
        aggregate_features_dict["icmp_code_std"] = icmp_code_std
        aggregate_features_dict["icmp_code_max"] = icmp_code_max
        aggregate_features_dict["icmp_code_min"] = icmp_code_min
        aggregate_features_dict["icmp_code_entropy"] = icmp_code_entropy
        aggregate_features_dict["icmp_type_mean"] = icmp_type_mean
        aggregate_features_dict["icmp_type_std"] = icmp_type_std
        aggregate_features_dict["icmp_type_max"] = icmp_type_max
        aggregate_features_dict["icmp_type_min"] = icmp_type_min
        aggregate_features_dict["icmp_type_entropy"] = icmp_type_entropy
        aggregate_features_dict["igmp_type_mean"] = igmp_type_mean
        aggregate_features_dict["igmp_type_std"] = igmp_type_std
        aggregate_features_dict["igmp_type_max"] = igmp_type_max
        aggregate_features_dict["igmp_type_min"] = igmp_type_min
        aggregate_features_dict["igmp_type_entropy"] = igmp_type_entropy
        aggregate_features_dict["ip_off_mean"] = ip_off_mean
        aggregate_features_dict["ip_off_std"] = ip_off_std
        aggregate_features_dict["ip_off_max"] = ip_off_max
        aggregate_features_dict["ip_off_min"] = ip_off_min
        aggregate_features_dict["ip_off_entropy"] = ip_off_entropy
        aggregate_features_dict["ip_ttl_mean"] = ip_ttl_mean
        aggregate_features_dict["ip_ttl_std"] = ip_ttl_std
        aggregate_features_dict["ip_ttl_max"] = ip_ttl_max
        aggregate_features_dict["ip_ttl_min"] = ip_ttl_min
        aggregate_features_dict["ip_ttl_entropy"] = ip_ttl_entropy
        aggregate_features_dict["tcp_ack_mean"] = tcp_ack_mean
        aggregate_features_dict["tcp_ack_std"] = tcp_ack_std
        aggregate_features_dict["tcp_ack_max"] = tcp_ack_max
        aggregate_features_dict["tcp_ack_min"] = tcp_ack_min
        aggregate_features_dict["tcp_ack_entropy"] = tcp_ack_entropy
        aggregate_features_dict["tcp_flags_mean"] = tcp_flags_mean
        aggregate_features_dict["tcp_flags_std"] = tcp_flags_std
        aggregate_features_dict["tcp_flags_max"] = tcp_flags_max
        aggregate_features_dict["tcp_flags_min"] = tcp_flags_min
        aggregate_features_dict["tcp_flags_entropy"] = tcp_flags_entropy
        aggregate_features_dict["tcp_seq_mean"] = tcp_seq_mean
        aggregate_features_dict["tcp_seq_std"] = tcp_seq_std
        aggregate_features_dict["tcp_seq_max"] = tcp_seq_max
        aggregate_features_dict["tcp_seq_min"] = tcp_seq_min
        aggregate_features_dict["tcp_seq_entropy"] = tcp_seq_entropy
        aggregate_features_dict["tcp_window_mean"] = tcp_window_mean
        aggregate_features_dict["tcp_window_std"] = tcp_window_std
        aggregate_features_dict["tcp_window_max"] = tcp_window_max
        aggregate_features_dict["tcp_window_min"] = tcp_window_min
        aggregate_features_dict["tcp_window_entropy"] = tcp_window_entropy
        aggregate_features_dict["core_payload_bytes_mean"] = core_payload_bytes_mean
        aggregate_features_dict["core_payload_bytes_std"] = core_payload_bytes_std
        aggregate_features_dict["core_payload_bytes_max"] = core_payload_bytes_max
        aggregate_features_dict["core_payload_bytes_min"] = core_payload_bytes_min
        aggregate_features_dict["core_payload_bytes_entropy"] = core_payload_bytes_entropy
        aggregate_features_dict["core_payload_mean_mean"] = core_payload_mean_mean
        aggregate_features_dict["core_payload_mean_std"] = core_payload_mean_std
        aggregate_features_dict["core_payload_mean_max"] = core_payload_mean_max
        aggregate_features_dict["core_payload_mean_min"] = core_payload_mean_min
        aggregate_features_dict["core_payload_mean_entropy"] = core_payload_mean_entropy
        aggregate_features_dict["core_payload_std_mean"] = core_payload_std_mean
        aggregate_features_dict["core_payload_std_std"] = core_payload_std_std
        aggregate_features_dict["core_payload_std_max"] = core_payload_std_max
        aggregate_features_dict["core_payload_std_min"] = core_payload_std_min
        aggregate_features_dict["core_payload_std_entropy"] = core_payload_std_entropy
        aggregate_features_dict["core_payload_max_mean"] = core_payload_max_mean
        aggregate_features_dict["core_payload_max_std"] = core_payload_max_std
        aggregate_features_dict["core_payload_max_max"] = core_payload_max_max
        aggregate_features_dict["core_payload_max_min"] = core_payload_max_min
        aggregate_features_dict["core_payload_max_entropy"] = core_payload_max_entropy
        aggregate_features_dict["core_payload_min_mean"] = core_payload_min_mean
        aggregate_features_dict["core_payload_min_std"] = core_payload_min_std
        aggregate_features_dict["core_payload_min_max"] = core_payload_min_max
        aggregate_features_dict["core_payload_min_min"] = core_payload_min_min
        aggregate_features_dict["core_payload_min_entropy"] = core_payload_min_entropy
        aggregate_features_dict["core_payload_entropy_mean"] = core_payload_entropy_mean
        aggregate_features_dict["core_payload_entropy_std"] = core_payload_entropy_std
        aggregate_features_dict["core_payload_entropy_max"] = core_payload_entropy_max
        aggregate_features_dict["core_payload_entropy_min"] = core_payload_entropy_min
        aggregate_features_dict["core_payload_entropy_entropy"] = core_payload_entropy_entropy
        
        
        aggregate_features_table.insert_one(aggregate_features_dict)
        
        
        
def joint_category_1_features(aggregate_features_table:Collection,
                              mediacy_group_5_tuples_time_db:Collection,
                              category_1_features_db:Collection
                              ):
    _5_tuples_time_data_len = mediacy_group_5_tuples_time_db.count_documents({})
    _5_tuples_time_data_list = mediacy_group_5_tuples_time_db.find()
    
    for _5_tuples_time_data_t in tqdm(_5_tuples_time_data_list, total=_5_tuples_time_data_len):
        srcip = _5_tuples_time_data_t["srcip"]
        dstip = _5_tuples_time_data_t["dstip"]
        proto = _5_tuples_time_data_t["proto"]
        sport = _5_tuples_time_data_t["sport"]
        dsport = _5_tuples_time_data_t["dsport"]
        start_time = _5_tuples_time_data_t["start_time"]
        end_time = _5_tuples_time_data_t["end_time"]
        id_list = _5_tuples_time_data_t["id_list"]
        
        ip_proto = None
        Ethernet_proto = None
        if proto == "TCP":
            ip_proto = 6
            Ethernet_proto = 2048
        elif proto == "UDP":
            ip_proto = 17
            Ethernet_proto = 2048
        elif proto == "ICMP":
            ip_proto = 1
            Ethernet_proto = 2048
            sport = None
            dsport = None
        elif proto == "ARP":
            ip_proto = None
            Ethernet_proto = 2054
        else:
            if srcip is not None:
                Ethernet_proto = 2048
        
        
        match_data_1 = None
        match_data_2 = None
        
        if ip_proto is not None \
            and Ethernet_proto is not None:
            match_data_1 = aggregate_features_table.find_one({"srcip": srcip, "dstip": dstip, "sport": sport, "dsport": dsport, "ip_proto": ip_proto, "Ethernet_proto": Ethernet_proto, "start_time": {"$gte": start_time}, "end_time": {"$lte": end_time}})
            match_data_2 = aggregate_features_table.find_one({"srcip": dstip, "dstip": srcip, "sport": dsport, "dsport": sport, "ip_proto": ip_proto, "Ethernet_proto": Ethernet_proto, "start_time": {"$gte": start_time}, "end_time": {"$lte": end_time}})
        elif Ethernet_proto is not None \
            and ip_proto is None:
            match_data_1 = aggregate_features_table.find_one({"srcip": srcip, "dstip": dstip, "sport": sport, "dsport": dsport, "Ethernet_proto": Ethernet_proto, "start_time": {"$gte": start_time}, "end_time": {"$lte": end_time}})
            match_data_2 = aggregate_features_table.find_one({"srcip": dstip, "dstip": srcip, "sport": dsport, "dsport": sport, "Ethernet_proto": Ethernet_proto, "start_time": {"$gte": start_time}, "end_time": {"$lte": end_time}})
        elif Ethernet_proto is None \
            and ip_proto is not None:
            match_data_1 = aggregate_features_table.find_one({"srcip": srcip, "dstip": dstip, "sport": sport, "dsport": dsport, "ip_proto": ip_proto, "start_time": {"$gte": start_time}, "end_time": {"$lte": end_time}})
            match_data_2 = aggregate_features_table.find_one({"srcip": dstip, "dstip": srcip, "sport": dsport, "dsport": sport, "ip_proto": ip_proto, "start_time": {"$gte": start_time}, "end_time": {"$lte": end_time}})
        else:
            if srcip is not None or dstip is not None:
                if sport is not None or dsport is not None:
                    match_data_1 = aggregate_features_table.find_one({"srcip": srcip, "dstip": dstip, "sport": sport, "dsport": dsport, "start_time": {"$gte": start_time}, "end_time": {"$lte": end_time}})
                    match_data_2 = aggregate_features_table.find_one({"srcip": dstip, "dstip": srcip, "sport": dsport, "dsport": sport, "start_time": {"$gte": start_time}, "end_time": {"$lte": end_time}})
                else:
                    match_data_1 = aggregate_features_table.find_one({"srcip": srcip, "dstip": dstip, "start_time": {"$gte": start_time}, "end_time": {"$lte": end_time}})
                    match_data_2 = aggregate_features_table.find_one({"srcip": dstip, "dstip": srcip, "start_time": {"$gte": start_time}, "end_time": {"$lte": end_time}})
            else:
                match_data_1 = None
                match_data_2 = None
        
        c1_features_dict = col_name.get_default_dict_characteristics_1_category()
        
        c1_features_dict["srcip"] = srcip
        c1_features_dict["dstip"] = dstip
        c1_features_dict["proto"] = proto
        c1_features_dict["sport"] = sport
        c1_features_dict["dsport"] = dsport
        c1_features_dict["start_time"] = start_time
        c1_features_dict["end_time"] = end_time
        c1_features_dict["id_list"] = id_list
        
        c1_features_dict["Ethernet_proto"] = Ethernet_proto
        c1_features_dict["ip_proto"] = ip_proto
        
        if match_data_1 is not None:
            match_data_1 = col_name.updata_dict_col(match_data_1, "_c1_1")
            c1_features_dict = col_name.update_dict(c1_features_dict, match_data_1)
        if match_data_2 is not None:
            match_data_2 = col_name.updata_dict_col(match_data_2, "_c1_2")
            c1_features_dict = col_name.update_dict(c1_features_dict, match_data_2)
        
        category_1_features_db.insert_one(c1_features_dict)


def split_index(raw_2_category_features_db:Collection, map_index_db:Collection):
    pipeline = [ 
                    { "$unwind": "$NUSW_id_list" }, 
                    { 
                        "$project": {
                            "_id":0, 
                            "original_id": "$_id",  
                            "NUSW_id_list": 1 
                            }
                    },
                    {
                        "$out": "map_index"
                    }
                ]
    print(pipeline)
    
    raw_2_category_features_db.aggregate(pipeline)
    pipeline = [
                    {
                        "$group":
                            {
                                "_id": "$NUSW_id_list", 
                                "count":
                                    {
                                        "$sum": 1
                                    }
                            }
                    }, 
                    {
                        "$project": 
                            {
                                "_id":0, 
                                "count":1,  
                                "NUSW_id": "$_id" 
                            }
                    }, 
                    {
                        "$out": "map_index_index"
                    }
                ]
    
    print(pipeline)
    
    map_index_db.aggregate(pipeline)
    

def expand_1_category_features(testing_data_db:Collection,
                               raw_2_category_features:Collection, 
                               output_features_db:Collection
                               ):
    
    data_len = testing_data_db.count_documents({})
    data_list = testing_data_db.find()
    
    tqdm_t = tqdm(data_list, total=data_len)
    for data_t in tqdm_t:
        NUSW_id_t = data_t["_id_c1"]
        
        output_dict = col_name.get_default_dict(col_name.get_col_name_1022())
        output_dict = col_name.update_dict(output_dict, data_t)
        
        data_2_features_count = raw_2_category_features.count_documents({"NUSW_id_list": NUSW_id_t})
        
        if data_2_features_count == 0:
            # print("no data_2_features")
            # print(NUSW_id_t)
            # print(data_t)
            output_features_db.insert_one(output_dict)
            continue
        
        data_2_features = \
            raw_2_category_features.find({"NUSW_id_list": NUSW_id_t})
        
        
        
        
        

        
        srcip = data_t["srcip_c1"]
        dstip = data_t["dstip_c1"]
        
        for data_2_feature in data_2_features:
            srcip_2 = data_2_feature["srcip"]
            dstip_2 = data_2_feature["dstip"]
            
            if srcip == srcip_2 and dstip == dstip_2:
                data_2_feature = col_name.updata_dict_col(data_2_feature, "_c2_1")
            else:
                data_2_feature = col_name.updata_dict_col(data_2_feature, "_c2_2")
                
            output_dict = col_name.update_dict(output_dict, data_2_feature)
        
        # del output_dict["_id"]
        # print(output_dict)
        # exit()
        output_features_db.insert_one(output_dict)
    tqdm_t.close()
    
    return 
            
            
            
def main():
    print(sys.argv)
    # vim main_characteristics_1_category.py
    # crontab -e
    # cd /data1/code_me/code2
    # pcap_path = "C:\\work_2014\\论文\\z202408\\code2\\test_pcap"
    # python main_characteristics_1_category2.py
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    mongoDB_client = MongoClient("mongodb://127.0.0.1:27017/")
    raw_pcap_data = mongoDB_client["paper3"]["data1010"]
    raw_NB15_data = mongoDB_client["paper2"]["data1008"]
    
    NB15_col_name_list = col_name.get_col_name_1008()
    pcap_col_name_list = col_name.get_col_name()
    
    mediacy_group_5_tuples_time_name = "mediacy_group_5_tuples_time"
    
    
    
    make_mediacy_group_5_tuples_time(raw_NB15_data, mediacy_group_5_tuples_time_name)
    
    mediacy_group_5_tuples_time_name_db = \
        mongoDB_client["paper2"][mediacy_group_5_tuples_time_name]
    mediacy_pkt_info_set_db = mongoDB_client["paper3"]["mediacy_pkt_info_set"]
    # {
    #     _id: ObjectId('670dc4f5cf069a029eaad038'),
    #     dsport: 0,
    #     dstip: '10.40.170.2',
    #     end_time: 1421927955,
    #     id_list: [
    #     ObjectId('6704dda413c0f134fde148c1'),
    #     ObjectId('6704dda413c0f134fde148c2')
    #     ],
    #     proto: 'ARP',
    #     sport: 0,
    #     srcip: '10.40.170.2',
    #     start_time: 1421927955
    # }
    
    data_count = mediacy_group_5_tuples_time_name_db.count_documents({})
    data_list = mediacy_group_5_tuples_time_name_db.find()
    tqdm_t = tqdm(data_list, total=data_count)
    for data_t in tqdm_t:
        dsport = data_t["dsport"]
        dstip = data_t["dstip"]
        end_time = data_t["end_time"]
        id_list = data_t["id_list"]
        proto = data_t["proto"]
        sport = data_t["sport"]
        srcip = data_t["srcip"]
        start_time = data_t["start_time"]
        ip_proto = None
        Ethernet_proto = None
        id_str = data_t["_id"]
        
        if mediacy_pkt_info_set_db.count_documents({"NUSW_group_id_str": id_str}) != 0:
            # print("has data")
            # exit()
            continue
        # print("no data")
        # print(type(id_str))
        # print(id_str)
        # exit()
        
        
        
        make_mediacy_pkt_info_set(raw_pcap_data,
                              "mediacy_pkt_info_set2",
                              srcip, dstip, 
                              ip_proto, Ethernet_proto, 
                              sport, dsport, 
                              start_time, end_time, 
                              id_list, 
                              id_str, 
                              )
    
    # use paper3 
    # db.mediacy_pkt_info_set.aggregate([ { $unwind: "$NUSW_id_list" }, { $project: {_id:0, original_id: "$_id",  NUSW_id_list: 1 } },{$out: "map_index"}] )
    # db.map_index.findOne()
    # db.map_index.aggregate([{$group:{_id:"$NUSW_id_list", count:{$sum: 1}}}, {$project: {_id:0, count:1,  NUSW_id: "$_id" }}, {$out: "map_index_index"}])
    # db.map_index_index.findOne()
    # db.map_index_index.find({count: {$gt:2}})
    
    
    # use paper4
    
    
    mediacy_pkt_info_set_db = \
        mongoDB_client["paper3"]["mediacy_pkt_info_set2"]
    
    raw_2_category_features = \
        mongoDB_client["paper4"]["raw_2_category_features"]
    
    calculate_aggregate_features(raw_2_category_features, mediacy_pkt_info_set_db)
    # use paper4
    # paper4> db.raw_2_category_features.aggregate([ { $unwind: "$NUSW_id_list" }, { $project: {_id:0, original_id: "$_id",  NUSW_id_list: 1 } },{$out: "map_index"}] )
    # db.map_index.aggregate([ { $group: { _id: "$NUSW_id_list" } }, { $count: "totalCount" }] )
    
    # db.map_index.aggregate([{$group:{_id:"$NUSW_id_list", count:{$sum: 1}}}, {$project: {_id:0, count:1,  NUSW_id: "$_id" }}, {$out: "map_index_index"}])
    
    # map_index_db = \
    #     mongoDB_client["paper4"]["map_index"]
    
    # split_index(raw_2_category_features, map_index_db)
    
    # map_index_index_db = \
    #     mongoDB_client["paper4"]["map_index_index"]
    
    # paper5> show tables
    # give_the_optimal_class_1_features
    # give_the_optimal_class_1_features_training
    testing_data_db = \
        mongoDB_client["paper5"]["give_the_optimal_class_1_features_testing"]
    # training_data_db = \
    #     mongoDB_client["paper5"]["give_the_optimal_class_1_features_training"] 
    
    
    #     paper4> db.raw_2_category_features.findOne()
    # {
    #   _id: ObjectId('670f2abba9648c9b50d501a4'),
    #   mediacy_pkt_info_set_id: ObjectId('670f1792cf069a029e6109db'),
    #   NUSW_group_id_str: ObjectId('670dd992cf069a029ee84762'),
    #   NUSW_id_list: [
    #     ObjectId('6704dda013c0f134fde11417'),
    #     ObjectId('6704dda013c0f134fde11418')
    #   ],   
    
    output_features_db = \
        mongoDB_client["paper5"]["output_2_category_features_testing"]
    # output_features_db = \
    #     mongoDB_client["paper5"]["output_2_category_features_training"]
    
    expand_1_category_features(testing_data_db,
                               raw_2_category_features, 
                               output_features_db
                               )
                               
    training_data_db = \
        mongoDB_client["paper5"]["give_the_optimal_class_1_features_training"] 
    output_features_db = \
        mongoDB_client["paper5"]["output_2_category_features_training"]
    
    expand_1_category_features(training_data_db,
                               raw_2_category_features, 
                               output_features_db
                               )
    
    
    return


if __name__ == "__main__":
    # rm ./out_main_characteristics_1_category2.log
    # vim main_characteristics_1_category2.py
    # python main_characteristics_1_category2.py >>./out_main_characteristics_1_category2.log 2>&1 &
    # cat ./out_main_characteristics_1_category2.log
    # db.raw_2_category_features.aggregate([ { $group: { _id: "$NUSW_group_id_str" } }, { $count: "totalCount" }] )
    main()