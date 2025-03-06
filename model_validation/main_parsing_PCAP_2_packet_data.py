import netflow.v9
import file_io
from tqdm import tqdm
import data_cover
import sys
import pandas
import time
import numpy as np
import os
from pymongo import MongoClient
# import netflow
from scapy.utils import RawPcapReader
# from scapy.utils import PacketMetadataNg
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, UDP, TCP
from pymongo import MongoClient 
from scapy.all import PcapReader
import dpkt
import col_name
import socket

# from scapy import 

from scapy.all import rdpcap
# from nfstream import NFStreamer
# from flowcontainer.extractor import extract


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


def mac_address(address):
    if len(address) != 6:
        return None
    return ':'.join('%02x' % dpkt.compat.compat_ord(b) for b in address)


def parser_tcp(tcp_data:dpkt.tcp.TCP, info_dict):
    if isinstance(tcp_data, dpkt.tcp.TCP) == False:
        if tcp_data.__len__() == 0:
            return info_dict
        else:
            tcp_data:dpkt.tcp.TCP = dpkt.tcp.TCP(tcp_data)
    info_dict["layer_3_proto"] = "TCP"
    info_dict["src_port"] = tcp_data.sport
    info_dict["dst_port"] = tcp_data.dport
    info_dict["tcp_flags"] = tcp_data.flags
    info_dict["tcp_seq"] = tcp_data.seq
    info_dict["tcp_ack"] = tcp_data.ack
    info_dict["tcp_window"] = tcp_data.win
    info_dict["tcp_options"] = tcp_data.opts
    # info_dict["tcp_data"] = tcp_data.data
    # info_dict["payload"] = tcp_data.payload
    info_dict["layer_3_payload_size"] = len(tcp_data.data)
    
    if tcp_data.data is not None:
        if bytes(tcp_data.data).__len__() != 0:
            info_dict["core_payload_bytes"] = bytes(tcp_data.data)
    
    return info_dict


def parser_udp(udp_data:dpkt.udp.UDP, info_dict):
    
    if isinstance(udp_data, dpkt.udp.UDP) == False:
        if udp_data.__len__() == 0:
            return info_dict
        else:
            udp_data:dpkt.udp.UDP = dpkt.udp.UDP(udp_data)
            
    info_dict["layer_3_proto"] = "UDP"
    info_dict["src_port"] = udp_data.sport
    info_dict["dst_port"] = udp_data.dport
    info_dict["layer_3_payload_size"] = len(udp_data.data)
    # info_dict["udp_data"] = udp_data.data
    if udp_data.data is not None:
        if bytes(udp_data.data).__len__() != 0:
            info_dict["core_payload_bytes"] = bytes(udp_data.data)
    
    return info_dict

def parser_icmp(icmp_data:dpkt.icmp.ICMP, info_dict):
    
    if isinstance(icmp_data, dpkt.icmp.ICMP) == False:
        if icmp_data.__len__() == 0:
            return info_dict
        else:
            icmp_data:dpkt.icmp.ICMP = dpkt.icmp.ICMP(icmp_data)
            
    info_dict["layer_3_proto"] = "ICMP"
    info_dict["icmp_type"] = icmp_data.type
    info_dict["icmp_code"] = icmp_data.code
    info_dict["layer_3_payload_size"] = len(icmp_data.data)
    # info_dict["icmp_data"] = icmp_data.data
    
    if icmp_data.data is not None:
        if bytes(icmp_data.data).__len__() != 0:
            info_dict["core_payload_bytes"] = bytes(icmp_data.data)
    
    return info_dict

def parser_igmp(igmp_data:dpkt.igmp.IGMP, info_dict):
    if isinstance(igmp_data, dpkt.igmp.IGMP) == False:
        if igmp_data.__len__() == 0:
            return info_dict
        else:
            igmp_data:dpkt.igmp.IGMP = dpkt.igmp.IGMP(igmp_data)
            
    info_dict["layer_3_proto"] = "IGMP"
    info_dict["igmp_type"] = igmp_data.type
    info_dict["layer_3_payload_size"] = len(igmp_data.data)
    
    if igmp_data.data is not None:
        if bytes(igmp_data.data).__len__() != 0:
            info_dict["core_payload_bytes"] = bytes(igmp_data.data)
        
    # info_dict["igmp_data"] = igmp_data.data
    return info_dict



def parser_arp(arp_data:dpkt.arp.ARP, info_dict):
    
    if isinstance(arp_data, dpkt.arp.ARP) == False:
        if arp_data.__len__() == 0:
            return info_dict
        else:
            arp_data:dpkt.arp.ARP = dpkt.arp.ARP(arp_data)
    
    info_dict["arp_op"] = arp_data.op
    info_dict["arp_sha"] = mac_address(arp_data.sha)
    info_dict["arp_spa"] = socket.inet_ntoa(arp_data.spa)
    info_dict["arp_tha"] = mac_address(arp_data.tha)
    info_dict["arp_tpa"] = socket.inet_ntoa(arp_data.tpa)
    info_dict["layer_2_proto"] = "ARP"
    info_dict["layer_2_payload_size"] = len(arp_data.data)
    info_dict["src_ip"] = info_dict["arp_spa"]
    info_dict["dst_ip"] = info_dict["arp_tpa"]
    
    if arp_data.data is not None:
        if bytes(arp_data.data).__len__() != 0:
            info_dict["core_payload_bytes"] = bytes(arp_data.data)
    
    return info_dict


# def parser_cdp(cdp_data:dpkt.cdp.CDP, info_dict):
#     info_dict["cdp_version"] = cdp_data.version
#     info_dict["cdp_ttl"] = cdp_data.ttl
#     info_dict["cdp_cksum"] = cdp_data.cksum
#     info_dict["cdp_data_size"] = len(cdp_data.data)
#     # info_dict["cdp_data"] = cdp_data.data
#     return info_dict

# def parser_dtp(dtp_data:dpkt.dtp.DTP, info_dict):
#     info_dict["dtp_version"] = dtp_data.version
#     info_dict["dtp_type"] = dtp_data.type
#     info_dict["dtp_data_size"] = len(dtp_data.data)
#     # info_dict["dtp_data"] = dtp_data.data
#     return info_dict


# def parser_llc(llc_data:dpkt.llc.LLC, info_dict):
#     info_dict["llc_dsap"] = llc_data.dsap
#     info_dict["llc_ssap"] = llc_data.ssap
#     info_dict["llc_ctrl"] = llc_data.ctrl
#     info_dict["llc_data_size"] = len(llc_data.data)
#     # info_dict["llc_data"] = llc_data.data
#     return info_dict



# def parser_ipx(ipx_data:dpkt.ipx.IPX, info_dict):
#     info_dict["layer_3_proto"] = "IPX"
#     info_dict["ipx_sum"] = ipx_data.sum
#     info_dict["ipx_len"] = ipx_data.len
#     info_dict["ipx_tc"] = ipx_data.tc
#     info_dict["ipx_pt"] = ipx_data.pt
#     info_dict["ipx_dst_net"] = ipx_data.dst_net
#     info_dict["ipx_dst_host"] = ipx_data.dst_host
#     info_dict["ipx_dst_port"] = ipx_data.dst_port
#     info_dict["ipx_src_net"] = ipx_data.src_net
#     info_dict["ipx_src_host"] = ipx_data.src_host
#     info_dict["ipx_src_port"] = ipx_data.src_port
#     info_dict["ipx_data_size"] = len(ipx_data.data)
#     # info_dict["ipx_data"] = ipx_data.data
#     return info_dict

# def parser_ppp(ppp_data:dpkt.ppp.PPP, info_dict):
#     info_dict["layer_3_proto"] = "PPP"
#     info_dict["ppp_addr"] = ppp_data.addr
#     info_dict["ppp_ctl"] = ppp_data.ctl
#     info_dict["ppp_data_size"] = len(ppp_data.data)
#     # info_dict["ppp_data"] = ppp_data.data
#     return info_dict

# def parser_pppoe(pppoe_data:dpkt.pppoe.PPPoE, info_dict):
#     info_dict["layer_3_proto"] = "PPPoE"
#     info_dict["pppoe_ver"] = pppoe_data.ver
#     info_dict["pppoe_type"] = pppoe_data.type
#     info_dict["pppoe_code"] = pppoe_data.code
#     info_dict["pppoe_sid"] = pppoe_data.sid
#     info_dict["pppoe_len"] = pppoe_data.len
#     info_dict["pppoe_data_size"] = len(pppoe_data.data)
#     # info_dict["pppoe_data"] = pppoe_data.data
#     return info_dict



def parser_ip(ip_data:dpkt.ip.IP, info_dict):
    
    if isinstance(ip_data, dpkt.ip.IP) == False:
        if ip_data.__len__() == 0:
            return info_dict
        else:
            ip_data:dpkt.ip.IP = dpkt.ip.IP(ip_data)
    
    info_dict["src_ip"] = socket.inet_ntoa(ip_data.src)
    info_dict["dst_ip"] = socket.inet_ntoa(ip_data.dst)
    info_dict["ip_proto"] = ip_data.p
    info_dict["ip_ttl"] = ip_data.ttl
    info_dict["ip_id"] = ip_data.id
    info_dict["ip_off"] = ip_data.off
    info_dict["ip_options"] = ip_data.opts
    info_dict["layer_2_proto"] = "IP"
    info_dict["layer_2_payload_size"] = len(ip_data.data)
    
    # info_dict["layer_3_proto_number"] = ip_data.p
    if ip_data.data is not None:
        if bytes(ip_data.data).__len__() != 0:
            info_dict["core_payload_bytes"] = bytes(ip_data.data)
    
    # print(type(ip_data.data), ip_data.data.__len__())
    try:
        if ip_data.p == dpkt.ip.IP_PROTO_TCP:
            info_dict = parser_tcp(ip_data.data, info_dict)
        elif ip_data.p == dpkt.ip.IP_PROTO_UDP:
            info_dict = parser_udp(ip_data.data, info_dict)
        elif ip_data.p == dpkt.ip.IP_PROTO_ICMP:
            info_dict = parser_icmp(ip_data.data, info_dict)
        elif ip_data.p == dpkt.ip.IP_PROTO_IGMP:
            info_dict = parser_igmp(ip_data.data, info_dict)
        else:
            info_dict["layer_3_proto"] = "Unknown"
            info_dict["layer_3_payload_size"] = len(ip_data.data)
    except Exception as e:
        print(e)
        print(ip_data.p)
        print(ip_data.data)
        info_dict["layer_3_proto"] = "Exception"
        info_dict["layer_3_payload_size"] = len(ip_data.data)
        
        
    return info_dict



def parser_pkt(pkt_bytes:bytes):
    # dpkt_eth = dpkt.ethernet.Ethernet(pkt_bytes)
    # print(dpkt_eth.type,
    #         mac_address(dpkt_eth.src), mac_address(dpkt_eth.dst),
    #         )
    dpkt_eth:dpkt.sll.SLL = dpkt.sll.SLL(pkt_bytes)
    

    info_dict = col_name.get_default_dict()
    
    info_dict["packet_type"] = dpkt_eth.type
    info_dict["link_layer_address_type"] = dpkt_eth.hrd
    info_dict["link_layer_address_length"] = dpkt_eth.hlen
    info_dict["src_mac"] = mac_address(dpkt_eth.hdr[:6])
    info_dict["dst_mac"] = mac_address(dpkt_eth.hdr[6:])
    info_dict["Ethernet_proto"] = dpkt_eth.ethtype
    info_dict["eth_size"] = len(pkt_bytes)

    
    payload_data:dpkt.ip.IP = dpkt_eth.data
    if payload_data == None:
        return info_dict

    if payload_data.__bytes__().__len__() != 0:
        info_dict["core_payload_bytes"] = payload_data.__bytes__()
    
    info_dict["layer_1_payload_size"] = len(payload_data)
    
    if dpkt_eth.ethtype == dpkt.ethernet.ETH_TYPE_IP:
        # print("IP")
        
        info_dict = parser_ip(payload_data, info_dict)
        # print(info_dict)
        # exit()
    elif dpkt_eth.ethtype == dpkt.ethernet.ETH_TYPE_ARP:
        info_dict = parser_arp(payload_data, info_dict)
    else:
        info_dict["layer_2_proto"] = "Unknown"
        info_dict["layer_2_payload_size"] = len(payload_data)
        # info_dict["proto_2_data"] = payload_data
        
    
    core_payload_bytes = info_dict["core_payload_bytes"]
    
    if core_payload_bytes is not None:
        if core_payload_bytes.__len__() > 0:
            np_array = np.array(bytearray(core_payload_bytes))
            info_dict["core_payload_mean"] = float(np_array.mean())
            info_dict["core_payload_std"] = float(np_array.std())
            info_dict["core_payload_max"] = int(np_array.max())
            info_dict["core_payload_min"] = int(np_array.min())
            info_dict["core_payload_entropy"] = float(calculate_entropy(np_array))
            info_dict["core_payload_len"] = int(np_array.__len__())
        
    
    
    return info_dict
        


def pcap2DB(in_pcap):
    
    # pcap_pyshark = pyshark.FileCapture(in_pcap, only_summaries=True)
    # pcap_pyshark.load_packets()
    # pcap_pyshark.reset()
    result_list = []
    frame_num = 0
    ignored_packets = 0
    # tqdm(rdpcap(in_pcap), desc='Reading packets', ncols=100)
    pcap_reader = RawPcapReader(in_pcap)
    # print(help(pcap_reader))
    # exit()
    # _, p_test = pcap_reader._read_packet()
    # print(p_test)
    # print(p_test.time)
    # exit()
    # print(p_test.sent_time)
    
    # pcap_reader = dpkt.pcap.Reader(open(in_pcap, 'rb'))
    
    
    tqdm_t = tqdm(pcap_reader, desc='Reading packets', ncols=100)
    
    for pkt_scapy in tqdm_t:
        # print(pkt_scapy)
        # print(type(pkt_scapy))
        pkt_bytes = pkt_scapy[0]
        PacketMetadataNg = pkt_scapy[1]
        
        # print(pkt_scapy)
        ts_high = PacketMetadataNg.tshigh
        ts_low = PacketMetadataNg.tslow
        ts_resol = PacketMetadataNg.tsresol
        timestamp = (ts_high << 32) + ts_low
        timestamp = timestamp / ts_resol

        info_dict = parser_pkt(pkt_bytes)

        
        info_dict["timestamp"] = float(timestamp)
        
        key_list = col_name.get_col_name()
        new_dict = {}
        for key_t in key_list:
            if key_t in info_dict:
                new_dict[key_t] = info_dict[key_t]
            else:
                new_dict[key_t] = None
        # print(info_dict)
        result_list = result_list + [new_dict]
    
    tqdm_t.close()
    # print(result_list)
    # exit()
        
    return result_list


def main():
    print(sys.argv)
    # pcap_path = "C:\\work_2014\\论文\\z202408\\code2\\test_pcap"
    if len(sys.argv) > 1:
        pcap_path = sys.argv[1]
    
    pcap_path = "/data/pcap_raw/pcap_fg/"
    # pcap_path = "D:\\pcaps"
    
    mongoDB_client = MongoClient("mongodb://127.0.0.1:27017/")
    paper_c = mongoDB_client["paper3"]["data1010"]
    
    has_done = mongoDB_client["paper3"]["data1010_hasdone"]
    
    pcap_list = file_io.get_file_list_all(pcap_path)
    
    tqdm_t = tqdm(pcap_list, desc='Reading pcap files', ncols=100)
    for pcap_file in pcap_list:
        pcap_file:str = pcap_file
        # print(pcap_file)
        tqdm_t.update(1)
        # /data/pcap_raw/pcap_fg/
        # "D:\\pcap_fg\\17-2-2015_1.pcap\\output_00000_20150218082327"
        has_done_path = "D:\\pcap_fg\\"
        if pcap_file.startswith("/data/pcap_raw/pcap_fg/"):
            sub_str = pcap_file.split("/data/pcap_raw/pcap_fg/")[1]
            sub_str = sub_str.replace("/", "\\")
            has_done_path = "D:\\pcap_fg\\" + sub_str
            
        else:
            has_done_path = pcap_file
        
        # print(has_done_path)
        # exit()
        
        if has_done.find_one({"pcap_file": has_done_path}):
            # print(pcap_file + " has done")
            # print(has_done_path)
            continue
        # exit()
        if not os.path.isfile(pcap_file):
            continue
        # if not pcap_file.endswith(".pcap"):
        #     continue
        
        result_list = pcap2DB(pcap_file)

        paper_c.insert_many(result_list)
        has_done.insert_one({"pcap_file": has_done_path})
        
    
    tqdm_t.close()
        # exit()
    return







if __name__ == "__main__":
    
    main()