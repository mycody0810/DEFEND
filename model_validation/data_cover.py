import base64
import re
import binascii

import hashlib
import os
import pickle

import numpy as np
# import cv2
import binascii



def byte_to_image(data_m):
    image = np.frombuffer(data_m, dtype=np.ubyte)
    filesize = image.shape[0]
    # print(filesize)
    width = 256  # 设置图片宽度为256
    rem = filesize % width
    # print(rem)
    if rem != 0:
        image = image[:-rem]
    height = int(image.shape[0] / width)
    image = image.reshape(height, width)


    return image

def IP_str_2_int(ip_str):
    ip_list = ip_str.split(".")
    ip_int = 0
    for i in range(4):
        ip_int = ip_int * 256 + int(ip_list[i])
    return ip_int

def IP_int_2_str(ip_int):
    ip_str = ""
    for i in range(4):
        ip_str = str(ip_int % 256) + "." + ip_str
        ip_int = ip_int // 256
    ip_str = ip_str[:-1]
    return ip_str

# def byte_to_image_compression(data_m):
#     image = np.frombuffer(data_m, dtype=np.ubyte)
#     filesize = image.shape[0]
#     # print(filesize)
#     width = 256  # 设置图片宽度为256
#     rem = filesize % width
#     # print(rem)
#     if rem != 0:
#         image = image[:-rem]
#     height = int(image.shape[0] / width)
#     image = image.reshape(height, width)
#     image = cv2.resize(image, (10, 10), interpolation=cv2.INTER_AREA)

#     return image


# def byte_feature(data_m, feature_len):
#     image = np.frombuffer(data_m, dtype=np.ubyte)
#     filesize = image.shape[0]
#     if filesize < 1:
#         return 'null_info'
#     # print(filesize)
#     width = 1  # 设置图片宽度为256
#     rem = filesize % width
#     # print(rem)
#     if rem != 0:
#         image = image[:-rem]
#     height = int(image.shape[0] / width)
#     image = image.reshape(height, width)
#     image = cv2.resize(image, (1, feature_len), interpolation=cv2.INTER_AREA)

#     return image.tobytes().hex()

def check_list(list_t:list):
    if list_t == None:
        return False
    
    if str(type(list_t)).count("list") == 0:
        return False
    
    if list_t.__len__() == 0:
        return False
    
    return True


def is_data_list(list_t:list):
    
    if list_t == None:
        return False
    
    if str(type(list_t)).count("list") == 0:
        return False
    
    if list_t.__len__() == 0:
        return False
    
    return True


def is_data_dict(dict_t:dict):
    
    if dict_t == None:
        return False
    
    if str(type(dict_t)).count("dict") == 0:
        return False
    
    if dict_t.__len__() == 0:
        return False
    
    return True


def is_data_tuple(tuple_t:tuple):
    if tuple_t == None:
        return False
    
    if str(type(tuple_t)).count("tuple") == 0:
        return False
    
    if tuple_t.__len__() == 0:
        return False
    
    return True


def int_2_str_list(list_t:list):
    
    if check_list(list_t) == False:
        return None
    list_out = []
    
    for list_value in list_t:
        list_out = list_out + [str(list_value)]
        
    return list_out


def str_2_int_list(list_t:str):
    list_t = list(list_t)
    if check_list(list_t) == False:
        return None
    list_out = []
    
    for list_value in list_t:
        list_out = list_out + [int(list_value)]
        
    return list_out

def str_2_float_list(list_t:str):
    list_t = str_to_bytes(list_t)
    list_t = list(list_t)
    if check_list(list_t) == False:
        return None
    list_out = []
    
    for list_value in list_t:
        list_out = list_out + [float(list_value)]
        
    return list_out


def is_data_str(str_t:str):
    if str_t == None:
        return False
    
    if str(type(str_t)).count("str") == 0:
        return False
    
    if str_t.__len__() == 0:
        return False
    
    return True


def is_data_bytes(bytes_t:bytes):
    if bytes_t == None:
        return False
    
    if str(type(bytes_t)).count("bytes") == 0:
        return False
    
    if bytes_t.__len__() == 0:
        return False
    
    return True


def is_data_int(int_t:int):
    if int_t == None:
        return False
    
    if str(type(int_t)).count("int") == 0:
        return False
    
    return True


def obj_to_hex_str(obj_t):
    if obj_t == None:
        return None
    obj_bytes = pickle.dumps(obj_t)
    
    if is_data_bytes(obj_bytes) == False:
        return None
    
    return bytes_to_str_hex(obj_bytes)


def hex_str_to_obj(hex_str:str):
    if hex_str == None:
        return None
    
    obj_bytes = hex_str_to_bytes(hex_str)
    if is_data_bytes(obj_bytes) == False:
        return None
    
    obj_t = pickle.loads(obj_bytes)
    
    return obj_t


def hex_str_to_bytes(hex_str:str):
    if hex_str == None:
        return None
    return bytes.fromhex(hex_str)

def bytes_to_str_hex(bytes_val:bytes):
    if bytes_val == None:
        return None
    return binascii.hexlify(bytes_val).decode('utf-8')

def bytes_to_str(bytes_val:bytes):
    return bytes_val.decode("utf-8") 


def hex_str_to_str(hex_str:str):
    return bytes_to_str(hex_str_to_bytes(hex_str))


def str_to_hex_str(str_t:str):
    if is_data_str(str_t) == False:
        return None
    test_str = str_to_bytes(str_t)
    str_hex = bytes_to_str_hex(test_str)
    return str_hex
    
def str_to_bytes(text_m:str, encoding='utf-8'):
    return bytes(text_m, encoding=encoding)

def str_to_hash(text_m:str, encoding='utf-8'):
    bytes_t = str_to_bytes(text_m, encoding)
    return bytes_to_hash(bytes_t)

def bytes_to_hash(bytes_t:bytes):
    
    bytes_list = list(bytes_t)
    int_hash = 0x2e1f2e1f
    int_hash_2 = 0
    for byte_t in bytes_list:
        int_t = int(byte_t)
        int_hash_2 = int_hash_2 + int_t
        int_hash = (int_hash << 1) + int_t + ((int_hash_2 << 16) ^ int_hash)
        int_hash = int_hash & 0xFFFFFFFF
        
    int_hash = int_hash & 0xFFFFFFFF
    sum_hash = int_hash_2 & 0xFFFFFFFF
    return sum_hash, int_hash

def base64_encode_string(text_m, encoding='utf-8'):
    return base64.b64encode(bytes(text_m, encoding=encoding))

def base64_decode_string(text_m):
    return base64.b64decode(text_m)

def get_base64_string(text_m):
    return re.compile("^[a-zA-Z0-9+/=]+$").findall(text_m)


def get_data_from_dict(dict_t:dict, key_t, def_ret=None):
    if is_data_dict(dict_t) == False:
        return def_ret
    
    if key_t not in dict_t.keys():
        return def_ret
    
    if dict_t[key_t] == None:
        return def_ret
    
    return dict_t[key_t]


def get_file_md5_8096(filename):
    if not os.path.isfile(filename):
        return
    myhash = hashlib.md5()
    f = open(filename, 'rb')
    b = f.read(8096)
    myhash.update(b)
    f.close()
    return myhash.hexdigest()


def get_file_md5(filename):
    if not os.path.isfile(filename):
        return
    myhash = hashlib.md5()
    f = open(filename, 'rb')
    while True:
        b = f.read()
        if not b:
            break
        myhash.update(b)
    f.close()
    return myhash.hexdigest()


def get_string_md5(string_m):
    myhash = hashlib.md5()
    myhash.update(bytes(str(string_m), encoding='utf-8'))
    return myhash.hexdigest()


