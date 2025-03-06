import os
import shutil

import numpy as np
import tqdm
import traceback
import logging
import pickle


def read_file_byte(path_m):
    f = open(path_m, "rb")
    lines = f.read()
    return lines

def read_txt_file(path_m, encoding="utf-8"):
    f = open(path_m, encoding=encoding)
    lines = f.read()
    return lines


def write_txt_file(path_m, text_m, errors="ignore"):
    delete_file(path_m)
    file_object = open(path_m, 'wb')
    file_object.write(bytes(text_m, encoding="utf8", errors=errors))
    file_object.close()

def write_bytes_file(path_m, bytes_t):
    delete_file(path_m)
    file_object = open(path_m, 'wb')
    file_object.write(bytes_t)
    file_object.close()

# ,mode="a"

def write_txt_file_a(path_m, text_m):
    # delete_file(path_m)
    file_object = open(path_m, 'ab')
    file_object.write(bytes(text_m, encoding="utf8"))
    file_object.close()


def delete_file(path_m):
    if is_file_exists(path_m):
        os.remove(path_m)


def is_file_exists(path_m):
    return os.path.exists(path_m)


def get_dir_file_name(dir_path, call_back_func, call_back_func_arg):
    g = os.walk(dir_path)
    last_result = None
    total_len = 0
    total_file_list = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            file_full_name = os.path.join(path, file_name)
            total_file_list = total_file_list + [file_full_name]

    total_len = total_file_list.__len__()
    index_m = 0
    bar_m = tqdm.tqdm(total=total_len, desc="get_dir_file_name", ncols=90)
    for file_full_name in total_file_list:
        bar_m.update()
        try:
            last_result = call_back_func(last_result, index_m, total_len, file_full_name, call_back_func_arg)
        except Exception as e:
            print(file_full_name)
            print(e)
            traceback.format_exc()
            logging.exception(e)
            # exit()

        index_m = index_m + 1

    bar_m.close()
    return last_result


def get_file_list(dir_path):
    total_file_list = []
    for i in os.listdir(dir_path):
        path_file = os.path.join(dir_path, i)
        if os.path.isfile(path_file):
            total_file_list = total_file_list + [path_file]
        # else:
            # del_dir_file(path_file)
    return total_file_list

def get_file_list_all(dir_path):
    total_file_list = []
    for i in os.listdir(dir_path):
        path_file = os.path.join(dir_path, i)
        if os.path.isfile(path_file):
            total_file_list = total_file_list + [path_file]
        elif os.path.isdir(path_file):
            total_file_list = total_file_list + get_file_list_all(path_file)
        # else:
            # del_dir_file(path_file)
    return total_file_list


def del_dir_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_dir_file(path_file)

    return


def copy_dir_file(src_dir, tg_dir):
    for file in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file)
        target_file = os.path.join(tg_dir, file)
        shutil.copyfile(src_file, target_file)

    return


def copy_file_to_dir(src_file, tg_dir):

    file_path, file_name = os.path.split(src_file)

    target_file = os.path.join(tg_dir, file_name)
    shutil.copyfile(src_file, target_file)
    return


def get_file_size(file_path):
    if is_file_exists(file_path):
        return os.path.getsize(file_path)
    return 0


def split_file_name(file_path):
    filepath, tempfilename = os.path.split(file_path)
    shotname, extension = os.path.splitext(tempfilename)
    return filepath, shotname, extension


def write_obj_file(obj_t, out_put:str):
    obj_bytes:bytes = pickle.dumps(obj_t)
    write_bytes_file(out_put, obj_bytes)

    return 


def read_obj_file(file_path:str):
    obj_bytes:bytes = read_file_byte(file_path)
    obj_t = pickle.loads(obj_bytes)
    return obj_t


def get_dir_list(dir_path):
    total_file_list = []
    for i in os.listdir(dir_path):
        path_file = os.path.join(dir_path, i)
        if os.path.isdir(path_file):
            total_file_list = total_file_list + [path_file]
    return total_file_list
    






