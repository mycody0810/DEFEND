import os
from feature_process.feature import *
from utils.sample_utils import *

# Process timestamp-related features
def features_cal_duration_(data, col1, col2, col_new):
    data[col_new] = data[col1] - data[col2]

def load_dataset(filepath, size=-1, shuffle=False, random_state=-1, target_percent_file=None):
    # Loading training set into dataframe
    combined_data = pd.read_csv(filepath, low_memory=False).reset_index(drop=True)
    if shuffle:
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)
    print("===================Dataset loaded: {}===================".format(combined_data.shape))
    if random_state != -1:
        if size != -1:  # Sample by specified total count
            sampled_df = keep_class_num_sample(target_percent_file, combined_data, random_state=random_state,
                                               all_count=size)
        else:  # Sample by maximizing proportion
            sampled_df = keep_class_percent_sample(target_percent_file, combined_data,
                                                   random_state=random_state)
        print("=================Dataset sampled according to txxx-set ratio: random_state={}===================".format(random_state))
        return sampled_df
    return combined_data

def find_string_by_prefix(columns, prefixes):
    drop_list = []
    for col in columns:
        for prefix in prefixes:
            if col.startswith(prefix):
                drop_list.append(col)
    return drop_list

def column_process(data, feature_ver):
    print(data.shape)
    data = process_normal(data)

    if "raw" in feature_ver.split("/"):
        # If only raw features are retained
        drop_cols = data.columns[42:]
        data = data.drop(drop_cols, axis=1)
        data = process_raw_feature(data)
    elif "second" in feature_ver.split("/"):
        # If only second-order features are retained
        drop_cols = data.columns[:41]
        data = data.drop(drop_cols, axis=1)
        data = process_second_feature(data)
    else:
        # Retain both raw and second-order features
        data = process_second_feature(data)
        data = process_raw_feature(data)

    if "super-one" in feature_ver.split("/"):
        data = process_test_feature(data, "super-one")
    print("Is any null value：", data.isnull().values.any())  # Check for any missing values
    print("After feature processing, the number of initial feature fields is {}".format(data.shape[1]))
    return data

# General processing
def process_normal(data):
    # Drop irrelevant columns
    drop_cols = ['Label', 'srcip', 'sport', 'dstip', 'dsport', 'Ltime', 'Stime']
    data = data.drop(drop_cols, axis=1)
    # Handle anomalies in the ct_ftp_cmd feature
    data["ct_ftp_cmd"] = data["ct_ftp_cmd"].apply(lambda x: 0 if x == ' ' else int(x)).astype(int)
    # Handle missing values
    column_fillna(data, 'is_ftp_login', 0)
    column_fillna(data, 'ct_flw_http_mthd', 0)
    column_fillna(data, 'attack_cat', 'Normal')
    data.fillna(0, inplace=True)
    return data
def process_raw_feature(data):
    # defining enum list
    cols = ['proto', 'state', 'service']
    # Applying one hot encoding to combined data
    data = one_hot(data, cols)
    return data
def process_second_feature(data):
    # timestamp
    col1_list = ["start_time_c2_1", "start_time_c2_2"]
    col2_list = ["end_time_c2_1", "end_time_c2_2"]
    col_new_list = ["dur_time_c2_1", "dur_time_c2_2"]
    for i in range(len(col1_list)):
        features_cal_duration_(data, col1_list[i], col2_list[i], col_new_list[i])
    data = data.drop(col1_list + col2_list, axis=1)
    # tcp flag
    aggregate_TCP_Flags_list = ["tcp_flags_c2_1", "tcp_flags_c2_2"]
    for col in aggregate_TCP_Flags_list:
        if col in data.columns:
            process_tcp_flag(data, col)
            data = data.drop([col], axis=1)
    return data
def process_test_feature(data, ver):
    # Customized features super-one
    if ver == "super-one":
        remain_cols = []
        enum_col = [
            # "proto",
            "service",
            # "state",
            "tcp_flags_c2",
        ]
        for col in data.columns:
            for ecol in enum_col:
                if col.startswith(ecol):
                    remain_cols.append(col)
                    continue
        remain_cols += [
            "attack_cat",
            "layer_3_payload_size_min_c2_2",
            "layer_3_payload_size_min_c2_1",
            "layer_3_payload_size_mean_c2_2",
            "layer_3_payload_size_mean_c2_1",
            "layer_3_payload_size_max_c2_2",
            "layer_3_payload_size_max_c2_1",
            "layer_2_payload_size_std_c2_2",
            "layer_2_payload_size_std_c2_1",
            "layer_2_payload_size_min_c2_2",
            "layer_2_payload_size_min_c2_1",
            "layer_2_payload_size_mean_c2_2",
            "layer_2_payload_size_mean_c2_1",
            "layer_2_payload_size_max_c2_2",
            "layer_2_payload_size_max_c2_1",
            "layer_1_payload_size_min_c2_2",
            "layer_1_payload_size_min_c2_1",
            "layer_1_payload_size_max_c2_2",
            "layer_1_payload_size_max_c2_1",
            "tcp_window_min_c2_2",
            "tcp_window_min_c2_1",
            "tcp_flags_min_c2_2",
            "tcp_flags_min_c2_1",
            # "tcp_flags_c2_2",
            # "tcp_flags_c2_1",
            "ip_ttl_min_c2_2",
            "ip_ttl_min_c2_1",
            "ip_ttl_mean_c2_2",
            "ip_ttl_mean_c2_1",
            "ip_ttl_max_c2_2",
            "ip_ttl_max_c2_1",
            "ip_off_min_c2_2",
            "ip_off_min_c2_1",
            "ip_off_mean_c2_2",
            "ip_off_mean_c2_1",
            "ip_off_max_c2_2",
            "ip_off_max_c2_1",
            "core_payload_std_min_c2_2",
            "core_payload_std_min_c2_1",
            "core_payload_std_mean_c2_2",
            "core_payload_std_mean_c2_1",
            "core_payload_std_max_c2_2",
            "core_payload_std_max_c2_1",
            "core_payload_min_mean_c2_2",
            "core_payload_min_mean_c2_1",
            "core_payload_mean_min_c2_2",
            "core_payload_mean_min_c2_1",
            "core_payload_mean_mean_c2_2",
            "core_payload_mean_mean_c2_1",
            "core_payload_mean_max_c2_2",
            "core_payload_mean_max_c2_1",
            "core_payload_max_min_c2_2",
            "core_payload_max_min_c2_1",
            "core_payload_max_mean_c2_2",
            "core_payload_max_mean_c2_1",
            "core_payload_entropy_min_c2_2",
            "core_payload_entropy_min_c2_1",
            "core_payload_entropy_mean_c2_2",
            "core_payload_entropy_mean_c2_1",
            "core_payload_entropy_max_c2_2",
            "core_payload_entropy_max_c2_1",
            "is_sm_ips_ports",
            "ct_state_ttl",
            "ct_flw_http_mthd",
            "is_ftp_login",
            "ct_ftp_cmd",
            "ct_srv_src",
            "ct_srv_dst",
            "ct_dst_ltm",
            "ct_src_ltm",
            "ct_src_dport_ltm",
            "ct_dst_sport_ltm",
            "ct_dst_src_ltm",
            "dur",

        ]
        for col in remain_cols:
            if col not in data.columns:
                print("{} not in data.columns".format(col))
        data = data[remain_cols]
        return data
    # Customized features test-three
    if ver == "test-three":
        remain_cols = ["attack_cat"]
        # Customized features three
        remain_cols += [
            "core_payload_entropy_max_c2_1",
            "core_payload_std_min_c2_1",
            "core_payload_bytes_max_c2_1",
            "core_payload_entropy_mean_c2_1",
            "core_payload_mean_mean_c2_1",
            "core_payload_min_std_c2_1",
            "core_payload_bytes_min_c2_1",
            "core_payload_max_min_c2_1",
            "core_payload_std_max_c2_1",
            "core_payload_min_mean_c2_1",
            "ip_ttl_max_c2_1",
            "ip_off_mean_c2_1",
            "ip_ttl_mean_c2_1",
            "ip_off_max_c2_1",
            "ip_ttl_min_c2_1",
            "ip_off_min_c2_1",
            "ip_off_max_c2_2",
            "ip_off_mean_c2_2",
            "ip_off_min_c2_2",
            "layer_2_payload_size_min_c2_1",
            "layer_3_payload_size_min_c2_1",
            "layer_1_payload_size_max_c2_1",
            "layer_3_payload_size_max_c2_1",
            "layer_1_payload_size_min_c2_1",
            "layer_2_payload_size_max_c2_1",
            "layer_1_payload_size_mean_c2_1",
            "layer_3_payload_size_mean_c2_1",
            "layer_2_payload_size_mean_c2_1",
            "layer_3_payload_size_std_c2_1",
            "ct_dst_src_ltm",
            "ct_srv_dst",
            "ct_srv_src",
            "ct_state_ttl",

            "proto_UDP",
            "service_-",
            "service_DNS",
            "proto_3pc",
            "proto_A/N",
            "proto_AES-SP3-D",
            "proto_ARIS",
            "proto_BBN-RCC",
            "proto_BNA",
            "proto_CFTP",
            "proto_CRTP",
            "proto_DCN",
            "proto_DDX",
            "proto_DGP",
            "proto_ETHERIP",
            "proto_HMP",
            "proto_IATP",
            "proto_IPIP",
            "proto_IPLT",
            "proto_IPNIP",
            "proto_IPV6",
            "proto_IPV6-FRAG",
            "proto_IPV6-NO",
            "proto_IPV6-OPTS",
            "proto_ISO-TP4",
            "proto_MERIT-INP",
            "proto_MFE-NSP",
            "proto_MICP",
            "proto_MOBILE",
            "proto_MTP",
            "proto_NSFNET-IGP",
            "proto_NVP",
            "proto_PGM",
            "proto_PNNI",
            "proto_PUP",
            "proto_PVP",
            "proto_RDP",
            "proto_RSVP",
            "proto_RVD",
            "proto_SAT-EXPAK",
            "proto_SCPS",
            "proto_SECURE-VMTP",
            "proto_SMP",
            "proto_STP",
            "proto_SWIPE",
            "proto_TCP",
            "proto_UTI",
            "proto_VINES",
            "proto_VRRP",
            "proto_WSN",
            "proto_XNET",
            "proto_XNS-IDP",
            "service_FTP",
            "service_FTP-DATA",
            "service_HTTP",
            "state_INT",
        ]
        data = data[remain_cols]
        return data
    return data

def normalize_data(df, label_col):
    # Before normalization, replace all -1 values with 0
    df.replace(-1, 0, inplace=True)
    # Temporarily store the label column
    tmp = df.pop(label_col)
    # Normalizing the training set
    new_train_df = normalize(df, df.columns)
    # Appending the class column to the training set
    new_train_df["Class"] = tmp
    print("Does the processed data have any null values: {}".format(new_train_df.isnull().values.any()))
    return new_train_df

def test_feature():
    # Parameter settings
    model_name = "RFC"
    # Dataset splitting method: kfold or random
    split_mode = "kfold"
    # Feature version 5: new feature5/all, old feature5/raw
    feature_version = "feature5/raw"
    output_dir = os.path.join("../output_us/{}".format(feature_version))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_dir = os.path.join("../model_us/{}".format(feature_version))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    dataset_path = '../input/us_features/feature_5.csv'
    # Load the dataset
    combined_data = load_dataset(dataset_path)
    # Handle abnormal labels
    cate_qualify(combined_data)
    # Process dataset features
    combined_data = column_process(combined_data, feature_ver=feature_version)
    # Normalize
    new_train_df = normalize_data(combined_data, label_col='attack_cat')
    # Get feature set and label set
    combined_data_X, y = split_feature_label(new_train_df, timestep=100, random_state=0)
    # Map labels to indices (string->int)
    category_list, category_map = get_label_map(y)
    # Split training and test sets
    # train_X, test_X, train_y, test_y, x_train_1, y_train_1, x_test_2, y_test_2 = train_test_split(combined_data_X, y, category_list, split_mode=split_mode)
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


def get_feature(raw_dataset, label_col, feature_ver, model_name):
    # Handle abnormal labels
    cate_qualify(raw_dataset)
    # Process dataset features
    combined_data = column_process(raw_dataset, feature_ver=feature_ver)
    # Normalize
    new_train_df = normalize_data(combined_data, label_col=label_col)
    # Get feature set and label set
    combined_data_X, y = split_feature_label(new_train_df, timestep=50, random_state=0)
    # Map labels to indices (string->int)
    category_list, category_map = get_label_map(y)
    return combined_data_X, y, category_list, category_map


def process_feature(input_dir, feature_file, feature_version, label_col, dataset_dir, shuffle=False, random_state=-1, all_count=-1, target_file="feature_1.csv"):
    # Load the dataset
    combined_data = load_dataset(os.path.join(input_dir, feature_file), size=all_count, shuffle=shuffle, random_state=random_state, target_percent_file=os.path.join(input_dir, target_file))

    # Handle abnormal attack_cat labels
    cate_qualify(combined_data)
    # Process dataset features
    combined_data = column_process(combined_data, feature_ver=feature_version)
    # Normalize
    new_train_df = normalize_data(combined_data, label_col=label_col)
    # Do not save the processed dataset when sampling by target dataset class proportions
    if all_count == -1 and random_state == -1:
        new_train_df.to_csv(os.path.join(dataset_dir, "dataset.csv"), index=False)

    return new_train_df