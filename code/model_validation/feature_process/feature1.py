import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from feature_process.feature import *
from utils.sample_utils import *

# Process timestamp-related features
def features_cal_duration_(data, col1, col2, col_new):
    data[col_new] = data[col1] - data[col2]

def load_dataset(filepath, size=-1, shuffle=False, random_state=-1, target_percent_file=None):
    # Loading training set into dataframe
    combined_data = pd.read_csv(filepath).reset_index(drop=True)
    if shuffle:
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)
    print("===================Dataset loaded: {}===================".format(combined_data.shape))
    if random_state != -1:
        sampled_df = keep_class_percent_sample("../input/us_features/feature_1.csv", combined_data,
                                               random_state=random_state)
        print("=================Dataset sampled according to txxx-set ratio: random_state={}===================".format(random_state))
        return sampled_df
    return combined_data

def column_process(data, feature_ver):
    print(data.shape)
    drop_cols = ['label', 'id']
    for col in drop_cols:
        if col in data.columns:
            data.drop([col], axis=1, inplace=True)
    # Defining enum list
    cols = ['proto', 'state', 'service']
    # Applying one-hot encoding to combined data
    data = one_hot(data, cols)
    print("Number of initial feature fields after processing: {}".format(data.shape[1]))
    return data

def normalize_data(df, label_col):
    # Temporarily store the label column
    tmp = df.pop(label_col)
    # Normalizing the dataset
    new_train_df = normalize(df, df.columns)
    # Appending class column to the dataset
    new_train_df["Class"] = tmp
    print("Does the processed data have any null values: {}".format(new_train_df.isnull().values.any()))
    return new_train_df

def test_feature():
    # Parameter settings
    # Dataset splitting method: kfold or random
    split_mode = "kfold"
    # Feature version 1: feature1 original features 250k (train+test) feature1/raw
    feature_version = "feature1/raw"
    output_dir = os.path.join("../output_us/{}".format(feature_version))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_dir = os.path.join("../model_us/{}".format(feature_version))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    dataset_path = '../input/us_features/feature_1.csv'
    # Load the dataset
    combined_data = load_dataset(dataset_path, -1, shuffle=False)
    # Handle abnormal labels
    cate_qualify(combined_data)
    # Process dataset features
    combined_data = column_process(combined_data, feature_version)
    # Normalize
    new_train_df = normalize_data(combined_data, label_col='attack_cat')
    # Get feature set and label set
    combined_data_X, y = split_feature_label(new_train_df, random_state=0)
    # Map labels to indices (string->int)
    category_list, category_map = get_label_map(y)
    # # Split training and test sets
    # train_X, test_X, train_y, test_y, x_train_1, y_train_1, x_test_2, y_test_2 = train_test_split(combined_data_X, y, category_list, split_mode=split_mode)
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def get_feature(raw_dataset, label_col, feature_ver):
    # Handle abnormal labels
    cate_qualify(raw_dataset)
    # Process dataset features
    combined_data = column_process(raw_dataset, feature_ver=feature_ver)
    # Normalize
    new_train_df = normalize_data(combined_data, label_col=label_col)
    # Get feature set and label set
    combined_data_X, y = split_feature_label(new_train_df, random_state=0)
    # Map labels to indices (string->int)
    category_list, category_map = get_label_map(y)
    return combined_data_X, y, category_list, category_map

def process_feature(input_dir, feature_file, feature_version, label_col, dataset_dir, shuffle=False, random_state=-1, all_count=-1, target_file="feature_1.csv"):
    # Load the dataset
    combined_data = load_dataset(os.path.join(input_dir, feature_file), size=all_count, shuffle=False, random_state=random_state, target_percent_file=os.path.join(input_dir, target_file))

    # Handle abnormal labels
    cate_qualify(combined_data)
    # Process dataset features
    combined_data = column_process(combined_data, feature_ver=feature_version)
    # Normalize
    new_train_df = normalize_data(combined_data, label_col=label_col)

    new_train_df.to_csv(os.path.join(dataset_dir, "dataset.csv"), index=False)

    return new_train_df
