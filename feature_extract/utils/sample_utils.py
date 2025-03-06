import numpy as np
import pandas as pd
from feature_process.feature import cate_qualify

def calculate_class_num(file, data, class_col, all_count=0):
    if data is None:
        data = pd.read_csv(file)
        cate_qualify(data)
    counts = data[class_col].value_counts()
    count_df = pd.DataFrame(np.array([counts.index, counts.values]).T, columns=["attack_cat", "count"])
    target_percent = data[class_col].value_counts(normalize=True)
    print("Dataset target class proportions:{}".format(target_percent))
    if all_count > 0:
        for i in range(target_percent.shape[0]):
            count_df.iloc[i, 1] = int(target_percent[i] * all_count)
    return count_df
def keep_class_num_sample(target_percent_data_file, df, random_state=0, all_count=0):
    target_num = calculate_class_num(target_percent_data_file, None, "attack_cat", all_count)
    # Create an empty list to store the sampled DataFrames
    sampled_dfs = []

    print("Class proportions before sampling in this dataset:{}".format(df["attack_cat"].value_counts(normalize=True)))

    # Sample each class and store it in the list
    for i in range(len(target_num)):
        category = target_num.iloc[i, 0]
        size = target_num.iloc[i, 1]
        # Randomly sample from the class, with replace=False indicating sampling without replacement
        sampled_df = df[df['attack_cat'] == category].sample(n=size, replace=False, random_state=random_state)
        sampled_dfs.append(sampled_df)

    # Concatenate all sampled DataFrames
    result_df = pd.concat(sampled_dfs)
    # Sort by the original index order
    sampled_df = result_df.sort_index()

    percent = sampled_df["attack_cat"].value_counts(normalize=True)
    print("Class proportions after sampling in this dataset:{}".format(percent))
    return sampled_df

def calculate_class_percent(file, data, class_col):
    if data is None:
        data = pd.read_csv(file)
        cate_qualify(data)
    counts = data[class_col].value_counts()
    percent = counts / data.shape[0]
    percent_df = pd.DataFrame(np.array([percent.index, percent.values]).T, columns=["attack_cat", "percent"])
    return percent_df

def keep_class_percent_sample(target_percent_data_file, df, random_state=0):
    cate_qualify(df)

    target_percent = calculate_class_percent(target_percent_data_file, None, "attack_cat")
    print("Dataset target class proportions:{}".format(target_percent))
    sampled_df_list = []
    np.random.seed(random_state)

    # Total data size
    total_size = len(df)
    for i in range(len(target_percent)):
        category = target_percent.iloc[i, 0]
        target_ratio = target_percent.iloc[i, 1]
        category_data = df[df["attack_cat"] == category]  # Get all data for this category
        tem_size = len(category_data) / target_ratio
        if tem_size < total_size:
            total_size = tem_size
    for i in range(len(target_percent)):
        target_size = int(total_size * target_percent.iloc[i, 1])  # Calculate the number of samples to draw based on the target ratio
        category = target_percent.iloc[i, 0]
        category_data = df[df["attack_cat"] == category]  # Get all data for this category
        sampled_category_data = category_data.sample(n=target_size, random_state=random_state)

        sampled_df_list.append(sampled_category_data)
    # Concatenate all sampled data
    sampled_df = pd.concat(sampled_df_list).reset_index(drop=True)
    # Check the final proportions
    percent = calculate_class_percent(file=None, data=sampled_df, class_col="attack_cat")
    print("Class proportions after sampling in this dataset:{}".format(percent))
    return sampled_df