import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler


# Handle missing values
def column_fillna(data, col, value):
    col_data = data.loc[:, col]
    df_col = col_data.fillna(value)
    data.loc[:, col] = df_col


# Handle anomalies in attack_cat
def cate_qualify(data):
    column_fillna(data, 'attack_cat', 'Normal')
    data["attack_cat"] = data["attack_cat"].apply(lambda x: x.strip())
    data["attack_cat"] = data["attack_cat"].replace("Backdoors", "Backdoor")


# One-hot encoding
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False).astype(int)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each, axis=1)
    return df


# Function to min-max normalize
def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy()  # do not touch the original df
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        # if feature_name == "service_DNS":
        #     print(feature_name, max_value, min_value)
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def split_feature_label(new_train_df, timestep=-1):
    if timestep != -1:
        data_X, data_y = construct_sequence(new_train_df, timestep)
    else:
        new_train_df = new_train_df.sample(frac=1).reset_index(drop=True)
        data_y = new_train_df["Class"]
        data_X = new_train_df.drop('Class', axis=1)
    return data_X, data_y


# label->index
def cate2index(data, category_map):
    array = np.array(data)
    array = np.reshape(array, (array.size,))
    for i in range(len(array)):
        array[i] = category_map[array[i]]
    return array.astype(int)


# Random sampling with equal number of samples for each category
def class_balance_sample(df, x, y, n_samples_per_class=10, index_exclude=[]):
    import pandas as pd
    if "Class" not in df.columns:
        # Assume X_train is the training set and y_train is the corresponding label
        # Merge X_train and y_train for easy sampling by class
        df_x = pd.DataFrame(x)
        if y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        df_y = pd.DataFrame(y, columns=['Class'])
        df = pd.DataFrame(pd.concat([df_x, df_y], axis=1))
    # if index_exclude.size > 0:
    tem_df = df.drop(index=index_exclude, inplace=False, axis=0)

    sampled_df = tem_df.groupby('Class').apply(lambda a: a.sample(n_samples_per_class))
    index_exclude = [i[1] for i in sampled_df.index]
    sampled_df = sampled_df.reset_index(drop=True)
    # Sample from each class, assuming n_samples_per_class samples per class, or proportionally based on class size
    # sampled_df = df.groupby('Class').apply(lambda a: a.sample(n_samples_per_class)).reset_index(drop=True)

    # Separate features and labels
    background_X = sampled_df.drop(columns=['Class'])
    return background_X, index_exclude


# SHAPley method to obtain background and test data, maintaining class proportions with random sampling
def class_percent_sample(df, x, y, class_num=10, n_samples_per_class=10):
    # Ensure the background data covers all label classes by initially adding one sample from each class
    sampled_df_1 = df.groupby('Class').apply(lambda a: a.sample(1))
    index_exclude_1 = [i[1] for i in sampled_df_1.index]
    sampled_df_1 = sampled_df_1.reset_index(drop=True)
    # Remove the already sampled data
    df_1 = df.drop(index=index_exclude_1, inplace=False, axis=0)
    # Ensure the test data covers all label classes by initially adding one sample from each class
    sampled_df_2 = df_1.groupby('Class').apply(lambda a: a.sample(1))
    index_exclude_2 = [i[1] for i in sampled_df_2.index]
    sampled_df_2 = sampled_df_2.reset_index(drop=True)
    # Remove the already sampled data
    df_2 = df_1.drop(index=index_exclude_2, inplace=False, axis=0)

    # Sample proportionally based on the number of samples in each class in the training set
    from sklearn.model_selection import StratifiedShuffleSplit
    test_percent = (n_samples_per_class * class_num) / df_2.shape[0]
    stratified_shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=test_percent, train_size=test_percent,
                                                      random_state=42)
    background, test = None, None
    for train_index, test_index in stratified_shuffle_split.split(df_2.iloc[:, :-1], df_2.iloc[:, -1]):
        background = df_2.iloc[train_index]
        test = df_2.iloc[test_index]
    background = pd.concat([background, sampled_df_1], axis=0)
    test = pd.concat([test, sampled_df_2], axis=0)
    print(background.index)
    print(test.index)
    print(max(max(background.index), max(test.index)))
    background = background.drop(columns=['Class']).reset_index(drop=True)
    test = test.drop(columns=['Class']).reset_index(drop=True)
    return background, test

# get label_map
def get_label_map(data_y):
    data_df = pd.DataFrame(data_y)
    category_list = sorted(data_df.drop_duplicates().iloc[:, 0].tolist())
    category_map = {}
    for each in category_list:
        category_map[each] = category_list.index(each)
    print("category_map:{}".format(category_map))
    return category_list, category_map

def sample_target_num(data_X, data_Y, category_list, n_samples_per_class=10):
    from sklearn.utils import resample

    df = pd.concat([data_X, data_Y], axis=1).reset_index(drop=True)
    balanced_data = []
    for label in category_list:
        class_data = df[df['Class'] == label]
        if len(class_data) < n_samples_per_class:
            resampled_data = resample(
                class_data,
                replace=True,  # Enable sampling with replacement
                n_samples=n_samples_per_class,
                random_state=42
            )
            # If the number of samples is greater than the target number, perform under-sampling
        else:
            resampled_data = resample(
                class_data,
                replace=False,  # Disable sampling with replacement
                n_samples=n_samples_per_class,
                random_state=42
            )

            # Add the resampled data
        balanced_data.append(resampled_data)

    # Combine data from all classes
    balanced_df = pd.concat(balanced_data)
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    data_X_over, data_y_over = split_feature_label(balanced_df)
    return data_X_over, data_y_over

def get_data_with_target_category(all_x, all_y, category):
    import random
    target_data_df = all_x[all_y == category]
    # target_data_y = all_y[all_y == category]
    index = random.randint(0, len(target_data_df)-1)
    return target_data_df.iloc[index, :]


# Ensure the training set and test set contain the same categories
def confirm_all_category(train_X_over, train_y_over, test_X, test_y):
    train_category_list = train_y_over.unique().tolist()
    test_category_list = test_y.unique().tolist()
    if sorted(train_category_list) == sorted(test_category_list):
        return test_X, test_y
    for each in train_category_list:
        if each not in test_category_list:
            # Add a random data point of category 'each' to test_X
            single_x = get_data_with_target_category(train_X_over, train_y_over, each)
            test_X = pd.concat([test_X, single_x.to_frame().T], axis=0)
            test_y = pd.concat([test_y, pd.Series(data=each, index=single_x.to_frame().T.index)], axis=0)
    return test_X, test_y

def training_data_process(data_X, data_y, train_index, test_index, cate_cnt=1, category_list=None):
    if category_list is None:
        category_list = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
                         "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"]
    train_X, test_X = data_X.iloc[train_index], data_X.iloc[test_index]
    train_y, test_y = data_y.iloc[train_index], data_y.iloc[test_index]
    print(train_y.value_counts())
    train_X_over = train_X
    train_y_over = train_y
    # 0 Do not perform oversampling
    # 1 Oversample, the least frequent class is replicated to match the most frequent class
    # -1 Oversample, all classes are replicated to match the most frequent class
    # -2 ADASYN interpolation oversampling, oversample according to cate_cnt * the most frequent class
    # -3 Undersample, align with the least frequent class
    if cate_cnt > 10:
        train_X_over, train_y_over = sample_target_num(train_X_over, train_y, category_list, n_samples_per_class=cate_cnt)
    elif cate_cnt == -3:
        # Undersample to align with the least frequent class
        oversample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        train_X_over, train_y_over = oversample.fit_resample(train_X_over, train_y_over)
    elif cate_cnt == 0:
        # Do not perform sampling
        pass
    elif type(cate_cnt) is float:
        # Oversample using interpolation
        oversample = ADASYN(sampling_strategy='auto', random_state=42)
        train_X_over, train_y_over = oversample.fit_resample(train_X_over, train_y_over)
    else:
        if cate_cnt == -1:
            # Balance the dataset by randomly replicating minority class samples
            oversample = RandomOverSampler(sampling_strategy='minority')
            # Perform 9 rounds of oversampling for 10 classes
            for _ in range(9):
                train_X_over, train_y_over = oversample.fit_resample(train_X_over, train_y_over)
        else:  # cate_cnt == 1
            # Balance the dataset by randomly replicating minority class samples
            oversample = RandomOverSampler(sampling_strategy='minority')
            for _ in range(cate_cnt):
                train_X_over, train_y_over = oversample.fit_resample(train_X_over, train_y_over)
    test_X, test_y = confirm_all_category(train_X_over, train_y_over, test_X, test_y)
    print(train_y_over.value_counts())

    return train_X, test_X, train_y, test_y, train_X_over, train_y_over


def train_test_split(X, y, test_size=0.2, random_state=0):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y,
                                                        random_state=random_state)

    # Check the class distribution in the training and test sets
    print("Training set class distribution:\n", y_train.value_counts(normalize=True))
    print("Test set class distribution:\n", y_test.value_counts(normalize=True))

    print("Training set shape:", y_train.shape)
    print("Test set shape:", y_test.shape)
    return X_train, X_test, y_train, y_test

# process tcp flag
def process_tcp_flag(data, col):
    if col not in data.columns:
        return
    data[col] = data[col].apply(lambda x: 0 if x == -1 else x)
    data[col] = data[col].astype(int)
    # Define the order of flag bits
    flag_names = ["CWR", "ECE", "URG", "ACK", "PSH", "RST", "SYN", "FIN"]
    for i in range(len(flag_names)):
        flag_names[i] = col + "_" + flag_names[i]
    data[flag_names] = data[col].apply(lambda x: pd.Series(list(f"{x:08b}")))
    # Convert the data type of the new columns from string to integer
    data[flag_names] = data[flag_names].astype(int)

# process port
def process_port(data, col):
    if col not in data.columns:
        return
    data[col] = data[col].apply(classify_port)


def classify_port(port):
    if port < 0:
        return 0  # Unknown port
    elif port <= 1023:
        return 1  # Well-known ports 0-1023
    elif port <= 49151:
        return 2  # Registered ports 1024-49151
    else:
        return 3  # Private ports


def construct_sequence(data_df, timestep=50):
    columns = [data_df.shift(i) for i in range(timestep, 0, -1)]
    new_df = pd.concat(columns, axis=1)
    new_df.drop(index=[i for i in range(timestep)], axis=0, inplace=True)
    new_df = new_df.sample(frac=1).reset_index(drop=True)
    y = new_df.iloc[:, -1]
    x = new_df.drop(columns='Class', axis=1)
    return x, y
