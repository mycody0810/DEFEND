import os

import pandas as pd
import numpy as np

def category_class_analysis(df, row):
    cate = df.head(row)["attack_cat"].value_counts()
    cate_per = cate / df.shape[0]
    pd.DataFrame(np.array([cate_per.index, cate_per.values, cate.values]).T, columns=["attack_cat", "percent", "count"]).to_csv("../input/us_features/feature_7_percent.csv")
    print(cate_per)
def category_pie_plot():
    import matplotlib.pyplot as plt
    data_df = pd.read_csv("../input/us_features/feature_7_percent.csv")
    labels = data_df["attack_cat"].values
    sizes = data_df["count"].values

    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.axis('equal')

    plt.show()

def category_class_analysis(file):
    df = pd.read_csv(file).head(10)
    df["sampled_percent"] = df["count"].astype(float) / sum(df["count"])
    for i in df.index:
        df.loc[i, "percent"] = round(df.loc[i, "percent"], 4)
        df.loc[i, "sampled_percent"] = round(df.loc[i, "sampled_percent"], 4)
    df.drop(["Unnamed: 0", "count"], axis=1, inplace=True)
    df.to_csv("../input/us_features/feature_1_oversample_1_percent.csv")

# category_class_analysis("../input/us_features/feature_1_oversample_1_count.csv")

# Handle missing values
def column_fillna(data, col, value):
    col_data = data.loc[:, col]
    df_col = col_data.fillna(value)
    data.loc[:, col] = df_col

# Handle anomalies in attack categories
def cate_qualify(data):
    column_fillna(data, 'attack_cat', 'Normal')
    data["attack_cat"] = data["attack_cat"].apply(lambda x: x.strip())
    data["attack_cat"] = data["attack_cat"].replace("Backdoors", "Backdoor")

def analysis_false_predict_label_percent(feature_dir, file_name):
    category_list = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic',
       'Normal', 'Reconnaissance', 'Shellcode', 'Worms']
    error_df = pd.read_csv(os.path.join(feature_dir, file_name))
    error_percent_df = pd.DataFrame({
        'actual_label': [],
        'predicted_label': [],
        'percent': []
    })
    for category in category_list:
        normal_errors_df = error_df[error_df['actual_label'] == category]
        # Count the number of mispredicted labels
        error_counts = normal_errors_df['predicted_label'].value_counts()
        # Calculate the proportion of each label
        error_percentages = (error_counts / len(normal_errors_df))

        # Output statistical results
        for i in range(len(error_counts.index)):
            row = error_percent_df.shape[0]
            error_percent_df.loc[row,'actual_label'] = category
            error_percent_df.loc[row,'predicted_label'] = error_counts.index[i]
            error_percent_df.loc[row,'percent'] = error_percentages.values[i]

    error_percent_df.to_csv(os.path.join(feature_dir, 'false_predict_percent.csv'))

# us_data = pd.read_csv("../input/us_features/feature_7.csv")
# cate_qualify(us_data)
# category_class_analysis(us_data, us_data.shape[0])

# confusion_matrix_analysis()

# analysis_false_predict_label_percent("../output_analysis/feature7/all/single_sample/RFC", "false_predict_feature.csv")

# category_pie_plot()