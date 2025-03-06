# 25w随机抽样Efficient模型feature1\2\12表现
def draw_25w_sample_accuracy_detect_rate_false_positive(file, info_start, feature_ver, model_name):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import ast

    df_all = pd.read_csv(file)
    df = pd.DataFrame(columns=df_all.columns)
    for i in range(df_all.shape[0]):
        info = df_all.iloc[i]["info"]
        feature_version = df_all.iloc[i]["feature_version"]
        model_n = df_all.iloc[i]["model_name"]
        if info.startswith(info_start) and feature_version == feature_ver and model_n == model_name:
            df.loc[len(df)] = df_all.iloc[i]

    # num = [int(x.split('，')[0].split('=')[-1]) for x in info]
    num_set = df["all_count"].values
    accuracy = df["val_accuracy"].values
    detection_rate = df['detect_rate'].values
    false_positive = df['false_positive'].values
    precision_weighted = df['precision_weighted'].values
    recall_weighted = df['recall_weighted'].values
    f1_weighted = df['f1_weighted'].values
    accuracy_score = np.array([num_set, accuracy]).T
    detection_rate_score = np.array([num_set, detection_rate]).T
    false_positive_score = np.array([num_set, false_positive]).T
    precision_weighted_score = np.array([num_set, precision_weighted]).T
    recall_weighted_score = np.array([num_set, recall_weighted]).T
    f1_weighted_score = np.array([num_set, f1_weighted]).T

    accuracy_score_map = {}
    accuracy_score_list = [[], []]
    for i in range(len(accuracy_score)):
        num = accuracy_score[i][0]
        accuracy = ast.literal_eval(accuracy_score[i][1])
        if num in accuracy_score_map:
            accuracy_score_map[num].extend(accuracy)
        else:
            accuracy_score_map[num] = accuracy
    for num, value in accuracy_score_map.items():
        accuracy_score_list[0].append(num)
        accuracy_score_list[1].append(np.average(value))

    detection_rate_map = {}
    detection_rate_list = [[], []]
    for i in range(len(detection_rate_score)):
        num = detection_rate_score[i][0]
        detection_rate = ast.literal_eval(detection_rate_score[i][1])
        if num in detection_rate_map:
            detection_rate_map[num].extend(detection_rate)
        else:
            detection_rate_map[num] = detection_rate
    for num, value in detection_rate_map.items():
        detection_rate_list[0].append(num)
        detection_rate_list[1].append(np.average(value))

    false_positive_map = {}
    false_positive_list = [[], []]
    for i in range(len(false_positive_score)):
        num = false_positive_score[i][0]
        false_positive = ast.literal_eval(false_positive_score[i][1])
        if num in false_positive_map:
            false_positive_map[num].extend(false_positive)
        else:
            false_positive_map[num] = false_positive
    for num, value in false_positive_map.items():
        false_positive_list[0].append(num)
        false_positive_list[1].append(np.average(value))

    precision_weighted_map = {}
    precision_weighted_list = [[], []]
    for i in range(len(precision_weighted_score)):
        num = precision_weighted_score[i][0]
        precision_weighted = ast.literal_eval(precision_weighted_score[i][1])
        if num in precision_weighted_map:
            precision_weighted_map[num].extend(precision_weighted)
        else:
            precision_weighted_map[num] = precision_weighted
    for num, value in precision_weighted_map.items():
        precision_weighted_list[0].append(num)
        precision_weighted_list[1].append(np.average(value))

    recall_weighted_map = {}
    recall_weighted_list = [[], []]
    for i in range(len(recall_weighted_score)):
        num = recall_weighted_score[i][0]
        recall_weighted = ast.literal_eval(recall_weighted_score[i][1])
        if num in recall_weighted_map:
            recall_weighted_map[num].extend(recall_weighted)
        else:
            recall_weighted_map[num] = recall_weighted
    for num, value in recall_weighted_map.items():
        recall_weighted_list[0].append(num)
        recall_weighted_list[1].append(np.average(value))

    f1_weighted_map = {}
    f1_weighted_list = [[], []]
    for i in range(len(f1_weighted_score)):
        num = f1_weighted_score[i][0]
        f1_weighted = ast.literal_eval(f1_weighted_score[i][1])
        if num in f1_weighted_map:
            f1_weighted_map[num].extend(f1_weighted)
        else:
            f1_weighted_map[num] = f1_weighted
    for num, value in f1_weighted_map.items():
        f1_weighted_list[0].append(num)
        f1_weighted_list[1].append(np.average(value))

    plt.figure(figsize=(10, 6))
    accuracy_set = [[], []]
    for num, accuracy in accuracy_score_map.items():
        # num = accuracy_unit[0]
        # accuracy = ast.literal_eval(accuracy_unit[1])
        for acc in accuracy:
            accuracy_set[0].append(num)
            accuracy_set[1].append(acc)
    plt.scatter(accuracy_set[0], accuracy_set[1], label="Accuracy", marker='o')
    detection_set = [[], []]
    for detection_unit in detection_rate_score:
        num = detection_unit[0]
        detection_rate = ast.literal_eval(detection_unit[1])
        for detect in detection_rate:
            detection_set[0].append(num)
            detection_set[1].append(detect)
    # plt.scatter(detection_set[0], detection_set[1], label="Detection Rate", marker='x')
    false_positive_set = [[], []]
    for false_positive_unit in false_positive_score:
        num = false_positive_unit[0]
        false_positive_rate = ast.literal_eval(false_positive_unit[1])
        for detect in false_positive_rate:
            false_positive_set[0].append(num)
            false_positive_set[1].append(detect)
    # plt.scatter(false_positive_set[0], false_positive_set[1], label="False Positive", marker='x')
    precision_weighted_set = [[], []]
    for precision_weighted_unit in precision_weighted_score:
        num = precision_weighted_unit[0]
        precision_weighted_rate = ast.literal_eval(precision_weighted_unit[1])
        for detect in precision_weighted_rate:
            precision_weighted_set[0].append(num)
            precision_weighted_set[1].append(detect)
    # plt.scatter(precision_weighted_set[0], precision_weighted_set[1], label="Detection Rate", marker='x')
    recall_weighted_set = [[], []]
    for recall_weighted_unit in recall_weighted_score:
        num = recall_weighted_unit[0]
        recall_weighted_rate = ast.literal_eval(recall_weighted_unit[1])
        for detect in recall_weighted_rate:
            recall_weighted_set[0].append(num)
            recall_weighted_set[1].append(detect)
    # plt.scatter(recall_weighted_set[0], recall_weighted_set[1], label="Detection Rate", marker='x')
    f1_weighted_set = [[], []]
    for f1_weighted_unit in f1_weighted_score:
        num = f1_weighted_unit[0]
        f1_weighted_rate = ast.literal_eval(f1_weighted_unit[1])
        for detect in f1_weighted_rate:
            f1_weighted_set[0].append(num)
            f1_weighted_set[1].append(detect)
    plt.scatter(f1_weighted_set[0], f1_weighted_set[1], label="F1 Wighted", marker='x')

    plt.plot(accuracy_score_list[0], accuracy_score_list[1], label="Accuracy", linestyle='--')
    # plt.plot(detection_rate_list[0], detection_rate_list[1], label="Detection Rate", linestyle='--')
    # plt.plot(false_positive_list[0], false_positive_list[1], label="False Positive", linestyle='--')
    # plt.plot(precision_weighted_list[0], precision_weighted_list[1], label="Precision Weighted", linestyle='--')
    # plt.plot(recall_weighted_list[0], recall_weighted_list[1], label="Recall Weighted", linestyle='--')
    plt.plot(f1_weighted_list[0], f1_weighted_list[1], label="F1 Weighted", linestyle='--')

    # 设置 x轴 和 y轴 标签
    plt.xlabel("Dataset Count", fontsize=12)
    plt.ylabel("Metrics", fontsize=12)

    # 设置标题
    plt.title("25w random sample on {} model with Feature {}".format(model_name, feature_ver.split("/")[1]),
              fontsize=15)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 展示图形
    plt.savefig(os.path.join("../output_us",
                             "25w随机抽样{}模型feature{}表现.png".format(model_name, feature_ver.split("/")[1])))
    plt.show()


# draw_25w_sample_accuracy_detect_rate_false_positive("../output_us/output_process_info.csv", info_start="按照比例随机抽样25w进行数据集测试", feature_ver="feature7/all/single_sample", model_name="LR")

# 随机小批量抽样，一张图一个指标所有模型均值曲线
def draw_25w_sample_accuracy_detect_rate_false_positive_all_model(file, info_start, feature_ver, metric):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import ast
    from scipy import interpolate

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun']  # 宋体和Time New Roman

    df_all = pd.read_csv(file, encoding="utf-8")
    df = pd.DataFrame(columns=df_all.columns)
    for i in range(df_all.shape[0]):
        info = df_all.iloc[i]["info"]
        feature_version = df_all.iloc[i]["feature_version"]
        if info.startswith(info_start) and feature_version == feature_ver:
            df.loc[len(df)] = df_all.iloc[i]

    model_name = df["model_name"].values
    num_set = df["all_count"].values
    metric_values = df[metric].values
    metric_score = np.array([num_set, metric_values, model_name]).T

    metric_score_map = {}
    for i in range(len(metric_score)):
        num = metric_score[i][0]
        metric_avg = np.average(ast.literal_eval(metric_score[i][1]))
        model = metric_score[i][2]
        # if model == "Efficient":
        #     continue
        #     pass

        if model not in metric_score_map:
            metric_score_map[model] = {}
        if num not in metric_score_map[model]:
            metric_score_map[model][num] = metric_avg
        else:
            metric_score_map[model][num] = (metric_score_map[model][num] + metric_avg) / 2
        #
        # if model in metric_score_map:
        #     metric_score_map[model][num] = (metric_score_map[model][num] + metric_avg) / 2
        # else:
        #     metric_score_map[model][num] = metric_avg
    max_y = 0
    for model, score_map in metric_score_map.items():
        x = list(score_map.keys())
        y = list(score_map.values())
        max_y = max(max_y, max(y))
        sql = interpolate.CubicSpline(x, y)
        x_dense = np.linspace(min(x), max(x), 200)
        y_dense = sql(x_dense)
        plt.plot(x, y, label=model, linestyle='--')
        # plt.plot(x_dense, y_dense, label=model, linestyle='--')

    # 设置 x轴 和 y轴 标签
    plt.xlabel("样本数量", fontsize=12, fontname="SimSun")
    plt.ylabel("分数", fontsize=12, fontname="SimSun")

    # 设置仅显示纵轴0.6-1.0
    plt.ylim(0.65, min(1.0, max_y + 0.03))
    # 设置标题
    plt.title("25w random sample on {} Metric with Feature {}".format(metric, feature_ver.split("/")[1]) , fontsize=15)
    # plt.title("数据集规模对分类准确率的影响".format(metric, feature_ver.split("/")[1]), fontsize=15, fontname="SimSun")

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 展示图形
    plt.savefig(os.path.join("../output_us", "25w随机抽样所有模型在指标{}特征{}表现_part.png".format(metric,
                                                                                                     feature_ver.split(
                                                                                                         "/")[1])))
    plt.show()


# 指标取值：val_accuracy、f1_weighted、false_positive、detect_rate
# draw_25w_sample_accuracy_detect_rate_false_positive_all_model("../result/feature7/RFC/single/实验三-小批量随机抽样稳定性验证/小批量抽样验证稳定性-avg.csv", info_start="按照比例随机抽样25w进行数据集测试", feature_ver="feature7/all/single_sample", metric="val_accuracy")


# 随机小批量抽样，一张图一个指标一个模型两种特征均值曲线
def draw_25w_sample_accuracy_detect_rate_false_positive_two_feature(file, info_start, target_model, metric):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import ast
    from scipy import interpolate
    feature_ver_list = ["feature7/all/single_sample", "feature7/raw/single_sample"]
    for feature_ver in feature_ver_list:
        df_all = pd.read_csv(file, encoding="utf-8")
        df = pd.DataFrame(columns=df_all.columns)
        for i in range(df_all.shape[0]):
            info = df_all.iloc[i]["info"]
            feature_version = df_all.iloc[i]["feature_version"]
            if info.startswith(info_start) and feature_version == feature_ver:
                df.loc[len(df)] = df_all.iloc[i]

        model_name = df["model_name"].values
        num_set = df["all_count"].values
        metric_values = df[metric].values
        metric_score = np.array([num_set, metric_values, model_name]).T

        metric_score_map = {}
        for i in range(len(metric_score)):
            num = metric_score[i][0]
            metric_avg = np.average(ast.literal_eval(metric_score[i][1]))
            model = metric_score[i][2]
            if model != target_model:
                continue
                pass

            if model not in metric_score_map:
                metric_score_map[model] = {}
            if num not in metric_score_map[model]:
                metric_score_map[model][num] = metric_avg
            else:
                metric_score_map[model][num] = (metric_score_map[model][num] + metric_avg) / 2
            #
            # if model in metric_score_map:
            #     metric_score_map[model][num] = (metric_score_map[model][num] + metric_avg) / 2
            # else:
            #     metric_score_map[model][num] = metric_avg
        for model, score_map in metric_score_map.items():
            x = list(score_map.keys())
            y = list(score_map.values())
            sql = interpolate.CubicSpline(x, y)
            x_dense = np.linspace(min(x), max(x), 100)
            y_dense = sql(x_dense)
            plt.plot(x, y, label=feature_ver.split("/")[1], linestyle='--')
            # plt.plot(x_dense, y_dense, label=feature_ver.split("/")[1], linestyle='--')

    # 设置 x轴 和 y轴 标签
    plt.xlabel("Dataset Count", fontsize=12)
    plt.ylabel("Score", fontsize=12)

    # 设置标题
    plt.title("25w random sample on {} Metric with Feature {}".format(metric, feature_ver.split("/")[1]), fontsize=15)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 展示图形
    plt.savefig(os.path.join("../output_us", "25w随机抽样模型{}在指标{}对比特征表现.png".format(target_model, metric)))
    plt.show()


# 指标取值：val_accuracy、f1_weighted、false_positive、detect_rate
# draw_25w_sample_accuracy_detect_rate_false_positive_two_feature("../output_us/25w_percent_sample_predict_info.csv", info_start="按照比例随机抽样25w进行数据集测试", target_model="MLP", metric="val_accuracy")

# 25w随机抽样Efficient模型feature1\2\12表现
def draw_25w_sample_box(file, info_start, feature_ver, model_name):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import ast

    feature_type = feature_ver.split("/")[1]

    df_all = pd.read_csv(file)
    df = pd.DataFrame(columns=df_all.columns)
    for i in range(df_all.shape[0]):
        info = df_all.iloc[i]["info"]
        feature_version = df_all.iloc[i]["feature_version"]
        model_n = df_all.iloc[i]["model_name"]
        if info.startswith(info_start) and feature_version == feature_ver and model_n == model_name:
            df.loc[len(df)] = df_all.iloc[i]

    # num = [int(x.split('，')[0].split('=')[-1]) for x in info]
    num_set = df["all_count"].values
    accuracy = df["val_accuracy"].values
    detection_rate = df['detect_rate'].values
    false_positive = df['false_positive'].values
    precision_weighted = df['precision_weighted'].values
    recall_weighted = df['recall_weighted'].values
    f1_weighted = df['f1_weighted'].values
    accuracy_score = np.array([num_set, accuracy]).T
    detection_rate_score = np.array([num_set, detection_rate]).T
    false_positive_score = np.array([num_set, false_positive]).T
    precision_weighted_score = np.array([num_set, precision_weighted]).T
    recall_weighted_score = np.array([num_set, recall_weighted]).T
    f1_weighted_score = np.array([num_set, f1_weighted]).T

    accuracy_score_list = [[], []]
    for i in range(len(accuracy_score)):
        num = accuracy_score[i][0]
        accuracy = ast.literal_eval(accuracy_score[i][1])
        accuracy_score_list[0].append(num)
        accuracy_score_list[1].append(np.average(accuracy))
    detection_rate_list = [[], []]
    for i in range(len(detection_rate_score)):
        num = detection_rate_score[i][0]
        detection_rate = ast.literal_eval(detection_rate_score[i][1])
        detection_rate_list[0].append(num)
        detection_rate_list[1].append(np.average(detection_rate))
    false_positive_list = [[], []]
    for i in range(len(false_positive_score)):
        num = false_positive_score[i][0]
        false_positive = ast.literal_eval(false_positive_score[i][1])
        false_positive_list[0].append(num)
        false_positive_list[1].append(np.average(false_positive))
    precision_weighted_list = [[], []]
    for i in range(len(precision_weighted_score)):
        num = precision_weighted_score[i][0]
        precision_weighted = ast.literal_eval(precision_weighted_score[i][1])
        precision_weighted_list[0].append(num)
        precision_weighted_list[1].append(np.average(precision_weighted))
    recall_weighted_list = [[], []]
    for i in range(len(recall_weighted_score)):
        num = recall_weighted_score[i][0]
        recall_weighted = ast.literal_eval(recall_weighted_score[i][1])
        recall_weighted_list[0].append(num)
        recall_weighted_list[1].append(np.average(recall_weighted))
    f1_weighted_list = [[], []]
    for i in range(len(f1_weighted_score)):
        num = f1_weighted_score[i][0]
        f1_weighted = ast.literal_eval(f1_weighted_score[i][1])
        f1_weighted_list[0].append(num)
        f1_weighted_list[1].append(np.average(f1_weighted))

    # Accuracy箱线图
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams['font.size'] = 12
    accuracy_data = []
    f1_data = []
    labels = []
    max_all = 0
    max_raw = 0
    for i in range(len(accuracy_score)):
        num = accuracy_score[i][0]
        labels.append(num)
        accuracy = ast.literal_eval(accuracy_score[i][1])
        accuracy_data.append(accuracy)  # todo?
        f1 = ast.literal_eval(f1_weighted_score[i][1])
        f1_data.append(f1)
    bplot1 = ax.boxplot(accuracy_data,
                        vert=True,  # 箱体垂直对齐
                        patch_artist=True,  # 使用颜色填充
                        labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink',
              'pink', 'pink', 'pink']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    bplot2 = ax.boxplot(f1_data,
                        vert=True,  # 箱体垂直对齐
                        patch_artist=True,  # 使用颜色填充
                        labels=labels)  # 设置x轴刻度标签
    # # 为箱线图设置颜色
    colors = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green',
              'green', 'green', 'green', 'green', 'green']
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Sample Box Plot: Feature {}'.format(feature_type))
    ax.set_xlabel(model_name)
    ax.set_ylabel('Metrics')
    # 添加水平网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # 隐藏图形的上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置刻度向内显示
    ax.tick_params(axis='both', which='major', direction='in')
    text_y = max(max([item for data in accuracy_data for item in data]),
                 max([item for data in f1_data for item in data]))
    plt.text(1, text_y, "accuracy", color="pink", ha="center", fontsize=12)
    plt.text(2, text_y, "f1 score", color="green", ha="center", fontsize=12)
    # 调整布局以避免标签重叠
    plt.tight_layout()
    # 展示图形
    # 以高分辨率保存图形为PDF文件
    plt.savefig(os.path.join("../output_us", "箱线图小批量抽样{}_特征{}.png".format(model_name, feature_type)), dpi=300)
    plt.show()
    plt.close()  # 关闭图表，确保后续绘图不影响


# draw_25w_sample_box("../output_us/output_process_info.csv", info_start="按照比例随机抽样25w进行数据集测试", feature_ver="feature7/raw/single_sample", model_name="Autoencoder")

def draw_traing_accuracy_detect_rate_false_positive(file, info_start, feature_ver):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import ast

    df_all = pd.read_csv(file)
    df = pd.DataFrame(columns=df_all.columns)
    for i in range(df_all.shape[0]):
        info = df_all.iloc[i]["info"]
        feature_version = df_all.iloc[i]["feature_version"]
        if info.startswith(info_start) and feature_version == feature_ver:
            df.loc[len(df)] = df_all.iloc[i]

    num_set = df["model_name"].values
    accuracy = df["val_accuracy"].values
    detection_rate = df['detect_rate'].values
    false_positive = df['false_positive'].values
    precision_weighted = df['precision_weighted'].values
    recall_weighted = df['recall_weighted'].values
    f1_weighted = df['f1_weighted'].values
    # accuracy_score = np.array([num_set, accuracy]).T
    # detection_rate_score = np.array([num_set, detection_rate]).T
    # false_positive_score = np.array([num_set, false_positive]).T
    # precision_weighted_score = np.array([num_set, precision_weighted]).T
    # recall_weighted_score = np.array([num_set, recall_weighted]).T
    # f1_weighted_score = np.array([num_set, f1_weighted]).T

    plt.figure(figsize=(10, 6))
    k_fold = [i for i in range(len(ast.literal_eval(accuracy[0])))]
    for i in range(len(num_set)):
        plt.plot(k_fold, ast.literal_eval(accuracy[i]), label=num_set[i], linestyle='--')
    plt.xlabel("K", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Feature 1 Model Train Accuracy on KFold Validation", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "feature1模型训练k层交叉验证Accuracy.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(len(num_set)):
        plt.plot(k_fold, ast.literal_eval(detection_rate[i]), label=num_set[i], linestyle='--')
    plt.xlabel("K", fontsize=12)
    plt.ylabel("Detect Rate", fontsize=12)
    plt.title("Feature 1 Model Train Detect Rate on KFold Validation", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "feature1模型训练k层交叉验证DetectRate.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(len(num_set)):
        plt.plot(k_fold, ast.literal_eval(false_positive[i]), label=num_set[i], linestyle='--')
    plt.xlabel("K", fontsize=12)
    plt.ylabel("False Positive", fontsize=12)
    plt.title("Feature 1 Model Train False Positive on KFold Validation", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "feature1模型训练k层交叉验证FalsePositive.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(len(num_set)):
        plt.plot(k_fold, ast.literal_eval(precision_weighted[i]), label=num_set[i], linestyle='--')
    plt.xlabel("K", fontsize=12)
    plt.ylabel("Precision Weighted", fontsize=12)
    plt.title("Feature 1 Model Train Precision Weighted on KFold Validation", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "feature1模型训练k层交叉验证PrecisionWeighted.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(len(num_set)):
        plt.plot(k_fold, ast.literal_eval(recall_weighted[i]), label=num_set[i], linestyle='--')
    plt.xlabel("K", fontsize=12)
    plt.ylabel("Recall Weighted", fontsize=12)
    plt.title("Feature 1 Model Train Recall Weighted on KFold Validation", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "feature1模型训练k层交叉验证RecallWeighted.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(len(num_set)):
        plt.plot(k_fold, ast.literal_eval(f1_weighted[i]), label=num_set[i], linestyle='--')
    plt.xlabel("K", fontsize=12)
    plt.ylabel("F1 Weighted", fontsize=12)
    plt.title("Feature 1 Model Train F1 Weighted on KFold Validation", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "feature1模型训练k层交叉验证F1Weighted.png"))
    plt.show()


# draw_traing_accuracy_detect_rate_false_positive("../output_us/output_process_info.csv", info_start="25w特征1+2数据集训练", feature_ver="feature7/raw/single_sample")

def draw_traing_compare_plot(file, info_start):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import ast
    feature_ver_list = ["feature7/raw/single_sample", "feature7/all/single_sample"]
    csv = pd.read_csv(file)
    df_all = pd.DataFrame(columns=csv.columns)
    df_raw = pd.DataFrame(columns=csv.columns)
    for i in range(csv.shape[0]):
        info = csv.iloc[i]["info"]
        feature_version = csv.iloc[i]["feature_version"]
        if info.startswith(info_start) and feature_version == feature_ver_list[0]:
            df_raw.loc[len(df_raw)] = csv.iloc[i]
        if info.startswith(info_start) and feature_version == feature_ver_list[1]:
            df_all.loc[len(df_all)] = csv.iloc[i]

    model_name_set = df_all["model_name"].values
    model_name_map = {}
    for i in range(len(model_name_set)):
        model_name_map[i] = model_name_set
    accuracy_all = df_all["val_accuracy"].values
    detection_rate_all = df_all['detect_rate'].values
    false_positive_all = df_all['false_positive'].values
    precision_weighted_all = df_all['precision_weighted'].values
    recall_weighted_all = df_all['recall_weighted'].values
    f1_weighted_all = df_all['f1_weighted'].values

    model_name_set = df_raw["model_name"].values
    model_name_map = {}
    for i in range(len(model_name_set)):
        model_name_map[i] = model_name_set[i]
    accuracy_raw = df_raw["val_accuracy"].values
    detection_rate_raw = df_raw['detect_rate'].values
    false_positive_raw = df_raw['false_positive'].values
    precision_weighted_raw = df_raw['precision_weighted'].values
    recall_weighted_raw = df_raw['recall_weighted'].values
    f1_weighted_raw = df_raw['f1_weighted'].values

    # accuracy_score = np.array([num_set, accuracy_all]).T
    # detection_rate_score = np.array([num_set, detection_rate_all]).T
    # false_positive_score = np.array([num_set, false_positive_all]).T
    # precision_weighted_score = np.array([num_set, precision_weighted_all]).T
    # recall_weighted_score = np.array([num_set, recall_weighted_all]).T
    # f1_weighted_score = np.array([num_set, f1_weighted_all]).T

    plt.figure(figsize=(14, 6))
    k_fold = [i for i in range(len(ast.literal_eval(accuracy_all[0])))]
    for i in range(len(model_name_map)):
        score_list = ast.literal_eval(accuracy_all[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1 2" if i == 0 else "", marker='o', color='pink')
        score_list = ast.literal_eval(accuracy_raw[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1" if i == 0 else "", marker='x', color='green')
    plt.xlabel("model_name", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(list(model_name_map.keys()), list(model_name_map.values()))
    plt.title("Feature 1 2 Model Train Accuracy", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "模型训练对比Accuracy.png"))
    plt.show()

    plt.figure(figsize=(14, 6))
    k_fold = [i for i in range(len(ast.literal_eval(detection_rate_all[0])))]
    for i in range(len(model_name_map)):
        score_list = ast.literal_eval(detection_rate_all[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1 2" if i == 0 else "", marker='o', color='pink')
        score_list = ast.literal_eval(detection_rate_raw[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1" if i == 0 else "", marker='x', color='green')
    plt.xlabel("model_name", fontsize=12)
    plt.ylabel("Detect Rate", fontsize=12)
    plt.xticks(list(model_name_map.keys()), list(model_name_map.values()))
    plt.title("Feature 1 2 Model Train Detect Rate", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "模型训练对比DetectRate.png"))
    plt.show()

    plt.figure(figsize=(14, 6))
    k_fold = [i for i in range(len(ast.literal_eval(precision_weighted_all[0])))]
    for i in range(len(model_name_map)):
        score_list = ast.literal_eval(precision_weighted_all[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1 2" if i == 0 else "", marker='o', color='pink')
        score_list = ast.literal_eval(precision_weighted_raw[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1" if i == 0 else "", marker='x', color='green')
    plt.xlabel("model_name", fontsize=12)
    plt.ylabel("Precision Weighted", fontsize=12)
    plt.xticks(list(model_name_map.keys()), list(model_name_map.values()))
    plt.title("Feature 1 2 Model Train Precision Weighted", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "模型训练对比PrecisionWeighted.png"))
    plt.show()

    plt.figure(figsize=(14, 6))
    k_fold = [i for i in range(len(ast.literal_eval(false_positive_all[0])))]
    for i in range(len(model_name_map)):
        score_list = ast.literal_eval(false_positive_all[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1 2" if i == 0 else "", marker='o', color='pink')
        score_list = ast.literal_eval(false_positive_raw[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1" if i == 0 else "", marker='x', color='green')
    plt.xlabel("model_name", fontsize=12)
    plt.ylabel("False Positive", fontsize=12)
    plt.xticks(list(model_name_map.keys()), list(model_name_map.values()))
    plt.title("Feature 1 2 Model Train False Positive", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "模型训练对比FalsePositive.png"))
    plt.show()

    plt.figure(figsize=(14, 6))
    k_fold = [i for i in range(len(ast.literal_eval(recall_weighted_all[0])))]
    for i in range(len(model_name_map)):
        score_list = ast.literal_eval(recall_weighted_all[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1 2" if i == 0 else "", marker='o', color='pink')
        score_list = ast.literal_eval(recall_weighted_raw[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1" if i == 0 else "", marker='x', color='green')
    plt.xlabel("model_name", fontsize=12)
    plt.ylabel("Recall Weighted", fontsize=12)
    plt.xticks(list(model_name_map.keys()), list(model_name_map.values()))
    plt.title("Feature 1 2 Model Train Recall Weighted", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "模型训练对比RecallWeighted.png"))
    plt.show()

    plt.figure(figsize=(14, 6))
    k_fold = [i for i in range(len(ast.literal_eval(f1_weighted_all[0])))]
    for i in range(len(model_name_map)):
        score_list = ast.literal_eval(f1_weighted_all[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1 2" if i == 0 else "", marker='o', color='pink')
        score_list = ast.literal_eval(f1_weighted_raw[i])
        plt.scatter([i] * len(score_list), score_list, label="feature 1" if i == 0 else "", marker='x', color='green')
    plt.xlabel("model_name", fontsize=12)
    plt.ylabel("F1 Weighted", fontsize=12)
    plt.xticks(list(model_name_map.keys()), list(model_name_map.values()))
    plt.title("Feature 1 2 Model Train F1 Weighted", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("../output_us", "模型训练对比F1Weighted.png"))
    plt.show()


# draw_traing_compare_plot("../output_us/output_all_dataset_model_score.csv", info_start="25w特征1+2数据集训练")

# 25w训练箱线图，结果doc文件，指定metric
def draw_training_box_plot_metric(file, metric):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import ast

    csv = pd.read_csv(file, encoding='utf-8')
    raw_data = csv[csv["feature_version"] == "raw"]
    all_data = csv[csv["feature_version"] == "all"]
    labels = raw_data["model"].unique()
    raw_data_list = [ast.literal_eval(data) for data in raw_data[metric]]
    all_data_list = [ast.literal_eval(data) for data in all_data[metric]]
    max_raw = max([max(d) for d in raw_data_list])
    max_all = max([max(d) for d in all_data_list])
    # Accuracy箱线图
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams['font.size'] = 12
    bplot1 = ax.boxplot(all_data_list,
                        vert=True,  # 箱体垂直对齐
                        patch_artist=True,  # 使用颜色填充
                        labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    bplot2 = ax.boxplot(raw_data_list,
                        vert=True,  # 箱体垂直对齐
                        patch_artist=True,  # 使用颜色填充
                        labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green']
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Rectangular Box Plot ({})'.format(metric))
    ax.set_xlabel('Model Name')
    ax.set_ylabel(metric)
    # 添加水平网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # 隐藏图形的上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置刻度向内显示
    ax.tick_params(axis='both', which='major', direction='in')
    plt.text(1, max(max_all, max_raw) + 0.004, "feature1", color="green", ha="center", fontsize=12)
    plt.text(2, max(max_all, max_raw) + 0.004, "feature1+2", color="pink", ha="center", fontsize=12)
    # 调整布局以避免标签重叠
    plt.tight_layout()
    # 展示图形
    # 以高分辨率保存图形为PDF文件
    plt.savefig(os.path.join("../output_us", "箱线图模型训练对比{}.png".format(metric)))
    plt.show()
    plt.close()  # 关闭图表，确保后续绘图不影响

# draw_training_box_plot_metric("../result/feature7/doc/实验2-特征有效性验证.csv", "accuracy")

def draw_training_box_plot(file, info_start):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import ast
    category_list = ["LR", "RFC", "KNN", "MLP", "Efficient", "Autoencoder"]
    feature_ver_list = ["feature7/raw/single_sample", "feature7/all/single_sample"]
    csv = pd.read_csv(file, encoding='utf-8')
    df_all = pd.DataFrame(columns=csv.columns)
    df_raw = pd.DataFrame(columns=csv.columns)
    for i in range(csv.shape[0]):
        info = csv.iloc[i]["info"]
        feature_version = csv.iloc[i]["feature_version"]
        if info.startswith(info_start) and feature_version == feature_ver_list[0]:
            df_raw.loc[len(df_raw)] = csv.iloc[i]
        if info.startswith(info_start) and feature_version == feature_ver_list[1]:
            df_all.loc[len(df_all)] = csv.iloc[i]

    accuracy_all = df_all["val_accuracy"].values
    detection_rate_all = df_all['detect_rate'].values
    false_positive_all = df_all['false_positive'].values
    precision_weighted_all = df_all['precision_weighted'].values
    recall_weighted_all = df_all['recall_weighted'].values
    f1_weighted_all = df_all['f1_weighted'].values

    model_name_set = df_raw["model_name"].values
    model_name_map = {}
    for i in range(len(model_name_set)):
        model_name_map[i] = model_name_set[i]
    accuracy_raw = df_raw["val_accuracy"].values
    detection_rate_raw = df_raw['detect_rate'].values
    false_positive_raw = df_raw['false_positive'].values
    precision_weighted_raw = df_raw['precision_weighted'].values
    recall_weighted_raw = df_raw['recall_weighted'].values
    f1_weighted_raw = df_raw['f1_weighted'].values

    # Accuracy箱线图
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams['font.size'] = 12
    all_data = []
    raw_data = []
    labels = []
    max_all = 0
    max_raw = 0
    for i in range(len(model_name_map)):
        if model_name_map[i] not in category_list:
            continue
        score_list = ast.literal_eval(accuracy_all[i])
        all_data.append(score_list)
        max_all = max(max_all, max(score_list))
        labels.append(model_name_map[i])
        score_list = ast.literal_eval(accuracy_raw[i])
        raw_data.append(score_list)
        max_raw = max(max_raw, max(score_list))
    bplot1 = ax.boxplot(all_data,
                        vert=True,  # 箱体垂直对齐
                        patch_artist=True,  # 使用颜色填充
                        labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    bplot2 = ax.boxplot(raw_data,
                        vert=True,  # 箱体垂直对齐
                        patch_artist=True,  # 使用颜色填充
                        labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green']
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Rectangular Box Plot (Accuracy)')
    ax.set_xlabel('Model Name')
    ax.set_ylabel('Accuracy')
    # 添加水平网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # 隐藏图形的上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置刻度向内显示
    ax.tick_params(axis='both', which='major', direction='in')
    plt.text(1, max(max_all, max_raw) + 0.004, "feature1", color="green", ha="center", fontsize=12)
    plt.text(2, max(max_all, max_raw) + 0.004, "feature1+2", color="pink", ha="center", fontsize=12)
    # 调整布局以避免标签重叠
    plt.tight_layout()
    # 展示图形
    # 以高分辨率保存图形为PDF文件
    plt.savefig(os.path.join("../output_us", "箱线图模型训练对比Accuracy.png"))
    plt.show()
    plt.close()  # 关闭图表，确保后续绘图不影响

    # DetectRate箱线图
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams['font.size'] = 12
    all_data = []
    raw_data = []
    max_all = 0
    max_raw = 0
    labels = []
    for i in range(len(model_name_map)):
        if model_name_map[i] not in category_list:
            continue
        score_list = ast.literal_eval(detection_rate_all[i])
        all_data.append(score_list)
        max_all = max(max_all, max(score_list))
        labels.append(model_name_map[i])
        score_list = ast.literal_eval(detection_rate_raw[i])
        raw_data.append(score_list)
        max_raw = max(max_raw, max(score_list))
    bplot = ax.boxplot(all_data,
                       vert=True,  # 箱体垂直对齐
                       patch_artist=True,  # 使用颜色填充
                       labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    bplot = ax.boxplot(raw_data,
                       vert=True,  # 箱体垂直对齐
                       patch_artist=True,  # 使用颜色填充
                       labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Rectangular Box Plot (Detect Rate)')
    ax.set_xlabel('Model Name')
    ax.set_ylabel('Detect Rate')
    # 添加水平网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # 隐藏图形的上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置刻度向内显示
    ax.tick_params(axis='both', which='major', direction='in')
    plt.text(1, max(max_all, max_raw) + 0.004, "feature1", color="green", ha="center", fontsize=12)
    plt.text(2, max(max_all, max_raw) + 0.004, "feature1+2", color="pink", ha="center", fontsize=12)
    # 调整布局以避免标签重叠
    plt.tight_layout()
    # 展示图形
    # 以高分辨率保存图形为PDF文件
    plt.savefig(os.path.join("../output_us", "箱线图模型训练对比DetectRate.png"))
    plt.show()

    plt.close()  # 关闭图表，确保后续绘图不影响

    # FalsePositive箱线图
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams['font.size'] = 12
    all_data = []
    raw_data = []
    max_all = 0
    max_raw = 0
    labels = []
    for i in range(len(model_name_map)):
        if model_name_map[i] not in category_list:
            continue
        score_list = ast.literal_eval(false_positive_all[i])
        all_data.append(score_list)
        max_all = max(max_all, max(score_list))
        labels.append(model_name_map[i])
        score_list = ast.literal_eval(false_positive_raw[i])
        raw_data.append(score_list)
        max_raw = max(max_raw, max(score_list))
    bplot = ax.boxplot(all_data,
                       vert=True,  # 箱体垂直对齐
                       patch_artist=True,  # 使用颜色填充
                       labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    bplot = ax.boxplot(raw_data,
                       vert=True,  # 箱体垂直对齐
                       patch_artist=True,  # 使用颜色填充
                       labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Rectangular Box Plot (False Positive)')
    ax.set_xlabel('Model Name')
    ax.set_ylabel('False Positive')
    # 添加水平网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # 隐藏图形的上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置刻度向内显示
    ax.tick_params(axis='both', which='major', direction='in')
    plt.text(1, max(max_all, max_raw) + 0.004, "feature1", color="green", ha="center", fontsize=12)
    plt.text(2, max(max_all, max_raw) + 0.004, "feature1+2", color="pink", ha="center", fontsize=12)
    # 调整布局以避免标签重叠
    plt.tight_layout()
    # 展示图形
    # 以高分辨率保存图形为PDF文件
    plt.savefig(os.path.join("../output_us", "箱线图模型训练对比FalsePositive.png"))
    plt.show()
    plt.close()

    # Precision箱线图
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams['font.size'] = 12
    all_data = []
    raw_data = []
    max_all = 0
    max_raw = 0
    labels = []
    for i in range(len(model_name_map)):
        if model_name_map[i] not in category_list:
            continue
        score_list = ast.literal_eval(precision_weighted_all[i])
        all_data.append(score_list)
        max_all = max(max_all, max(score_list))
        labels.append(model_name_map[i])
        score_list = ast.literal_eval(precision_weighted_raw[i])
        raw_data.append(score_list)
        max_raw = max(max_raw, max(score_list))
    bplot = ax.boxplot(all_data,
                       vert=True,  # 箱体垂直对齐
                       patch_artist=True,  # 使用颜色填充
                       labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    bplot = ax.boxplot(raw_data,
                       vert=True,  # 箱体垂直对齐
                       patch_artist=True,  # 使用颜色填充
                       labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Rectangular Box Plot (Precision)')
    ax.set_xlabel('Model Name')
    ax.set_ylabel('Precision')
    # 添加水平网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # 隐藏图形的上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置刻度向内显示
    ax.tick_params(axis='both', which='major', direction='in')
    plt.text(1, max(max_all, max_raw) + 0.004, "feature1", color="green", ha="center", fontsize=12)
    plt.text(2, max(max_all, max_raw) + 0.004, "feature1+2", color="pink", ha="center", fontsize=12)
    # 调整布局以避免标签重叠
    plt.tight_layout()
    # 展示图形
    # 以高分辨率保存图形为PDF文件
    plt.savefig(os.path.join("../output_us", "箱线图模型训练对比Precision.png"))
    plt.show()
    plt.close()

    # Recall箱线图
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams['font.size'] = 12
    all_data = []
    raw_data = []
    max_all = 0
    max_raw = 0
    labels = []
    for i in range(len(model_name_map)):
        if model_name_map[i] not in category_list:
            continue
        score_list = ast.literal_eval(recall_weighted_all[i])
        all_data.append(score_list)
        max_all = max(max_all, max(score_list))
        labels.append(model_name_map[i])
        score_list = ast.literal_eval(recall_weighted_raw[i])
        raw_data.append(score_list)
        max_raw = max(max_raw, max(score_list))
    bplot = ax.boxplot(all_data,
                       vert=True,  # 箱体垂直对齐
                       patch_artist=True,  # 使用颜色填充
                       labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    bplot = ax.boxplot(raw_data,
                       vert=True,  # 箱体垂直对齐
                       patch_artist=True,  # 使用颜色填充
                       labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Rectangular Box Plot (Recall)')
    ax.set_xlabel('Model Name')
    ax.set_ylabel('Recall')
    # 添加水平网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # 隐藏图形的上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置刻度向内显示
    ax.tick_params(axis='both', which='major', direction='in')
    plt.text(1, max(max_all, max_raw) + 0.004, "feature1", color="green", ha="center", fontsize=12)
    plt.text(2, max(max_all, max_raw) + 0.004, "feature1+2", color="pink", ha="center", fontsize=12)
    # 调整布局以避免标签重叠
    plt.tight_layout()
    # 展示图形
    # 以高分辨率保存图形为PDF文件
    plt.savefig(os.path.join("../output_us", "箱线图模型训练对比Recall.png"))
    plt.show()
    plt.close()

    # F1箱线图
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams['font.size'] = 12
    all_data = []
    raw_data = []
    max_all = 0
    max_raw = 0
    labels = []
    for i in range(len(model_name_map)):
        if model_name_map[i] not in category_list:
            continue
        score_list = ast.literal_eval(f1_weighted_all[i])
        all_data.append(score_list)
        max_all = max(max_all, max(score_list))
        labels.append(model_name_map[i])
        score_list = ast.literal_eval(f1_weighted_raw[i])
        raw_data.append(score_list)
        max_raw = max(max_raw, max(score_list))
    bplot = ax.boxplot(all_data,
                       vert=True,  # 箱体垂直对齐
                       patch_artist=True,  # 使用颜色填充
                       labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'pink']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    bplot = ax.boxplot(raw_data,
                       vert=True,  # 箱体垂直对齐
                       patch_artist=True,  # 使用颜色填充
                       labels=labels)  # 设置x轴刻度标签
    # 为箱线图设置颜色
    colors = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Rectangular Box Plot (F1 Score)')
    ax.set_xlabel('Model Name')
    ax.set_ylabel('F1 Score')
    # 添加水平网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # 隐藏图形的上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置刻度向内显示
    ax.tick_params(axis='both', which='major', direction='in')
    plt.text(1, max(max_all, max_raw) + 0.004, "feature1", color="green", ha="center", fontsize=12)
    plt.text(2, max(max_all, max_raw) + 0.004, "feature1+2", color="pink", ha="center", fontsize=12)
    # 调整布局以避免标签重叠
    plt.tight_layout()
    # 展示图形
    # 以高分辨率保存图形为PDF文件
    plt.savefig(os.path.join("../output_us", "箱线图模型训练对比F1Score.png"))
    plt.show()
    plt.close()


# draw_training_box_plot("../result/feature7/RFC/single/实验二-特征有效性验证/特征有效性验证-avg.csv", info_start="全部25w进行数据集测试，")

def draw_auc_simple(y_test, y_pred_prob, category_list, output_dir, model_name, data_size, feature_ver):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import os
    if feature_ver == "all":
        feature_ver = "1+2"
    elif feature_ver == "raw":
        feature_ver = "1"
    elif feature_ver == "second":
        feature_ver = "2"
    # 将标签转换为二进制格式（每个类别为一个二元向量）
    y_test_bin = label_binarize(y_test, classes=[i for i in range(len(category_list))])  # 假设有5个类别

    # 获取类别数
    n_classes = y_test_bin.shape[1]

    # 绘图
    plt.figure(figsize=(10, 8))

    roc_auc_list = []
    # 对每个类别绘制ROC曲线
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_list.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f'Class {category_list[i]} (AUC = {roc_auc:.4f})')

    # 绘制对角线（随机分类的AUC）
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)

    # 设置图表的标签和标题
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve with Feature {}'.format(feature_ver))
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, '{}_{}_auc_1.png'.format(model_name, data_size)), bbox_inches='tight')
    # 显示图表
    # plt.show()
    auc_info = [":".join([category_list[i], str(round(roc_auc_list[i], 4))]) for i in range(len(category_list))]
    return ",".join(auc_info)


def draw_auc(y_test_df, pred1_df, category_list, output_dir, model_name, data_size, feature_ver):
    import numpy as np
    import pandas as pd
    from numpy import interp
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    import os

    if feature_ver == "all":
        feature_ver = "1+2"
    elif feature_ver == "raw":
        feature_ver = "1"
    elif feature_ver == "second":
        feature_ver = "2"
    # if model_name in ["RFC", "LR", ]:
    y_test = pd.get_dummies(y_test_df).astype(int).values
    pred1 = pred1_df

    class_num = y_test.shape[1]
    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(class_num):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred1[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred1.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= class_num

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['grey', 'yellow', 'green', 'red', 'pink', 'blue', 'black', 'maroon', 'purple', 'orange'])
    for i, color in zip(range(class_num), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(category_list[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} Receiver operating characteristic to multi-class on Feature {}'.format(model_name, feature_ver))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(os.path.join(output_dir, '{}_{}_auc_1.png'.format(model_name, data_size)), bbox_inches='tight')
    # plt.show()

    # zoom in the plot
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['grey', 'yellow', 'green', 'red', 'pink', 'blue', 'black', 'maroon', 'purple', 'orange'])
    for i, color in zip(range(class_num), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(category_list[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} Receiver operating characteristic to multi-class on Feature'.format(model_name, feature_ver))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(os.path.join(output_dir, '{}_{}_auc_4.png'.format(model_name, data_size)), bbox_inches='tight')
    # plt.show()

    auc_info = [":".join([category_list[i], str(round(roc_auc[i], 4))]) for i in range(len(category_list))]
    return ",".join(auc_info)


def calculate_detection_rate(pred, y, normal_index):
    import numpy as np
    pred_binary = np.not_equal(pred, normal_index)
    y_binary = np.not_equal(y, normal_index)
    tp = pred_binary & y_binary
    t = y_binary
    dr = np.sum(tp) / np.sum(t)
    print("Detection rate: {:.3f}".format(dr))
    return dr


def calculate_false_positive_rate(pred, y, normal_index):
    import numpy as np
    pred_binary = np.not_equal(pred, normal_index)
    y_binary = np.not_equal(y, normal_index)
    fp = pred_binary & ~y_binary
    n = ~y_binary
    fpr = np.sum(fp) / np.sum(n)
    return fpr


def model_train_process_plot(train_history, plot_output_dir, model_name, data_count, best_epoch, model_no):
    import matplotlib.pyplot as plt
    import os

    # 绘制损失曲线
    plt.figure()  # 创建一个新的画布
    plt.xticks(range(0, best_epoch, 1))  # 从最小值到最大值，步长为1
    plt.plot(train_history.history['loss'][:best_epoch], label='Training Loss')
    plt.plot(train_history.history['val_loss'][:best_epoch], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(
        os.path.join(plot_output_dir, '{}_{}_{}_epoch{}_loss.png'.format(model_name, data_count, model_no, best_epoch)))
    # plt.show()

    # 绘制准确率曲线
    plt.figure()  # 创建一个新的画布
    plt.plot(train_history.history['accuracy'][:best_epoch], label='Training Accuracy')
    plt.plot(train_history.history['val_accuracy'][:best_epoch], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(plot_output_dir,
                             '{}_{}_{}_epoch{}_accuracy.png'.format(model_name, data_count, model_no, best_epoch)))
    # plt.show()

    # 绘制总体训练曲线
    plt.figure()  # 创建一个新的画布
    plt.plot(train_history.history['loss'][:best_epoch], label='Training Loss')
    plt.plot(train_history.history['val_loss'][:best_epoch], label='Validation Loss')
    plt.plot(train_history.history['accuracy'][:best_epoch], label='Training Accuracy')
    plt.plot(train_history.history['val_accuracy'][:best_epoch], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title('Training Process')
    plt.savefig(
        os.path.join(plot_output_dir,
                     '{}_{}_{}_epoch{}_training_process.png'.format(model_name, data_count, model_no, best_epoch)))
    # plt.show()

# pred = [0,1,1,3,5,4,8,6,6,6,7,9]
# y = [0,1,6,3,5,4,8,2,6,6,7,9]
# calculate_detection_rate(pred, y, 6)
# calculate_false_positive_rate(pred, y, normal_index=6)
