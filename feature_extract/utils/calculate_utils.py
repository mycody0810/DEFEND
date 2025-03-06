
def convert_best2avg_from_csv(best_file, avg_file):
    import pandas as pd
    import numpy as np
    import ast

    best_df = pd.read_csv(best_file, encoding='gbk')
    avg_df = pd.read_csv(avg_file, encoding='utf-8')
    avg_df_add = pd.DataFrame(columns=avg_df.columns)
    best_columns = best_df.columns
    avg_columns = avg_df.columns
    for i in range(best_df.shape[0]):
        if best_df.loc[i, "all_count"] != -1:
            continue
        best_data = best_df.iloc[i]
        for column in best_columns:
            if column.startswith("best_"):
                metric = "_".join(column.split("_")[1:])
                avg_column = "avg_" + metric
                if avg_column not in avg_columns:
                    continue
                metric = "val_accuracy" if metric == "accuracy" else metric

                best_data[column] = round(np.mean(ast.literal_eval(best_data[metric])), 5)
        new_df = pd.DataFrame(best_data.to_frame().T.values, columns=avg_df.columns)
        avg_df_add = pd.concat([avg_df_add, new_df], axis=0, ignore_index=True)
    avg_df_add.to_csv(avg_file, encoding='gbk', index=False, mode='a', header=True)
    return

# Parameter tuning experiment 1-MLP
def experiment_1_MLP(param_file, output_file):
    import pandas as pd
    raw_pd = pd.read_csv(param_file, encoding='utf-8')
    params = raw_pd.loc[:, "params"]
    hidden_layers = []
    hidden_neurons = []
    dropout_rate = []
    learning_rate = []
    for i in range(params.shape[0]):
        param = params.iloc[i]
        param_list = param.split(",")
        hidden_layers.append(param_list[0].split(":")[1])
        hidden_neurons.append(param_list[1].split(":")[1])
        dropout_rate.append(param_list[2].split(":")[1])
        learning_rate.append(param_list[3].split(":")[1])

    output_data = {
        "model_name": raw_pd.loc[:, "model_name"],
        "hidden_layers": hidden_layers,
        "hidden_neurons": hidden_neurons,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "accuracy": raw_pd.loc[:, "avg_accuracy"],
        "detect_rate": raw_pd.loc[:, "avg_detect_rate"],
        "false_positive": raw_pd.loc[:, "avg_false_positive"],
        "f1": raw_pd.loc[:, "avg_f1_weighted"]
    }
    output_df = pd.DataFrame.from_dict(output_data)
    output_df.to_csv(output_file, encoding='utf-8', index=False, mode='a', header=True)

# Parameter tuning experiment 1-Autoencoder
def experiment_1_Autoencoder(param_file, output_file):
    import pandas as pd
    raw_pd = pd.read_csv(param_file, encoding='utf-8')
    params = raw_pd.loc[:, "params"]
    encoder_hidden_dim = []
    classifier_hidden_dim = []
    encoder_dropout_rate = []
    learning_rate = []
    for i in range(params.shape[0]):
        param = params.iloc[i]
        param_list = param.split(",")
        encoder_hidden_dim.append(param_list[0].split(":")[1])
        classifier_hidden_dim.append(param_list[1].split(":")[1])
        encoder_dropout_rate.append(param_list[2].split(":")[1])
        learning_rate.append(param_list[3].split(":")[1])

    output_data = {
        "model_name": raw_pd.loc[:, "model_name"],
        "encoder_hidden_dim": encoder_hidden_dim,
        "classifier_hidden_dim": classifier_hidden_dim,
        "dropout_rate": encoder_dropout_rate,
        "learning_rate": learning_rate,
        "accuracy": raw_pd.loc[:, "avg_accuracy"],
        "detect_rate": raw_pd.loc[:, "avg_detect_rate"],
        "false_positive": raw_pd.loc[:, "avg_false_positive"],
        "f1": raw_pd.loc[:, "avg_f1_weighted"]
    }
    output_df = pd.DataFrame.from_dict(output_data)
    output_df.to_csv(output_file, encoding='utf-8', index=False, mode='a', header=True)


#Experiment 2- Verification of Feature Validity (Feature Validity Comparison Table, Box plot)
def experiment_2(avg_file, output_file):
    import pandas as pd
    import numpy as np
    import ast
    raw_pd = pd.read_csv(avg_file, encoding='utf-8')
    feature_version = raw_pd.loc[:, "feature_version"]
    feature_version_list = [feature.split("/")[1] for feature in feature_version]
    accuracy_map = {}
    detect_rate_map = {}
    false_positive_rate_map = {}
    f1_map = {}
    for i in range(raw_pd.shape[0]):
        model = raw_pd.loc[:, "model_name"][i]
        feature_version = raw_pd.loc[:, "feature_version"][i].split("/")[1]
        # all_count = raw_pd.loc[:, "all_count"][i]
        model_name = ':'.join([model, str(feature_version)])
        if model_name not in accuracy_map:
            accuracy_map[model_name] = []
        ast.literal_eval(raw_pd.loc[i, "val_accuracy"])
        accuracy_map[model_name].extend(ast.literal_eval(raw_pd.loc[i, "val_accuracy"]))
        if model_name not in detect_rate_map:
            detect_rate_map[model_name] = []
        detect_rate_map[model_name].extend(ast.literal_eval(raw_pd.loc[i, "detect_rate"]))
        if model_name not in false_positive_rate_map:
            false_positive_rate_map[model_name] = []
        false_positive_rate_map[model_name].extend(ast.literal_eval(raw_pd.loc[i, "false_positive"]))
        if model_name not in f1_map:
            f1_map[model_name] = []
        f1_map[model_name].extend(ast.literal_eval(raw_pd.loc[i, "f1_weighted"]))

    model_name_list = []
    feature_version_list = []
    avg_accuracy = []
    avg_detect_rate = []
    avg_false_positive_rate = []
    avg_f1 = []
    accuracy = []
    detect_rate = []
    false_positive_rate = []
    f1 = []
    # df = pd.DataFrame(columns=["model_name", "avg_accuracy", "avg_detect_rate", "avg_false_positive_rate", "avg_f1", "accuracy", "detect_rate", "false_positive_rate", "f1"])
    for model_name in accuracy_map:
        model, feature_version = model_name.split(":")
        model_name_list.append(model)
        feature_version_list.append(feature_version)
        avg_accuracy.append(round(np.mean(accuracy_map[model_name]), 5))
        avg_detect_rate.append(round(np.mean(detect_rate_map[model_name]), 5))
        avg_false_positive_rate.append(round(np.mean(false_positive_rate_map[model_name]), 5))
        avg_f1.append(round(np.mean(f1_map[model_name]), 5))
        accuracy.append(accuracy_map[model_name])
        detect_rate.append(detect_rate_map[model_name])
        false_positive_rate.append(false_positive_rate_map[model_name])
        f1.append(f1_map[model_name])
    output_data = {
        "model": model_name_list,
        "feature_version": feature_version_list,
        "avg_accuracy": avg_accuracy,
        "avg_detect_rate": avg_detect_rate,
        "avg_false_positive_rate": avg_false_positive_rate,
        "avg_f1": avg_f1,
        "accuracy": accuracy,
        "detect_rate": detect_rate,
        "false_positive_rate": false_positive_rate,
        "f1": f1
    }
    df = pd.DataFrame.from_dict(output_data)
    df.to_csv(output_file, encoding='utf-8', index=False, mode='a', header=True)

def experiment_3(avg_file, output_file):
    import pandas as pd
    import numpy as np
    import ast
    raw_pd = pd.read_csv(avg_file, encoding='utf-8')
    feature_version = raw_pd.loc[:, "feature_version"]
    feature_version_list = [feature.split("/")[1] for feature in feature_version]
    accuracy_map = {}
    detect_rate_map = {}
    false_positive_rate_map = {}
    f1_map = {}
    for i in range(raw_pd.shape[0]):
        model = raw_pd.loc[:, "model_name"][i]
        all_count = raw_pd.loc[:, "all_count"][i]
        model_name = ':'.join([model, str(all_count)])
        if model_name not in accuracy_map:
            accuracy_map[model_name] = []
        ast.literal_eval(raw_pd.loc[i, "val_accuracy"])
        accuracy_map[model_name].extend(ast.literal_eval(raw_pd.loc[i, "val_accuracy"]))
        if model_name not in detect_rate_map:
            detect_rate_map[model_name] = []
        detect_rate_map[model_name].extend(ast.literal_eval(raw_pd.loc[i, "detect_rate"]))
        if model_name not in false_positive_rate_map:
            false_positive_rate_map[model_name] = []
        false_positive_rate_map[model_name].extend(ast.literal_eval(raw_pd.loc[i, "false_positive"]))
        if model_name not in f1_map:
            f1_map[model_name] = []
        f1_map[model_name].extend(ast.literal_eval(raw_pd.loc[i, "f1_weighted"]))

    model_name_list = []
    count_list = []
    avg_accuracy = []
    avg_detect_rate = []
    avg_false_positive_rate = []
    avg_f1 = []
    accuracy = []
    detect_rate = []
    false_positive_rate = []
    f1 = []
    # df = pd.DataFrame(columns=["model_name", "avg_accuracy", "avg_detect_rate", "avg_false_positive_rate", "avg_f1", "accuracy", "detect_rate", "false_positive_rate", "f1"])
    for model_name in accuracy_map:
        model, all_count = model_name.split(":")
        model_name_list.append(model)
        count_list.append(all_count)
        avg_accuracy.append(round(np.mean(accuracy_map[model_name]), 5))
        avg_detect_rate.append(round(np.mean(detect_rate_map[model_name]), 5))
        avg_false_positive_rate.append(round(np.mean(false_positive_rate_map[model_name]), 5))
        avg_f1.append(round(np.mean(f1_map[model_name]), 5))
        accuracy.append(accuracy_map[model_name])
        detect_rate.append(detect_rate_map[model_name])
        false_positive_rate.append(false_positive_rate_map[model_name])
        f1.append(f1_map[model_name])
    output_data = {
        "model": model_name_list,
        "count": count_list,
        "avg_accuracy": avg_accuracy,
        "avg_detect_rate": avg_detect_rate,
        "avg_false_positive_rate": avg_false_positive_rate,
        "avg_f1": avg_f1,
        "accuracy": accuracy,
        "detect_rate": detect_rate,
        "false_positive_rate": false_positive_rate,
        "f1": f1
    }
    df = pd.DataFrame.from_dict(output_data)
    df.to_csv(output_file, encoding='utf-8', index=False, mode='a', header=True)

def experiment_2_hiplot_boxline(raw_file, output_file, metric):
    import pandas as pd
    import ast
    df = pd.read_csv(raw_file, encoding='utf-8')
    output_df = pd.DataFrame(columns=["Accuracy", "Feature Set", "Model Name"])
    for i in range(df.shape[0]):
        model = df.loc[:, "model"][i]
        feature_version = df.loc[:, "feature_version"][i]
        if feature_version == "second":
            continue
        feature = "All Features"
        if feature_version == "raw":
            feature = "Raw Features"
        accuracy_list = ast.literal_eval(df.loc[i, metric])
        for j in range(len(accuracy_list)):
            output_df.loc[output_df.shape[0]] = [accuracy_list[j], feature, model]
    output_df.to_csv(output_file, encoding='gbk', index=False, mode='w', header=True)