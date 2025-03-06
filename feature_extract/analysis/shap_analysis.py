#%%
# -*- coding: utf-8 -*-
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import os
import shap
import numpy as np
import itertools
import seaborn as sns
import tensorflow as tf
from feature_process.feature import class_percent_sample
def get_shap_explanation_values(explainer, shap_arrays, test_df, columns):
    expected_values = explainer.expected_value
    shap_explanation = shap.Explanation(values=shap_arrays, base_values=expected_values, data=test_df,
                                        feature_names=columns)
    return shap_explanation


def SHAP_process_tree_explainer_test(model, columns, background_df, test_df, labels, top_feature, plot_dir, model_name):
    shap.initjs()

    explainer = shap.TreeExplainer(model, background_df)
    shap_arrays = explainer.shap_values(test_df, check_additivity=False)
    shap_values = get_shap_explanation_values(explainer, shap_arrays, test_df, columns)

    class_cnt = len(labels)
    row = 2
    col = int((class_cnt + 1) / row)
    fig, axs = plt.subplots(row, col, figsize=(350, 140), constrained_layout=True)
    plt.title('SHAP Value beeswarm plot', pad=20)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i in range(class_cnt):
        c_row = i // col
        c_col = i % col
        plt.sca(axs[c_row, c_col])
        axs[c_row, c_col].set_title(labels[i])
        axs[c_row, c_col].tick_params(axis='y', labelsize=2, labelrotation=30)
        shap.plots.beeswarm(shap_values[:, :, i], max_display=15, show=False, plot_size=(35, 14))
    # plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'all_class_beeswarm_plot.png'), bbox_inches='tight')


def shap_plot(explainer, shap_values, test_df, data_index, col_index, col_name, class_index, class_name, labels, plot_dir):
    # Partial dependence plot: for all data, a specific feature, and a specific class
    # shap_part_dependence_single_plot(shap_explanation, col_index, col_name=col_name, class_index=class_index, class_name=class_name, plot_dir=plot_dir)
    # Bar plot for a single data point: for a specific data point, all features, and a specific class
    shap_data_bar_single_plot(explainer, shap_values, test_df, data_index, class_index, class_name, plot_dir)
    # Beeswarm plot: for all classes
    shap_beeswarm_all_plot(explainer, shap_values, labels, test_df, row=2, plot_dir=plot_dir)
    # Heatmap: for all classes
    shap_heatmap_all_plot(explainer, shap_values, labels, test_df, row=2, plot_dir=plot_dir)
    # Average bar plot: for all classes
    shap_avg_bar_all_plot(explainer, shap_values, labels, test_df, row=2, plot_dir=plot_dir)

def SHAP_process_tree_explainer(model, columns, background_df, test_df):
    shap.initjs()
    explainer = shap.TreeExplainer(model, background_df)
    shap_values = explainer.shap_values(test_df, check_additivity=False)
    shap_explanation = get_shap_explanation_values(explainer, shap_values, test_df, columns)
    return explainer, shap_values, shap_explanation

def draw_SHAP_process_tree_explainer_plot(explainer, shap_values, shap_explanation, test_df, columns, labels, plot_dir,
                                          model_name):
    # Draw partial SHAP plots
    shap_plot(explainer, shap_values, test_df, data_index=-1, col_index=0, col_name=columns[0], class_index=6,
              class_name=labels[6],
              labels=labels, plot_dir=plot_dir)

    # pred = model.predict(test_df)
    # conf_matrix = plot_confusion_matrix(test_df, pred)

    # Extract the average SHAP values (feature contributions) for each class across all features in this model
    raw_avg = {}
    for i in range(shap_explanation.shape[2]):
        raw_avg[labels[i]] = np.abs(shap_explanation.values[:, :, i]).mean(axis=0)

    importance_df = pd.DataFrame(raw_avg, columns=labels, index=columns)
    print(importance_df)
    importance_df['row_sum'] = importance_df.sum(axis=1)
    # Save SHAP values to CSV
    importance_df.sort_values(by='row_sum', ascending=False, inplace=False,
                              ignore_index=False).to_csv(
        os.path.join(plot_dir, 'all_class_feature_importance.csv'.format(model_name)))
    # Plot the total SHAP values for all features
    sorted_importance_df = importance_df.sort_values(by='row_sum', ascending=True, inplace=False,
                                                     ignore_index=False)
    sorted_importance_df = sorted_importance_df.drop(columns=["row_sum"])
    # Plot only the top 20 features
    top_feature_num = 30
    top_sorted_importance_df = sorted_importance_df.iloc[-top_feature_num:, :]
    # Plot all features
    # top_sorted_importance_df = sorted_importance_df.iloc[:, :]
    elements = top_sorted_importance_df.index
    shap_class_feature_plot(top_sorted_importance_df, elements,
                            os.path.join(plot_dir, 'all_class_feature_importance.png'.format(model_name)))

def SHAP_tree_waterfall_plot(shap_values, data_index, class_index, class_name, plot_dir):
    fig = plt.figure()
    plt.title('SHAP Value waterfall plot | Class {}'.format(class_name), fontsize=20, pad=20)
    shap.plots.waterfall(shap_values=shap_values[data_index, :, class_index], max_display=20, show=False)
    fig.savefig(os.path.join(plot_dir, "waterfall_plot_single.png"), bbox_inches='tight')

def SHAP_process_deep_explainer(model, columns, background_df, test_df):
    shap.initjs()
    explainer = shap.KernelExplainer(model, background_df)
    shap_values = explainer.shap_values(test_df, check_additivity=False)
    shap_explanation = get_shap_explanation_values(explainer, shap_values, test_df, columns)
    return explainer, shap_values, shap_explanation

def SHAP_process_linear_explainer(model, columns, background_df, test_df, labels, top_feature, plot_dir, model_name):
    top_feature_num = 20
    shap.initjs()

    explainer = shap.LinearExplainer(model, background_df)
    shap_arrays = explainer.shap_values(test_df)
    shap_values = get_shap_explanation_values(explainer, shap_arrays, test_df, columns)
    # Draw partial SHAP plots
    shap_plot(explainer, shap_values, test_df=test_df, data_index=-1, col_index=0, col_name=columns[0], class_index=6, class_name=labels[6],
              labels=labels, plot_dir=plot_dir)

    # Extract the average SHAP values (feature contributions) for each class across all features in this model
    raw_avg = {}
    for i in range(shap_values.shape[2]):
        raw_avg[labels[i]] = np.abs(shap_values.values[:, :, i]).mean(axis=0)

    importance_df = pd.DataFrame(raw_avg, columns=labels, index=columns)
    print(importance_df)
    importance_df['row_sum'] = importance_df.sum(axis=1)
    # Save SHAP values to CSV
    importance_df.sort_values(by='row_sum', ascending=False, inplace=False,
                              ignore_index=False).to_csv(
        os.path.join(plot_dir, 'all_class_feature_importance.csv'.format(model_name)))
    # Plot the total SHAP values for all features
    sorted_importance_df = importance_df.sort_values(by='row_sum', ascending=True, inplace=False,
                                                     ignore_index=False)
    sorted_importance_df = sorted_importance_df.drop(columns=["row_sum"])
    top_sorted_importance_df = sorted_importance_df.iloc[-top_feature_num:, :]
    elements = top_sorted_importance_df.index
    shap_class_feature_plot(top_sorted_importance_df, elements,
                            os.path.join(plot_dir, 'all_class_feature_importance.png'.format(model_name)))

# Explainer for LR linear models
def single_class_partial_dependence(model, background, test_df, plot_dir, class_index, class_name, col_index, col_name,
                                    data_index):
    explainer = shap.Explainer(model, background)
    shap_values = explainer(test_df)

    # Partial dependence plot: for all data, a specific feature, and a specific class
    shap_part_dependence_single_plot(shap_values, col_index, col_name=col_name, class_index=class_index,
                                     class_name=class_name, plot_dir=plot_dir)
    # Waterfall plot for a single data point: for a specific data point, all features, and a specific class
    # shap_waterfall_single_plot(shap_values, data_index, class_index, class_name, plot_dir, None)
    # Beeswarm plot
    shap_beeswarm_single_plot(shap_values, class_index, class_name, plot_dir)
    # Heatmap
    shap_heatmap_single_plot(shap_values, class_index, class_name, plot_dir)
    # Average bar plot
    shap_avg_bar_single_plot(shap_values, class_index, class_name, plot_dir)
    # Bar plot for a single data point: for a specific data point, all features, and a specific class
    shap_data_bar_single_plot(shap_values, data_index, test_df, class_index, class_name, plot_dir)


# Partial dependence plot: for all data, a specific feature, and a specific class
def shap_part_dependence_single_plot(shap_values, col_index, col_name, class_index, class_name, plot_dir):
    fig = plt.figure()
    plt.title('SHAP Value part dependence plot | Class {} Feature {}'.format(class_name, col_name), fontsize=20, pad=20)
    shap.plots.scatter(shap_values[:, col_index, class_index], color=shap_values[:, col_index, class_index], show=False)
    fig.savefig(os.path.join(plot_dir, "part_dependence_single.png"), bbox_inches='tight')

# Waterfall plot: for a specific data point, all features, and a specific class
def shap_waterfall_single_plot(explainer, shap_values, test_df, data_index, class_index, class_name, plot_dir, predict_name=None):
    shap_explanation_true = shap.Explanation(values=shap_values[data_index,:, class_index], base_values=explainer.expected_value[class_index], data=test_df.iloc[data_index,:], feature_names=test_df.columns)
    fig = plt.figure()
    plt.title('SHAP Value waterfall plot | Class {}'.format(class_name), fontsize=20, pad=20)
    shap.plots.waterfall(shap_explanation_true, max_display=20, show=False)
    fig.savefig(os.path.join(plot_dir, "waterfall_plot_single_{}_{}.png".format(data_index, class_name)), bbox_inches='tight')
def shap_waterfall_plot(explainer, shap_values, test_df, category_list, plot_dir):
    save_dir = os.path.join(plot_dir, "waterfall")
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # for i in range(len(test_df)):
    for i in range(1):
        data = test_df.iloc[i,:]
        data.to_csv(os.path.join(save_dir,  "waterfall_data_{}.csv".format(i)), index=False)
        for class_index in range(len(category_list)):
            shap_waterfall_single_plot(explainer, shap_values, test_df, i, class_index, category_list[class_index], save_dir, )


# Waterfall plot: for a specific data point, all features, and a specific class
def shap_waterfall_compare_plot(explainer, shap_values, test_df, columns, data_index, class_index, class_name,
                                predict_index, predict_name, plot_dir):
    shap_explanation_true = shap.Explanation(values=shap_values[data_index, :, class_index],
                                             base_values=explainer.expected_value[class_index],
                                             data=test_df.iloc[data_index, :], feature_names=columns)
    shap_explanation_predict = shap.Explanation(values=shap_values[data_index, :, predict_index],
                                                base_values=explainer.expected_value[predict_index],
                                                data=test_df.iloc[data_index, :], feature_names=columns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 30), dpi=80)
    # Plot waterfall on subplot ax1
    shap.waterfall_plot(shap_explanation_true, max_display=15, show=False)
    plt.sca(ax1)  # Set ax1 as the current axis
    plt.title("SHAP Value waterfall plot | true Class {}".format(class_name))

    # Plot waterfall on subplot ax2
    shap.waterfall_plot(shap_explanation_predict, max_display=15, show=False)
    plt.sca(ax2)  # Set ax2 as the current axis
    plt.title("SHAP Value waterfall plot | predict Class {}".format(predict_name))

    # Adjust layout to ensure no overlap
    plt.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, "waterfall_plot_single_{}-{}_{}.png".format(class_name, predict_name, data_index)),
        bbox_inches='tight')

def draw_shap_waterfall_compare_plots(explainer, shap_values, test_df, columns, true_label, false_predict_label, category_list, plot_dir):
    save_dir = os.path.join(plot_dir, "waterfall")
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    error_map = {}
    for i in range(len(true_label)):
        class_name = true_label.iloc[i,0]
        predict_label = false_predict_label.iloc[i,0]
        error_set = "{}_{}".format(class_name, predict_label)
        if error_map.get(error_set) == 2:
            continue
        if error_set not in error_map:
            error_map[error_set] = 0
        error_map[error_set] = error_map.get(error_set) + 1
        class_index = category_list.index(class_name)
        predict_index = category_list.index(predict_label)
        shap_waterfall_compare_plot(explainer, shap_values, test_df, columns, i, class_index, class_name, predict_index, predict_label, save_dir)

# Beeswarm plot: for all data, all features, and a specific class
def shap_beeswarm_single_plot(shap_values, class_index, class_name, plot_dir):
    fig = plt.figure()
    plt.title('SHAP Value beeswarm plot | Class {}'.format(class_name), fontsize=20, pad=20)
    shap.plots.beeswarm(shap_values[:, :, class_index], max_display=20, show=False)
    fig.savefig(os.path.join(plot_dir, "beeswarm_{}_plot.png".format(class_name)), bbox_inches='tight')


def shap_beeswarm_all_plot(explainer, shap_values, labels, test_df, row, plot_dir):
    class_cnt = len(labels)
    col = int((class_cnt + 1) / row)
    fig, axs = plt.subplots(row, col, figsize=(350, 140), constrained_layout=True)
    plt.title('SHAP Value beeswarm plot', pad=20)
    for i in range(class_cnt):
        c_row = i // col  # Calculate row index
        c_col = i % col  # Calculate column index
        plt.sca(axs[c_row, c_col])
        axs[c_row, c_col].set_title(labels[i])
        axs[c_row, c_col].tick_params(axis='y', labelsize=2, labelrotation=30)
        shap_explanation_tem = shap.Explanation(values=shap_values[:, :, i], base_values=explainer.expected_value[i], data=test_df[:], feature_names=test_df.columns)
        shap.plots.beeswarm(shap_explanation_tem, max_display=15, show=False, plot_size=(35, 14))
    fig.savefig(os.path.join(plot_dir, 'all_class_beeswarm_plot.png'), bbox_inches='tight')


# Heatmap: for all data, all features, and a specific class
def shap_heatmap_single_plot(shap_values, class_index, class_name, plot_dir):
    fig = plt.figure()
    plt.title('SHAP Value heatmap plot | Class {}'.format(class_name), fontsize=20, pad=20)
    shap.plots.heatmap(shap_values[:, :, class_index], max_display=20, show=False)
    fig.savefig(os.path.join(plot_dir, "heatmap_{}_plot.png".format(class_name)), bbox_inches='tight')


def shap_heatmap_all_plot(explainer, shap_values, labels, test_df, row, plot_dir):
    class_cnt = len(labels)
    col = int((class_cnt + 1) / row)
    fig, axs = plt.subplots(row, col, figsize=(50, 14), constrained_layout=True)
    plt.title('SHAP Value heatmap plot', pad=20)
    for i in range(class_cnt):
        c_row = i // col  # Calculate row index
        c_col = i % col  # Calculate column index
        plt.sca(axs[c_row, c_col])
        axs[c_row, c_col].set_title(labels[i])
        axs[c_row, c_col].tick_params(axis='y', labelsize=10, labelrotation=30)
        shap_explanation_tem = shap.Explanation(values=shap_values[:, :, i], base_values=explainer.expected_value[i], data=test_df[:], feature_names=test_df.columns)
        shap.plots.heatmap(shap_explanation_tem, max_display=15, show=False, ax=axs[c_row, c_col])
    fig.savefig(os.path.join(plot_dir, 'all_class_heatmap_plot.png'), bbox_inches='tight')


# Average bar plot: for all data, all features, and a specific class
def shap_avg_bar_single_plot(shap_values, class_index, class_name, plot_dir):
    fig = plt.figure()
    plt.title('SHAP Value avg bar plot | Class {}'.format(class_name), fontsize=20, pad=20)
    shap.plots.bar(shap_values[:, :, class_index], max_display=20, show=False)
    fig.savefig(os.path.join(plot_dir, "avg_bar_{}_plot.png".format(class_name)), bbox_inches='tight')

def shap_avg_bar_all_plot(explainer, shap_values, labels, test_df, row, plot_dir):
    class_cnt = len(labels)
    col = int((class_cnt + 1) / row)
    fig, axs = plt.subplots(row, col, figsize=(25, 10), constrained_layout=True)
    plt.title('SHAP Value avg bar plot', pad=20)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i in range(class_cnt):
        c_row = i // col
        c_col = i % col
        plt.sca(axs[c_row, c_col])
        axs[c_row, c_col].set_title(labels[i])
        axs[c_row, c_col].tick_params(axis='y', labelsize=8, labelrotation=30)
        shap_explanation_tem = shap.Explanation(values=shap_values[:, :, i], base_values=explainer.expected_value[i], data=test_df[:], feature_names=test_df.columns)
        shap.plots.bar(shap_explanation_tem, max_display=15, show=False, ax=axs[c_row, c_col])
    # plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'all_class_avg_bar_plot.png'), bbox_inches='tight')


# Bar plot for a specific data point: for a specific data point, all features, and a specific class
def shap_data_bar_single_plot(explainer, shap_values, test_df, data_index, class_index, class_name, plot_dir):
    shap_explanation = shap.Explanation(values=shap_values[data_index,:,class_index], base_values=explainer.expected_value[class_index], data=test_df.iloc[data_index, :], feature_names=test_df.columns)
    fig = plt.figure()
    plt.title('SHAP Value target bar plot | Class {}'.format(class_name), fontsize=20, pad=20)
    shap.plots.bar(shap_explanation, max_display=20, show=False)
    fig.savefig(os.path.join(plot_dir, "data_bar_plot_single_{}.png".format(class_name)), bbox_inches='tight')
    # plt.show()

def get_shap_data(x_train_1, x_test_2, columns, background_num, test_num):
    X_test_df = pd.DataFrame(x_train_1, columns=columns)
    background_df = shap.sample(X_test_df, background_num)
    shap_test_df = X_test_df.iloc[:test_num, :]
    return background_df, shap_test_df

def false_predict_test_data(feature_dir, file_name, num=20):
    error_df = pd.read_csv(os.path.join(feature_dir, file_name)).iloc[:num, :]
    error_df.rename(columns={'actual_label': 'Class'}, inplace=True)
    error_feature = error_df.iloc[:, :-2]
    true_label = error_df.iloc[:, -2:-1]
    false_predict_label = error_df.iloc[:, -1:]
    return error_df, error_feature, true_label, false_predict_label

def compare_shap_data(x_train_1, y_train_1, x_test_2, columns, data_num=10):
    x_train_df = pd.DataFrame(x_train_1, columns=columns)
    if len(y_train_1.shape) > 1:
        y_train_df = pd.DataFrame(np.argmax(y_train_1, axis=1), columns=['Class'])
    else:
        y_train_df = pd.DataFrame(y_train_1, columns=['Class'])
    train_df = pd.concat([x_train_df, y_train_df], axis=1)
    # x_test_df = pd.DataFrame(x_test_2, columns=columns)
    # if len(y_test_2.shape) > 1:
    #     y_test_df = pd.DataFrame(np.argmax(y_test_2, axis=1), columns=['Class'])
    # else:
    #     y_test_df = pd.DataFrame(y_test_2, columns=['Class'])
    # test_df = pd.concat([x_test_df, y_test_df], axis=1)

    # background_df_balance, index_exclude = class_balance_sample(train_df, x=None, y=None, n_samples_per_class=data_num, index_exclude=[])
    # shap_test_df_balance, _ = class_balance_sample(test_df, x=None, y=None, n_samples_per_class=data_num, index_exclude=index_exclude)

    # random sample
    background_df_balance, shap_test_df_balance = class_percent_sample(train_df, x=x_train_df, y=y_train_df,
                                                                       class_num=10, n_samples_per_class=data_num)

    # random get shap data
    # background_df_more, shap_test_df_more = get_shap_data(x_train_1, x_test_2, columns, data_num, data_num)
    return background_df_balance, shap_test_df_balance

def shap_class_feature_plot(sorted_importance_df, elements, plot_path):
    # Use Seaborn's color palette, set to Set2 for higher contrast colors

    colors = sns.color_palette("Set2", n_colors=len(sorted_importance_df.columns))

    # Create figure and axis objects, set figure size to 12x6 inches, resolution to 1200 DPI

    fig, ax = plt.subplots(figsize=(12, 6), dpi=1200)

    # Initialize an array to record the bottom position of each bar, starting at 0

    bottom = np.zeros(len(elements))

    # Iterate over each class and plot horizontal bar charts

    for i, column in enumerate(sorted_importance_df.columns):
        ax.barh(

            sorted_importance_df.index,  # Feature names on the y-axis

            sorted_importance_df[column],  # SHAP values for the current class

            left=bottom,  # Set the starting position of the bar

            color=colors[i],  # Use colors from the palette

            label=column  # Add class names for the legend

        )

        # Update the bottom position for proper stacking of the next bar

        bottom += sorted_importance_df[column]

    # Set x-axis label and title

    ax.set_xlabel('mean(SHAP value|)(average impact on model output magnitude)', fontsize=8)

    ax.set_ylabel('Features', fontsize=8)

    ax.set_title('Feature Importance by Class', fontsize=10)

    # Set y-axis ticks and labels

    ax.set_yticks(np.arange(len(elements)))

    ax.set_yticklabels(elements, fontsize=8)

    # Add text labels at the end of the bars

    for i, el in enumerate(elements):
        ax.text(bottom[i], i, ' ' + str(el), va='center', fontsize=9)

    # Add legend and set legend font size and title

    ax.legend(title='Class', fontsize=10, title_fontsize=12)

    # Disable y-axis ticks and labels

    ax.set_yticks([])  # Remove y-axis ticks

    ax.set_yticklabels([])  # Remove y-axis tick labels

    ax.set_ylabel('')  # Remove y-axis label

    # Remove top and right spines for a cleaner plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    plt.savefig(plot_path, bbox_inches='tight')

    # plt.show()

def shap_run():
    # random_state = 0
    timestep = -1
    k_fold = 5
    model_name = "MLP"
    if model_name == "RFC":
        from algorithm.RFC import model_data_process, model_load
    elif model_name == "Efficient":
        from algorithm.Efficient import model_data_process, model_load
    elif model_name == "SVM":
        from algorithm.SVM import model_data_process, model_load
    elif model_name == "MLP":
        from algorithm.MLP import model_data_process, model_load
    elif model_name == "LR":
        from algorithm.LR import model_data_process, model_load
    elif model_name == "LSTM":
        from algorithm.LSTM import model_data_process, model_load
    elif model_name == "Autoencoder":
        from algorithm.AutoEncoder import model_data_process, model_load
    else:# model_name == "Autoencoder_Efficient":
        from algorithm.AutuEncoder_Efficient import model_data_process, model_load
    # oversample_all = 1.0
    oversample_all = 1
    feature_version = "feature7/all"
    if feature_version.split("/")[0][-1] == "1":  # feature0
        from feature_process.feature1 import process_feature, load_dataset, get_label_map, \
            split_feature_label
    else:  # feature7
        from feature_process.feature7 import process_feature, load_dataset, get_label_map, split_feature_label
    if type(oversample_all) is float:
        oversample_all = float(oversample_all)
        feature_version += "/" + str(oversample_all).replace(".", "_")
    else:
        oversample_all = int(oversample_all)
        if oversample_all == 1:
            feature_version += "/single_sample"
        elif oversample_all == -1:
            feature_version += "/all_sample"
        elif oversample_all == 0:  # 0
            feature_version += "/no_sample"
        else:  # oversample_all== -3 down_sampling
            feature_version += "/under_sample"
    dirname = r"D:\Work\research\netflow\code\Deep-Feature-for-Network-Threat-Detection\algorithm"
    output_dir = os.path.join(dirname, "../output_us/{}".format(feature_version))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_dir = os.path.join(dirname, "../model_us/{}".format(feature_version))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    feature_dir = os.path.join(dirname, "../output_analysis/{}/{}".format(feature_version, model_name))
    input_dir = os.path.join(dirname, "../input/us_features")
    if feature_version.split("/")[0] == "feature0":
        feature_file = 'feature_5.csv'
    elif feature_version.split("/")[0] == "feature1":
        feature_file = 'feature_1.csv'
    elif feature_version.split("/")[0] == "feature2":
        feature_file = 'feature_2.csv'
    elif feature_version.split("/")[0] == "feature3":
        feature_file = 'feature_3.csv'
    elif feature_version.split("/")[0] == "feature4":
        feature_file = 'feature_4.csv'
    elif feature_version.split("/")[0] == "feature5":
        feature_file = 'feature_5.csv'
    else:  # feature7
        feature_file = 'feature_7.csv'
    label_col = "attack_cat"
    feature_subdir = "/".join(feature_version.split("/")[:2])
    dataset_dir = os.path.join(dirname, "../dataset_us", feature_subdir)
    new_train_df = load_dataset(os.path.join(dataset_dir, "dataset.csv"), -1, shuffle=False)
    print("数据集存在空值：{}".format(new_train_df.isnull().values.any()))
    combined_data_X, y = split_feature_label(new_train_df, timestep=timestep)
    # 标签映射为index（string->int）
    category_list, category_map = get_label_map(y)

    x_train_1, y_train_1, x_test_2, y_test_2 = model_data_process(train_X_over=combined_data_X,
                                                                  train_y_over=y, test_X=combined_data_X,
                                                                  test_y=y,
                                                                  data_X_columns=combined_data_X.columns,
                                                                  category_map=category_map,
                                                                  timestep=timestep)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = model_load(os.path.join(model_dir, model_name, "{}_model_257673_0".format(model_name)))
    # background_df = None
    plot_dir = os.path.join(output_dir, 'shap/{}'.format(model_name))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    background_df, test_df = compare_shap_data(
        x_train_1, y_train_1, x_test_2, columns=combined_data_X.columns, data_num=1)

    print(background_df.shape)
    print(test_df.shape)

    explainer, shap_values, shap_explanation = None, None, None
    if model_name == "RFC":
        explainer, shap_values, shap_explanation = SHAP_process_tree_explainer(model=model, columns=combined_data_X.columns,
                                background_df=background_df, test_df=test_df)
    elif model_name in ["MLP", "Efficient"]:
        explainer, shap_values, shap_explanation = SHAP_process_deep_explainer(model=model, columns=combined_data_X.columns,
                                background_df=background_df, test_df=test_df)

    if explainer != None and shap_values != None and shap_explanation != None:
        draw_SHAP_process_tree_explainer_plot(explainer, shap_values, shap_explanation, test_df, columns=combined_data_X.columns, labels=category_list,plot_dir=plot_dir, model_name=model_name,)
        shap_waterfall_plot(explainer, shap_values, test_df, category_list, plot_dir)

# shap_run()