import os
import tensorflow as tf
from keras.callbacks import EarlyStopping
import argparse
from analysis import feature_importance, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from feature_process.feature import get_label_map, training_data_process
from analysis import draw_auc, calculate_detection_rate, calculate_false_positive_rate, model_train_process_plot

def get_false_predict_data(x_test_set, y_test_set, pred, feature_columns, save_dir):
    error_indices = [i for i in range(len(y_test_set)) if y_test_set[i] != pred[i]]
    error_features = x_test_set[error_indices]
    error_actual_labels = [category_list[y_test_set[i]] for i in error_indices]  # 实际标签
    error_pred_labels = [category_list[pred[i]] for i in error_indices]  # 预测标签

    error_df = pd.DataFrame(error_features, columns=feature_columns)
    error_df['actual_label'] = error_actual_labels
    error_df['predicted_label'] = error_pred_labels

    error_df.to_csv(os.path.join(save_dir, 'false_predict_feature.csv'), index=False)
    return error_df

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 创建解析器
parser = argparse.ArgumentParser(description='传入参数')

# 添加命令行参数
parser.add_argument('--best_model_no', type=int, required=True, help='最优模型编号')
parser.add_argument('--random_state', type=int, required=True, help='数据集抽样随机种子')
parser.add_argument('--all_count', type=int, required=False, help='随机抽样总量')
parser.add_argument('--k_fold', type=int, required=True, help='分层交叉验证')
parser.add_argument('--model_name', type=str, required=True, help='模型名称')
parser.add_argument('--feature_version', type=str, required=True, help='特征版本feature7/all')
parser.add_argument('--oversample_all', type=int, required=True, help='上采样策略')
# 解析参数
args = parser.parse_args()
# 打印参数
print(args.__dict__)

# 参数设置
# kfold随机种子
best_model_no = args.best_model_no
# 数据集抽样随机种子
random_state = args.random_state
# 规定随机抽样总量
all_count = args.all_count
# 时序的时间步，仅RNN模型中有效
timestep = -1
# 分层交叉验证
k_fold = args.k_fold
# 是否支持/进行SHAP分析
SHAP_explain = True
# 模型
model_name = args.model_name

# feature_version特征版本0： 自行测试版本
# feature_version特征版本1： 原始特征（train+test 25w）feature1/raw
# feature_version特征版本2： 原始数据集1 2 3 4 共250w
# （废弃）feature_version特征版本3： 新feature3/all 旧feature3/raw
# feature_version特征版本4： 新feature4/all 旧feature4/raw
# feature_version特征版本5： 新feature5/all 旧feature5/raw
# feature_version特征版本7， 特征1+2-feature7/all 旧（1）feature7/raw 特征2 feature7/second
feature_version = args.feature_version
feature_ver = feature_version.split("/")[-1]
if feature_version.split("/")[0][-1] == "1":  # feature1
    from feature_process.feature1 import process_feature, load_dataset, split_feature_label
else:  # feature7
    from feature_process.feature7 import process_feature, load_dataset, split_feature_label

# 上采样设置：0不进行上采样，1最少类别进行上采样，-1全部上采样，float使用ADASYN上采样
oversample_all = args.oversample_all
if oversample_all == 1:
    feature_version += "/single_sample"
elif oversample_all == -1:
    feature_version += "/all_sample"
elif oversample_all == 0:  # 0
    feature_version += "/no_sample"
else:  # ADASYN上采样方法
    feature_version += "/" + str(oversample_all).replace(".", "_")
# dirname = os.path.join(os.path.dirname(sys.argv[0]), ".")
dirname = "."
output_dir = os.path.join(dirname, "../output_analysis/{}/{}".format(feature_version, model_name))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_dir = os.path.join(dirname, "../model_us/{}/{}".format(feature_version, model_name))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
feature_subdir = "/".join(feature_version.split("/")[:2])
dataset_dir = os.path.join(dirname, "../dataset_us", feature_subdir)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
shap_plot_dir = os.path.join(output_dir, 'shap')
if not os.path.exists(shap_plot_dir):
    os.makedirs(shap_plot_dir)
label_col = "attack_cat"
# 读取原始数据集
input_dir = os.path.join(dirname, "../input/us_features")
if feature_version.split("/")[0] == "feature0":
    feature_file = 'feature_1.csv'
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

new_train_df = load_dataset(os.path.join(dataset_dir, "dataset.csv"), size=-1, shuffle=False,
                                random_state=random_state)

data_X, data_y = split_feature_label(new_train_df, timestep=timestep)
combined_data_X, y = data_X, data_y
dataset_shape = combined_data_X.shape[0]
x_columns = combined_data_X.columns
feature_num = combined_data_X.shape[1] if timestep == -1 else int(combined_data_X.shape[1] / timestep)
print("model_name:{}, feature_version:{}, dataset_shape:{}, feature_columns:{}, timestep:{}".format(
    model_name, feature_version, dataset_shape, x_columns, timestep))

# 标签映射为index（string->int）
category_list, category_map = get_label_map(y)

# 模型定义
model = None
if model_name == 'RFC':
    from algorithm.RFC import RFC_model, model_train, model_data_process, model_save, model_load, model_predict_test
    from analysis.feature_analysis import SHAP_process_tree_explainer as SHAP_explainer
    model = RFC_model()
elif model_name == "Efficient":
    from algorithm.Efficient import Efficient_model, model_train, model_data_process, model_save, model_load, \
        model_predict_test
    from algorithm.Efficient import k_fold  # 与Efficient论文k_fold保持一致
    from analysis.feature_analysis import SHAP_process_deep_explainer as SHAP_explainer
    model = Efficient_model(feature_num=feature_num, class_num=len(category_list))
elif model_name == "SVM":
    from algorithm.SVM import SVM_model, model_train, model_data_process, model_save, model_load, model_predict_test
    model = SVM_model(k_fold)
elif model_name == "MLP":
    from algorithm.MLP import MLP_model, model_train, model_data_process, model_save, model_load, model_predict_test
    from analysis.feature_analysis import SHAP_process_deep_explainer as SHAP_explainer
    model = MLP_model(feature_num=feature_num, class_num=len(category_list))
elif model_name == "LR":
    from algorithm.LR import LR_model, model_train, model_data_process, model_save, model_load, model_predict_test
    from analysis.feature_analysis import SHAP_process_linear_explainer as SHAP_explainer
    model = LR_model()
elif model_name == "LSTM":
    from algorithm.LSTM import LSTM_model, model_train, model_data_process, model_save, model_load, model_predict_test
    model = LSTM_model(timesteps=feature_num)
elif model_name == "Autoencoder":
    from algorithm.AutoEncoder import autoencoder_model, model_train, model_data_process, model_save, model_load, \
        model_predict_test
    model = autoencoder_model(feature_num=feature_num, class_num=len(category_list))
elif model_name == "Autoencoder_Efficient":
    from algorithm.AutuEncoder_Efficient import autoencoder_efficient_model, model_train, model_data_process, model_save, \
        model_load, model_predict_test
    model = autoencoder_efficient_model(feature_num=feature_num)
# elif model_name == "Transformer":
#     from algorithm.Transformer import transformer_model, model_train, model_data_process, model_save, model_load, \
#         model_predict_test
# 
#     model = transformer_model(feature_num=feature_num)
# elif model_name == "RNN":
#     from algorithm.RNN import RNN_model, model_train, model_data_process, model_save, model_load, model_predict_test
#     from algorithm.RNN import timestep, k_fold
#     model = RNN_model(timestep=timestep)
# 记录模型训练过程
accuracy_list = []
loss_list = []
val_accuracy_list = []
val_loss_list = []
# 模型评价指标
score_list = []
precision_macro_list = []
recall_macro_list = []
f1_macro_list = []
precision_weighted_list = []
recall_weighted_list = []
f1_weighted_list = []
detect_rate_list = []
false_positive_list = []

x_train_1, y_train_1, x_test_2, y_test_2, x_test_set, y_test_set = None, None, None, None, None, None
# model_num = 0
# 模型训练
# initial_weights = model.get_weights() if model_name in ["MLP", "RNN", "Autoencoder", "Autoencoder_Efficient", "Efficient", "LSTM"] else None
# 定义早停回调
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# k折交叉验证
kFold = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=0)
for train_index, test_index in kFold.split(combined_data_X, y):
    train_X, test_X, train_y, test_y, train_X_over, train_y_over = training_data_process(data_X=combined_data_X,
                                                                                         data_y=y,
                                                                                         train_index=train_index,
                                                                                         test_index=test_index,
                                                                                         cate_cnt=oversample_all,
                                                                                         category_list=category_list)
    x_train_1, y_train_1, x_test_2, y_test_2 = model_data_process(
        train_X_over=train_X_over, train_y_over=train_y_over, test_X=test_X, test_y=test_y,
        data_X_columns=combined_data_X.columns, category_map=category_map, timestep=timestep)
    if model_name in ["Autoencoder_Efficient", "Efficient", "MLP", "RNN", "LSTM", "Autoencoder"]:
        y_eval = np.argmax(y_test_2, axis=1)
    else:
        y_eval = y_test_2
    x_test_set = x_test_2
    y_test_set = y_eval
    break

# 获取最优效果的模型
# 加载模型
model_path = os.path.join(model_dir, '{}_model_{}_{}'.format(model_name, dataset_shape, best_model_no))
model = model_load(model_path)
pred, pred_raw = model_predict_test(model, x_test_set)

false_predict_data = get_false_predict_data(x_test_set, y_test_set, pred, combined_data_X.columns, output_dir)
# auc_info = draw_auc(y_test_set, pred_raw, category_list, output_dir, model_name, data_size=dataset_shape, feature_ver=feature_ver)

# 创建一个 DataFrame
# data = {'model_name': [model_name],
#         'feature_version': [feature_version],
#         'all_count': [all_count],
#         'dataset_shape': ["[{} * {}]".format(combined_data_X.shape[0], combined_data_X.shape[1])],
#         'best_accuracy': [round(score_list[best_model_no], 5)],
#         'best_detect_rate': [round(detect_rate_list[best_model_no], 5)],
#         'best_false_positive': [round(false_positive_list[best_model_no], 5)],
#         'best_precision_weighted': [round(precision_weighted_list[best_model_no], 5)],
#         'best_recall_weighted': [round(recall_weighted_list[best_model_no], 5)],
#         'best_f1_weighted': [round(f1_weighted_list[best_model_no], 5)],
#         'best_precision_macro': [round(precision_macro_list[best_model_no], 5)],
#         'best_recall_macro': [round(recall_macro_list[best_model_no], 5)],
#         'best_f1_macro': [round(f1_macro_list[best_model_no], 5)],
#         'best_model_auc': [auc_info],
#         'info': ["最优模型编号为{}".format(best_model_no)],
#         'val_accuracy': [score_list],
#         'detect_rate': [detect_rate_list],
#         'false_positive': [false_positive_list],
#         'precision_weighted': [precision_weighted_list],
#         'recall_weighted': [recall_weighted_list],
#         'f1_weighted': [f1_weighted_list],
#         'precision_macro': [precision_macro_list],
#         'recall_macro': [recall_macro_list],
#         'f1_macro': [f1_macro_list],
#         }
# df = pd.DataFrame(data)
# 追加写入 CSV 文件
# df.to_csv(output_process_result_path, mode='a', index=False, header=True)

# 计算混淆矩阵
# confusion_matrix = confusion_matrix(y_test_set, pred, labels=[i for i in range(len(category_list))])
# print("最优 confusion_matrix:{}".format(confusion_matrix))
# plot_confusion_matrix(cm=confusion_matrix,
#                       normalize=False,
#                       target_names=category_list,
#                       title="Confusion Matrix",
#                       output_dir=output_dir,
#                       model_name=model_name,
#                       data_size=dataset_shape)
#
# if model_name == "RFC":
#     # 特征重要性分析
#     feature_importance(model=model_load(os.path.join(model_dir, '{}_model_{}_{}'.format(model_name, dataset_shape, best_model_no))),
#                        feature_list=combined_data_X.columns, output_dir=output_dir, model_name=model_name,
#                        data_size=dataset_shape)
# SHAP分析特征重要性
SHAP_explain = False
if SHAP_explain:
    from analysis.feature_analysis import get_shap_data

    background_data = pd.DataFrame(x_train_1, columns=x_columns)
    background_df, shap_test_df = get_shap_data(x_train_1=x_train_1, x_test_2=x_test_set, columns=combined_data_X.columns,
                                                background_num=100, test_num=100)
    SHAP_explainer(model=model, columns=combined_data_X.columns,
                   background_df=background_df, test_df=shap_test_df, labels=category_list,
                   top_feature=["ct_state_ttl", "sttl", "sbytes", "smeansz", "synack", "dttl"],
                   plot_dir=shap_plot_dir, model_name=model_name)

    # 3.记录shap结果：output_shap_info_path
    # data = {'model_name': [model_name],
    #         'feature_version': [feature_version],
    #         'dataset_shape': ["[{} * {}]".format(new_train_df.shape[0] - 1, new_train_df.shape[1])],
    #         # 'accuracy': [round(score, 5)],
    #         'shap_dir': [shap_plot_dir],
    #         'info': ['最优模型：{}_model_{}_{}'.format(model_name, dataset_shape, best_model_no)]}
    # df = pd.DataFrame(data)
    # df.to_csv(output_shap_info_path, mode='a', index=False, header=False, encoding='utf-8')
