import os
import statistics

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from feature_process.feature import get_label_map, training_data_process
from analysis import *
from analysis.shap_analysis import draw_SHAP_process_tree_explainer_plot

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Create parser
parser = argparse.ArgumentParser(description='Input parameters')

# Add command line arguments
parser.add_argument('--kfold_random_state', type=int, required=False, default=0, help='kfold random seed')
parser.add_argument('--random_state', type=int, required=False, default=-1, help='Dataset sampling random seed')
parser.add_argument('--all_count', type=int, required=False, default=-1, help='Total amount for random sampling')
parser.add_argument('--k_fold', type=int, required=False, default=5, help='Stratified k-fold cross-validation')
parser.add_argument('--model_name', type=str, required=True, help='Model name')
parser.add_argument('--feature_version', type=str, required=True, help='Feature version feature7/all')
parser.add_argument('--oversample_all', type=str, required=False, default='1', help='Oversampling strategy')
# Parse parameters
args = parser.parse_args()
# Print parameters
print(args.__dict__)

# Parameter settings
# kfold random seed
kfold_random_state = args.kfold_random_state
# Dataset sampling random seed
random_state = args.random_state
# Specify the total random sampling amount
all_count = args.all_count
# Time steps for sequential data, only effective for RNN model
timestep = -1
# Stratified k-fold cross-validation
k_fold = args.k_fold
# Whether to support/conduct SHAP analysis
SHAP_explain = True
# Model
model_name = args.model_name

# feature_version feature version 1: Raw features (train + test 250k) feature1/raw
# feature_version feature version 7, features 1+2 -- feature7/all, feature 2 -- feature7/raw
feature_version = args.feature_version
feature_ver = feature_version.split("/")[-1]
if feature_version.split("/")[0][-1] == "1":  # feature1
    from feature_process.feature1 import process_feature, load_dataset, split_feature_label
else:  # feature7
    from feature_process.feature7 import process_feature, load_dataset, split_feature_label

# Oversampling settings: 0 means no oversampling, 1 means oversample the least frequent class, -1 means oversample all, float means ADASYN oversampling
oversample_all = args.oversample_all
if '.' in oversample_all:
    # ADASYN oversampling method
    oversample_all = float(oversample_all)
    feature_version += "/" + str(oversample_all).replace(".", "_")
else:
    oversample_all = int(oversample_all)
    if oversample_all > 10:
        feature_version += "/target{}_sample".format(oversample_all)
    elif oversample_all == 1:
        feature_version += "/single_sample"
    elif oversample_all == -1:
        feature_version += "/all_sample"
    elif oversample_all == 0:  # 0
        feature_version += "/no_sample"
    else: # oversample_all== -3 Down-sampling
        feature_version += "/under_sample"
# dirname = os.path.join(os.path.dirname(sys.argv[0]), "./")
dirname = "./"
output_dir = os.path.join(dirname, "./output_us/{}/{}/".format(feature_version, model_name))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_dir = os.path.join(dirname, "./model_us/{}/{}/".format(feature_version, model_name))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
feature_subdir = "/".join(feature_version.split("/")[:2])
dataset_dir = os.path.join(dirname, "dataset_us", feature_subdir)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
shap_plot_dir = os.path.join(output_dir, 'shap/'.format(model_name))
if not os.path.exists(shap_plot_dir):
    os.makedirs(shap_plot_dir)
# Record model training process and average results
output_process_result_path = 'output_us/avg_result_record.csv'
# experiment records
final_sample_path = 'output_us/final_sample.csv'
# Record shap save path
output_shap_info_path = 'output_us/output_shap_info.csv'
label_col = "attack_cat"
# Read the original dataset
input_dir = os.path.join(dirname, "input/us_features")
if feature_version.split("/")[0] == "feature1":
    feature_file = 'feature_1.csv'
else:  # feature7
    feature_file = 'feature_7.csv'

if not os.path.exists(os.path.join(dataset_dir, "dataset.csv")) or feature_version.split("/")[0][-1] in ["0"] or random_state!=-1 or True:
    new_train_df = process_feature(input_dir=input_dir, feature_file=feature_file, feature_version=feature_version,
                                   label_col=label_col, dataset_dir=dataset_dir, shuffle=False,
                                   random_state=random_state, all_count=all_count,
                                   target_file="feature_1.csv")
else:
    new_train_df = load_dataset(
        os.path.join(dataset_dir, "dataset.csv"),
        size=-1, shuffle=False, random_state=random_state,
        target_percent_file=os.path.join(input_dir, "feature_1.csv"))

data_X, data_y = split_feature_label(new_train_df, timestep=timestep)
combined_data_X, y = data_X, data_y
dataset_shape = combined_data_X.shape[0]
x_columns = combined_data_X.columns
feature_num = combined_data_X.shape[1] if timestep == -1 else int(combined_data_X.shape[1] / timestep)
print("model_name:{}, feature_version:{}, dataset_shape:{}, feature_columns:{}, timestep:{}".format(
    model_name, feature_version, dataset_shape, x_columns, timestep))

# Map labels to index (string->int)
category_list, category_map = get_label_map(y)

# Model definition
model = None
if model_name == 'RFC':
    from algorithm.RFC import RFC_model, model_train, model_data_process, model_save, model_load, model_predict_test
    from analysis.shap_analysis import SHAP_process_tree_explainer as SHAP_explainer

    model = RFC_model()
elif model_name == "Efficient":
    from algorithm.Efficient import Efficient_model, model_train, model_data_process, model_save, model_load, \
        model_predict_test
    from analysis.shap_analysis import SHAP_process_deep_explainer as SHAP_explainer

    model = Efficient_model(feature_num=feature_num, class_num=len(category_list))
elif model_name == "SVM":
    from algorithm.SVM import SVM_model, model_train, model_data_process, model_save, model_load, model_predict_test

    model = SVM_model(k_fold)
elif model_name == "MLP":
    from algorithm.MLP import MLP_model, model_train, model_data_process, model_save, model_load, model_predict_test
    from analysis.shap_analysis import SHAP_process_deep_explainer as SHAP_explainer

    model = MLP_model(feature_num=feature_num, class_num=len(category_list))
elif model_name == "LR":
    from algorithm.LR import LR_model, model_train, model_data_process, model_save, model_load, model_predict_test
    from analysis.shap_analysis import SHAP_process_linear_explainer as SHAP_explainer

    model = LR_model()
elif model_name == "LSTM":
    from algorithm.LSTM import LSTM_model, model_train, model_data_process, model_save, model_load, model_predict_test

    model = LSTM_model(timesteps=feature_num)
elif model_name == "Autoencoder":
    from algorithm.AutoEncoder import autoencoder_model, model_train, model_data_process, model_save, model_load, \
        model_predict_test
    SHAP_explain = False
    model = autoencoder_model(feature_num=feature_num, class_num=len(category_list))
elif model_name == "Autoencoder_Efficient":
    from algorithm.AutuEncoder_Efficient import autoencoder_efficient_model, model_train, model_data_process, \
        model_save, \
        model_load, model_predict_test

    model = autoencoder_efficient_model(feature_num=feature_num)
elif model_name == "XGB":
    from algorithm.XGB import XGB_model, model_train, model_data_process, model_save, model_load, model_predict_test
    model = XGB_model()
elif model_name == "KNN":
    from algorithm.KNN import KNN_model, model_train, model_data_process, model_save, model_load, model_predict_test
    model = KNN_model()

# Record model training process
accuracy_list = []
loss_list = []
val_accuracy_list = []
val_loss_list = []
# Model evaluation metrics
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
# Model training
if model_name in ["MLP", "RNN", "Efficient", "LSTM"]:
    initial_weights = model.get_weights()
elif model_name in ["Autoencoder", "Autoencoder_Efficient"]:
    initial_weights = [model[0].get_weights(), model[1].get_weights(), model[2].get_weights()]
else:
    initial_weights = None
# k-fold cross-validation
kFold = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=kfold_random_state)
model_no = 0
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
    if model_name in ["RFC", "LR", "SVM", "XGB", "KNN"]:
        score = model_train(model=model, model_name=model_name,
                            x_train_set=x_train_1, y_train_set=y_train_1,
                            x_test_set=x_test_2, y_test_set=y_test_2)

    else:
        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model_train(model=model, model_name=model_name,
                              initial_weights=initial_weights,
                              x_train_set=x_train_1, y_train_set=y_train_1,
                              x_test_set=x_test_2, y_test_set=y_test_2,
                              early_stopping=early_stopping)

        best_epoch = early_stopping.stopped_epoch + 1  # +1 because epoch starts from 0
        loss_list.append(history.history['loss'][:best_epoch])
        val_loss_list.append(history.history['val_loss'][:best_epoch])
        accuracy_list.append(history.history['accuracy'][:best_epoch])
        val_accuracy_list.append(history.history['val_accuracy'][:best_epoch])
        # Save model training process
        if random_state == -1:
            model_train_process_plot(train_history=history, plot_output_dir=output_dir, model_name=model_name,
                                     data_count=dataset_shape, best_epoch=best_epoch, model_no=model_no)
    # Save model
    model_save(model, os.path.join(model_dir, '{}_model_{}_{}'.format(model_name, dataset_shape, model_no)))
    pred, pred_raw = model_predict_test(model, x_test_2)
    if model_name in ["Autoencoder_Efficient", "Efficient", "MLP", "RNN", "LSTM", "Autoencoder"]:
        y_eval = np.argmax(y_test_2, axis=1)
    else:
        y_eval = y_test_2
    # Calculate validation accuracy
    score = accuracy_score(y_eval, pred)
    score_list.append(score)
    # Record best model validation set
    if max(score_list) == score:
        x_test_set = x_test_2
        y_test_set = y_eval
    # Calculate macro-averaged precision and recall
    precision_macro_list.append(precision_score(y_eval, pred, average='macro'))
    recall_macro_list.append(recall_score(y_eval, pred, average='macro'))
    f1_macro_list.append(f1_score(y_eval, pred, average='macro'))
    # Calculate weighted-averaged precision and recall
    precision_weighted_list.append(precision_score(y_eval, pred, average='weighted'))
    recall_weighted_list.append(recall_score(y_eval, pred, average='weighted'))
    f1_weighted_list.append(f1_score(y_eval, pred, average='weighted'))
    detect_rate_list.append(calculate_detection_rate(pred, y_eval, category_list.index("Normal")))
    false_positive_list.append(calculate_false_positive_rate(pred, y_eval, category_list.index("Normal")))
    model_no += 1

# Get the best model performance
best_model_no = score_list.index(max(score_list))
# Load model
model = model_load(os.path.join(model_dir, '{}_model_{}_{}'.format(model_name, dataset_shape, best_model_no)))
pred, pred_raw = model_predict_test(model, x_test_set)
if random_state == -1 or all_count == -1:
    auc_info = draw_auc(y_test_set, pred_raw, category_list, output_dir, model_name, data_size=dataset_shape, feature_ver=feature_ver)
else:
    auc_info = ""
# Create a DataFrame
data = {'model_name': [model_name],
    'feature_version': [feature_version],
    'all_count': [all_count],
    'dataset_shape': ["[{} * {}]".format(combined_data_X.shape[0], combined_data_X.shape[1])],
    'avg_accuracy': [round(statistics.mean(score_list), 5)],
    'avg_detect_rate': [round(statistics.mean(detect_rate_list), 5)],
    'avg_false_positive': [round(statistics.mean(false_positive_list), 5)],
    'avg_precision_weighted': [round(statistics.mean(precision_weighted_list), 5)],
    'avg_recall_weighted': [round(statistics.mean(recall_weighted_list), 5)],
    'avg_f1_weighted': [round(statistics.mean(f1_weighted_list), 5)],
    'avg_precision_macro': [round(statistics.mean(precision_macro_list), 5)],
    'avg_recall_macro': [round(statistics.mean(recall_macro_list), 5)],
    'avg_f1_macro': [round(statistics.mean(f1_macro_list), 5)],
    'best_model_auc': [auc_info],
    'info': ["Best model number is {}".format(best_model_no)],
    'val_accuracy': [score_list],
    'detect_rate': [detect_rate_list],
    'false_positive': [false_positive_list],
    'precision_weighted': [precision_weighted_list],
    'recall_weighted': [recall_weighted_list],
    'f1_weighted': [f1_weighted_list],
    'precision_macro': [precision_macro_list],
    'recall_macro': [recall_macro_list],
    'f1_macro': [f1_macro_list],
    }
df = pd.DataFrame(data)
# Append to CSV file
df.to_csv(output_process_result_path, mode='a', index=False, header=False, encoding='utf-8')

if random_state != -1:
    print("End of small batch random sampling experiment")
    exit()
# Calculate confusion matrix
confusion_matrix = confusion_matrix(y_test_set, pred, labels=[i for i in range(len(category_list))])
print("Optimal confusion_matrix:{}".format(confusion_matrix))
plot_confusion_matrix(cm=confusion_matrix,
                      normalize=False,
                      target_names=category_list,
                      title="Confusion Matrix",
                      output_dir=output_dir,
                      model_name=model_name,
                      data_size=dataset_shape)

if model_name == "RFC":
    # Feature importance analysis
    feature_importance(
        model=model_load(os.path.join(model_dir, '{}_model_{}_{}'.format(model_name, dataset_shape, best_model_no))),
        feature_list=combined_data_X.columns, output_dir=output_dir, model_name=model_name,
        data_size=dataset_shape)
# SHAP analysis of feature importance, limited by machine, only perform SHAP analysis for RFC
SHAP_explain = True
if SHAP_explain and random_state == -1 and oversample_all != -1 and model_name == 'RFC':
    shap_dir = 'output_shap/{}_{}_random_{}'.format(feature_version.replace("/", "_"), model_name, random_state)
    if not os.path.exists(shap_dir):
        os.makedirs(shap_dir)
    from analysis.shap_analysis import compare_shap_data
    background_data = pd.DataFrame(x_train_1, columns=x_columns)
    background_df, test_df = compare_shap_data(x_train_1,
                                               y_train_1,
                                               x_test_2,
                                               columns=combined_data_X.columns,
                                               data_num=50)
    explainer, shap_values, shap_explanation = SHAP_explainer(
        model=model, columns=combined_data_X.columns,
        background_df=background_df, test_df=test_df)
    draw_SHAP_process_tree_explainer_plot(explainer=explainer, shap_values=shap_values, shap_explanation=shap_explanation, test_df=test_df, columns=combined_data_X.columns,
                                          labels=category_list, plot_dir=shap_dir, model_name=model_name)

    # 3. Record shap results: output_shap_info_path
    data = {'model_name': [model_name],
            'feature_version': [feature_version],
            'dataset_shape': ["[{} * {}]".format(new_train_df.shape[0] - 1, new_train_df.shape[1])],
            # 'accuracy': [round(score, 5)],
            'shap_dir': [shap_dir],
            'info': ['Best model: {}_model_{}_{}'.format(model_name, dataset_shape, best_model_no)]}
    df = pd.DataFrame(data)
    df.to_csv(output_shap_info_path, mode='a', index=False, header=False, encoding='utf-8')
