import os
import statistics

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from feature_process.feature import get_label_map, training_data_process
from analysis import *

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Create parser
parser = argparse.ArgumentParser(description='传入参数')

# Add command line arguments
parser.add_argument('--kfold_random_state', type=int, required=False, default=0, help='kfold random seed')
parser.add_argument('--random_state', type=int, required=False, default=-1, help='Dataset sampling random seed')
parser.add_argument('--all_count', type=int, required=False, default=-1, help='Total amount for random sampling')
parser.add_argument('--k_fold', type=int, required=False, default=5, help='Stratified k-fold cross-validation')
parser.add_argument('--model_name', type=str, required=True, help='Model name')
parser.add_argument('--feature_version', type=str, required=True, help='Feature version feature7/all')
parser.add_argument('--oversample_all', type=str, required=False, default='1', help='Oversampling strategy')
# RFC tuning parameters
parser.add_argument("--n_estimators", type=int, required=False, default=150, help='Number of decision trees in RFC')
parser.add_argument("--max_depth", type=int, required=False, default=-1, help='Maximum depth of the RFC tree')
parser.add_argument("--min_samples_split", type=int, required=False, default=2, help='Minimum number of samples required to split an internal node in RFC')
parser.add_argument("--min_samples_leaf", type=int, required=False, default=1, help='Minimum number of samples required to be at a leaf node in RFC')
parser.add_argument("--max_features", type=str, required=False, default='sqrt', help='Maximum number of features considered at each split in RFC')

# Deep learning tuning parameters
parser.add_argument("--learning_rate", type=float, required=False, default=0.001, help='Learning rate')

# MLP tuning parameters
parser.add_argument("--hidden_layers", type=int, required=False, default=2, help='Number of hidden layers in MLP')
parser.add_argument("--hidden_neurons", type=int, required=False, default=128, help='Number of neurons in hidden layer of MLP')
parser.add_argument("--dropout_rate", type=float, required=False, default=0.5, help='Dropout rate')

# Autoencoder tuning parameters
parser.add_argument("--encoder_hidden_dim", type=int, required=False, default=128, help='Dimension of the hidden layer in the encoder')
parser.add_argument("--classifier_hidden_dim", type=int, required=False, default=64, help='Dimension of the hidden layer in the classifier')
parser.add_argument("--encoder_dropout_rate", type=float, required=False, default=0.2, help='Dropout rate of the Autoencoder model')

# Efficient tuning parameters
parser.add_argument("--filter_num", type=int, required=False, default=64, help='Number of channels in one-dimensional convolution')
parser.add_argument("--bilstm_1_num", type=int, required=False, default=64, help='Number of BiLSTM layer 1')
parser.add_argument("--bilstm_2_num", type=int, required=False, default=64, help='Number of BiLSTM layer 2')
parser.add_argument("--efficient_dropout_rate", type=float, required=False, default=0.6, help='Dropout rate of the Efficient model')

args = parser.parse_args()
# Print parameters
print(args.__dict__)

# Parameter settings
# Hyperparameter tuning
search_param = True
# RFC tuning parameters
n_estimators = args.n_estimators
max_depth = args.max_depth if args.max_depth != -1 else None
min_samples_split = args.min_samples_split
min_samples_leaf = args.min_samples_leaf
max_features = args.max_features

# Common deep learning parameters
learning_rate = args.learning_rate

# MLP tuning parameters
hidden_layers = args.hidden_layers
hidden_neurons = args.hidden_neurons
dropout_rate = args.dropout_rate

# Autoencoder tuning parameters
encoder_hidden_dim = args.encoder_hidden_dim
classifier_hidden_dim = args.classifier_hidden_dim
encoder_dropout_rate = args.encoder_dropout_rate

# Efficient tuning parameters
filter_num = args.filter_num
bilstm_1_num = args.bilstm_1_num
bilstm_2_num = args.bilstm_2_num
efficient_dropout_rate = args.efficient_dropout_rate

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

if model_name == "RFC":
    params = "n_estimators:{},max_depth:{},min_samples_split:{},min_samples_leaf:{},max_features:{}".format(
            n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
if model_name == "MLP":
    params = "hidden_layers:{},hidden_neurons:{},dropout_rate:{},learning_rate:{}".format(
        hidden_layers, hidden_neurons, dropout_rate, learning_rate)
if model_name == "Autoencoder":
    params = "encoder_hidden_dim:{},classifier_hidden_dim:{},encoder_dropout_rate:{},learning_rate:{}".format(
        encoder_hidden_dim, classifier_hidden_dim, encoder_dropout_rate, learning_rate)
if model_name == "Efficient":
    params = "filter_name:{},bilstm_1_num:{},bilstm_2_num:{},efficient_dropout_rate:{},learning_rate:{}".format(
        filter_num, bilstm_1_num, bilstm_2_num,efficient_dropout_rate, learning_rate)

# feature_version feature version 1: Raw features (train + test 250k) feature1/raw
# feature_version feature version 7, features 1+2 - feature7/all old (1) feature7/raw feature2 feature7/second
feature_version = args.feature_version
feature_ver = feature_version.split("/")[-1]
if feature_version.split("/")[0][-1] == "1":  # feature1
    from feature_process.feature1 import process_feature, load_dataset, split_feature_label
else:  # feature7
    from feature_process.feature7 import process_feature, load_dataset, split_feature_label

oversample_all = args.oversample_all
if '.' in oversample_all:
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
    else:  # oversample_all== -3 Down-sampling
        feature_version += "/under_sample"
# dirname = os.path.join(os.path.dirname(sys.argv[0]), "./")
dirname = "./"
output_dir = os.path.join(dirname, "./output_param/{}/{}/".format(feature_version, model_name))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_dir = os.path.join(dirname, "./model_param/{}/{}/".format(feature_version, model_name))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
feature_subdir = "/".join(feature_version.split("/")[:2])
dataset_dir = os.path.join(dirname, "dataset_param", feature_subdir)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
# Record model training process and average results
output_param_info_path = 'output_param/output_param_info.csv'

label_col = "attack_cat"
# Read the original dataset
input_dir = os.path.join(dirname, "input/us_features")
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

if not os.path.exists(os.path.join(dataset_dir, "dataset.csv")) or feature_version.split("/")[0][-1] in ["0"]:
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

    model = RFC_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,)
elif model_name == "Efficient":
    from algorithm.Efficient import Efficient_model, model_train, model_data_process, model_save, model_load, \
        model_predict_test

    model = Efficient_model(feature_num=feature_num, class_num=len(category_list), filter_num=filter_num,
                            bilstm_1_num=bilstm_1_num, bilstm_2_num=bilstm_2_num, learning_rate=learning_rate)
elif model_name == "SVM":
    from algorithm.SVM import SVM_model, model_train, model_data_process, model_save, model_load, model_predict_test

    model = SVM_model(k_fold)
elif model_name == "MLP":
    from algorithm.MLP import MLP_model, model_train, model_data_process, model_save, model_load, model_predict_test

    model = MLP_model(feature_num=feature_num, class_num=len(category_list), hidden_layers=hidden_layers,
                      hidden_neurons=hidden_neurons, dropout_rate=dropout_rate, learning_rate=learning_rate)
elif model_name == "LR":
    from algorithm.LR import LR_model, model_train, model_data_process, model_save, model_load, model_predict_test

    model = LR_model()
elif model_name == "LSTM":
    from algorithm.LSTM import LSTM_model, model_train, model_data_process, model_save, model_load, model_predict_test

    model = LSTM_model(timesteps=feature_num)
elif model_name == "Autoencoder":
    from algorithm.AutoEncoder import autoencoder_model, model_train, model_data_process, model_save, model_load, \
        model_predict_test

    model = autoencoder_model(feature_num=feature_num, class_num=len(category_list),
                              encoder_hidden_dim=encoder_hidden_dim, classifier_hidden_dim=classifier_hidden_dim,
                              dropout_rate=dropout_rate, learning_rate=learning_rate)
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

        best_epoch = early_stopping.stopped_epoch + 1  # +1 因为epoch是从0开始的
        loss_list.append(history.history['loss'][:best_epoch])
        val_loss_list.append(history.history['val_loss'][:best_epoch])
        accuracy_list.append(history.history['accuracy'][:best_epoch])
        val_accuracy_list.append(history.history['val_accuracy'][:best_epoch])
        # Save model training process
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
    break

# Get the best model performance
best_model_no = score_list.index(max(score_list))
# Load model
model = model_load(os.path.join(model_dir, '{}_model_{}_{}'.format(model_name, dataset_shape, best_model_no)), )
pred, pred_raw = model_predict_test(model, x_test_set)
auc_info = draw_auc(y_test_set, pred_raw, category_list, output_dir, model_name, data_size=dataset_shape, feature_ver=feature_ver)

# Create a DataFrame
data = {'model_name': [model_name],
        'feature_version': [feature_version],
        'params': [params],
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
df.to_csv(output_param_info_path, mode='a', index=False, header=False, encoding='gbk')