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


def get_feature_col(filedir="../input/us_features/", filename="feature_7.csv"):
    df = pd.read_csv(os.path.join(filedir, filename), low_memory=False)
    print(df.shape)
    columns = df.columns
    pd.DataFrame(columns.tolist(), columns=["feature"]).to_csv(
        os.path.join(filedir, filename.split(".")[0] + "_columns.csv"), index=False)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          output_dir="../output_us/",
                          model_name="test",
                          data_size=-1, ):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title("{} {}".format(model_name, title))
    plt.title("Confusion Matrix", fontname="SimSun")

    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=14)
    plt.savefig(os.path.join(output_dir, '{}_{}_confusion matrix.png'.format(model_name, data_size)),
                bbox_inches='tight')
    # plt.show()

# Feature Importance Analysis
def feature_importance(model, feature_list, output_dir, model_name, data_size):
    # Get feature importances
    importances = model.feature_importances_
    #
    importance_df = pd.DataFrame({
        'feature': feature_list,
        'importance': importances
    })
    importance_df['feature_index'] = importance_df.index
    importance_df.sort_values(by='importance', ascending=False, inplace=True, ignore_index=True)
    importance_df = importance_df[['feature_index', 'feature', 'importance']]
    importance_df.to_csv(
        os.path.join(output_dir, '{}_{}_feature_importance.csv'.format(model_name, data_size)))

    # Process the top 30% most important features
    # feature_cnt = int(len(importances) * 0.3)
    feature_cnt = 20
    # Plot feature importances
    # indices_all = np.argsort(importances)
    # indices = indices_all[:feature_cnt]
    # print(indices_all)
    plt.figure(figsize=(20, 15))
    plt.title("Feature Importances")
    plt.barh(range(0, feature_cnt), importance_df.loc[:feature_cnt - 1, "importance"], align="center")
    plt.yticks(range(0, feature_cnt), importance_df.loc[:feature_cnt - 1, "feature"], fontsize=16, rotation=45)
    plt.xlabel("Feature Importance")
    plt.savefig(os.path.join(output_dir, '{}_{}_feature_importance.png'.format(model_name, data_size)),
                bbox_inches='tight')
    # plt.show()


def REF_process():
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the base model
    model = RandomForestClassifier()

    # Define RFE to select 5 features
    rfe = RFE(estimator=model, n_features_to_select=5)

    # Fit RFE
    rfe.fit(X_train, y_train)

    # Selected features
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    # Train the model and make predictions
    model.fit(X_train_rfe, y_train)
    y_pred = model.predict(X_test_rfe)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy after RFE: {accuracy:.4f}')

    # Check the selected features
    print(f'Selected features: {rfe.support_}')
    print(f'Feature ranking: {rfe.ranking_}')

# get_feature_col()
