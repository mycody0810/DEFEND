from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from feature_process.feature import cate2index

def RFC_model(n_estimators=150, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
    RFC = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 max_features=max_features,
                                 random_state=42,
                                 n_jobs=-1)
    model = RFC
    return model
def model_predict(model, test):
    pred = model.predict(test)
    return pred
def model_predict_test(model, test):
    # pred_raw = model.predict(test)
    pred_raw = model.predict_proba(test)
    pred_label = np.argmax(pred_raw, axis=1)
    return pred_label, pred_raw

def model_train(model, model_name, x_train_set, y_train_set, x_test_set, y_test_set):
    print("[Model {} training]".format(model_name))
    model.fit(x_train_set, y_train_set)
    pred = model_predict(model, x_test_set)
    y_eval = y_test_set
    score = metrics.accuracy_score(y_eval, pred)
    print("Validation score: {}".format(score))
    # joblib.dump(model, os.path.join(model_dir, '{}_model_{}.pkl'.format(model_name, combined_data.shape[0])))
    return score

def model_data_process(train_X_over, train_y_over, test_X, test_y, data_X_columns, category_map=None, timestep=-1):
    x_train_array = train_X_over[data_X_columns].values
    x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1]))
    y_train_1 = cate2index(train_y_over, category_map)

    x_test_array = test_X[data_X_columns].values
    x_test_2 = np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1]))
    y_test_2 = cate2index(test_y, category_map)
    return x_train_1, y_train_1, x_test_2, y_test_2
def model_save(model, model_path):
    import joblib
    joblib.dump(model, model_path+".pkl")

def model_load(model_path, compile=True):
    import joblib
    model = joblib.load(model_path+".pkl")
    return model
