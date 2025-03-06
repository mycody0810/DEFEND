import numpy as np
from keras.optimizers import Adam
from sklearn import metrics
import pandas as pd
def MLP_model(feature_num, class_num, hidden_layers=2, hidden_neurons=256, dropout_rate=0.2, learning_rate=0.00005):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import BatchNormalization
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(feature_num,)))
    model.add(Dropout(dropout_rate))
    for i in range(1, hidden_layers):
        model.add(BatchNormalization())
        model.add(Dense(hidden_neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(class_num, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model
def model_predict(model, test):
    pred = model.predict(test)
    pred = np.argmax(pred, axis=1)
    return pred
def model_predict_test(model, test):
    pred_raw = model.predict(test)
    pred_label = np.argmax(pred_raw, axis=1)
    return pred_label, pred_raw
def model_train(model, model_name, initial_weights, x_train_set, y_train_set, x_test_set, y_test_set, early_stopping):
    model.set_weights(initial_weights)
    print("[Model {} training]".format(model_name))
    history = model.fit(x_train_set, y_train_set,
                        epochs=50, batch_size=128,
                        validation_data=(x_test_set, y_test_set),
                        callbacks=[early_stopping])
    return history
def model_train_kfold(model, model_name, x_train_set, y_train_set, x_test_set, y_test_set):
    print("[Model {} training]".format(model_name))
    model.fit(x_train_set, y_train_set, epochs=5, batch_size=128)
    pred = model_predict(model, x_test_set)
    y_eval = np.argmax(y_test_set, axis=1)
    score = metrics.accuracy_score(y_eval, pred)
    print("Validation score: {}".format(score))
    # joblib.dump(model, os.path.join(model_dir, '{}_model_{}.pkl'.format(model_name, combined_data.shape[0])))
    return score

def model_data_process(train_X_over, train_y_over, test_X, test_y, data_X_columns, category_map=None, timestep=-1):
    x_train_array = train_X_over[data_X_columns].values
    x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1]))
    dummies = pd.get_dummies(train_y_over)  # Classification
    y_train_1 = dummies.values

    x_test_array = test_X[data_X_columns].values
    x_test_2 = np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1]))
    dummies_test = pd.get_dummies(test_y)  # Classification
    y_test_2 = dummies_test.values
    return x_train_1, y_train_1, x_test_2, y_test_2

def model_save(model, model_path):
    # save
    # save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    model.save(model_path+".h5", options=None)

def model_load(model_path, compile=True):
    from keras.models import load_model
    # save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    model = load_model(model_path+".h5", options=None)
    return model