from tensorflow.keras.models import Model
import numpy as np  # linear algebra
from sklearn import metrics
import pandas as pd

def autoencoder_efficient_model(feature_num):
    encoder_feature_num = 32
    from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, \
        Reshape
    from tensorflow.keras.models import Sequential

    input_layer = Input(shape=(feature_num,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(encoder_feature_num, activation='relu')(encoded)

    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(feature_num, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)

    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    classifier = Sequential([
        Convolution1D(64, kernel_size=64, activation="relu", input_shape=(encoder_feature_num, 1), padding="same"),
        MaxPooling1D(pool_size=10),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=False)),
        Reshape((128, 1), input_shape=(128,)),
        MaxPooling1D(pool_size=5),
        BatchNormalization(),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.6),
        Dense(10, activation='softmax'),
    ])

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(autoencoder.summary())
    print(classifier.summary())
    return [autoencoder, encoder, classifier]

def model_train_pre(model, model_name, x_train_set, y_train_set, x_test_set, y_test_set):
    print("[Model {} training]".format(model_name))
    [autoencoder, encoder, classifier] = model
    autoencoder.fit(x_train_set, x_train_set, epochs=9, batch_size=32, shuffle=True)
    # Use the encoder part to extract features
    encoded_features = encoder.predict(x_train_set)
    # Feed the encoded features into the classifier
    classifier.fit(encoded_features, y_train_set, epochs=9, batch_size=32)

    # Test the model
    pred = model_predict(model, x_test_set)
    y_eval = np.argmax(y_test_set, axis=1)
    score = metrics.accuracy_score(y_eval, pred)
    print("Validation score: {}".format(score))
    print("classification_report".format(metrics.classification_report(y_eval, pred)))
    # joblib.dump(model, os.path.join(model_dir, '{}_model_{}.pkl'.format(model_name, combined_data.shape[0])))
    return score

def model_train(model, model_name, initial_weights, x_train_set, y_train_set, x_test_set, y_test_set, early_stopping):
    model[0].set_weights(initial_weights[0])  # Reset to initial weights
    model[1].set_weights(initial_weights[1])
    model[2].set_weights(initial_weights[2])
    print("[Model {} training]".format(model_name))
    [autoencoder, encoder, classifier] = model
    autoencoder.fit(x_train_set, x_train_set, epochs=8, batch_size=32, shuffle=True)
    # Use the encoder part to extract features
    encoded_features = encoder.predict(x_train_set)
    encoder_features_test = encoder.predict(x_test_set)
    # Feed the encoded features into the classifier
    history = classifier.fit(encoded_features, y_train_set, epochs=8, batch_size=32,
                             validation_data=(encoder_features_test, y_test_set),
                             callbacks=[early_stopping])
    # joblib.dump(model, os.path.join(model_dir, '{}_model_{}.pkl'.format(model_name, combined_data.shape[0])))
    return history

def model_predict(model, test):
    [_, encoder, classifier] = model
    encoded_features = encoder.predict(test)
    pred = classifier.predict(encoded_features)
    pred = np.argmax(pred, axis=1)
    return pred

def model_predict_test(model, test):
    [_, encoder, classifier] = model
    encoded_features = encoder.predict(test)
    pred_raw = classifier.predict(encoded_features)
    pred_label = np.argmax(pred_raw, axis=1)
    return pred_label, pred_raw

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
    model_list = ["autoencoder", "encoder", "classifier"]
    for i in range(len(model)):
    # save
    # save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
        model[i].save(model_path+"_{}.h5".format(model_list[i]), options=None)

def model_load(model_path, compile=True):
    from keras.models import load_model
    model_list = ["autoencoder", "encoder", "classifier"]
    model = []
    for i in range(len(model_list)):
        # save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
        model.append(load_model(model_path+"_{}.h5".format(model_list[i]), compile=False, options=None))
    return model