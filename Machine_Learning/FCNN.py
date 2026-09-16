import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def get_data(nl, num_nodes, selected_beta_values, tc, datadir_list):
    alldata = np.zeros((1, int(num_nodes + 1)))
    for datadir in datadir_list:
        # datadir = 'C:\\Users\\calga\\PycharmProjects\\Hyperbolic_Ising_ML\\ML_Final_Project_Final_Run'
        datafiles = os.listdir(datadir)
        # alldata = np.zeros((1, int(num_nodes+1)))
        for sbv in selected_beta_values:
            tval = 1 / sbv
            if tval < tc:
                labelval = 1
            else:
                labelval = 0
            for df in datafiles:
                if 'nl{}beta{}'.format(nl, sbv) in df:
                    reldata = np.load(datadir + '\\' + df)
                    reldata_with_label = np.hstack((np.copy(reldata), np.repeat(labelval, np.size(reldata, 0)).reshape(-1,1)))
                    alldata = np.vstack((alldata, reldata_with_label))
    return(alldata[1:, :])

def prepare_data(alldata):
    training_data, test_data = train_test_split(alldata, test_size=0.2, random_state=42, shuffle=True)
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]
    training_data, validation_data = train_test_split(training_data, test_size=0.2, random_state=42, shuffle=True)
    training_features = training_data[:, :-1]
    training_labels = training_data[:, -1]
    validation_features = validation_data[:, :-1]
    validation_labels = validation_data[:, -1]
    return(training_features, training_labels, test_features, test_labels, validation_features, validation_labels)

def prepare_data_novalidation(alldata):
    training_data, test_data = train_test_split(alldata, test_size=0.2, random_state=42, shuffle=True)
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]
    training_features = training_data[:, :-1]
    training_labels = training_data[:, -1]
    return(training_features, training_labels, test_features, test_labels)

def FCNN_Ising_model_singlelayer(alldata, savemodel=False, savedir=''):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Nadam(),
                  metrics=['accuracy'])

    training_features, training_labels, test_features, test_labels, validation_features, validation_labels = prepare_data(alldata)

    if savemodel == False:
        history = model.fit(training_features, training_labels, epochs=10,
                        validation_data=(validation_features, validation_labels))
    else:
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=savedir+'\\trained_FCNN_model_checkpoint.keras')
        history = model.fit(training_features, training_labels, epochs=10,
                            validation_data=(validation_features, validation_labels),
                            callbacks=[save_callback])

    test_loss, test_accuracy = model.evaluate(test_features, test_labels)

    pred_test_labels = model.predict(test_features)
    confusion_matrix_result = confusion_matrix(test_labels, np.argmax(pred_test_labels, axis=1))

    return(history, test_loss, test_accuracy, confusion_matrix_result)


def get_data_for_test(nl, num_nodes, selected_beta_values, tc, datadir_list):
    alldata = np.zeros((1, int(num_nodes + 1)))
    for datadir in datadir_list:
        datafiles = os.listdir(datadir)
        for sbv in selected_beta_values:
            tval = 1 / sbv
            if tval < tc:
                labelval = 1
            else:
                labelval = 0
            for df in datafiles:
                if 'nl{}beta{}'.format(nl, sbv) in df:
                    reldata = np.load(datadir + '\\' + df)
                    reldata_with_label = np.hstack((np.copy(reldata), np.repeat(labelval, np.size(reldata, 0)).reshape(-1,1)))
                    alldata = np.vstack((alldata, reldata_with_label[::500, :]))
    return(alldata[1:, :])