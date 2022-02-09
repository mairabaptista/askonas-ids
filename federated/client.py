import warnings
import flwr as fl
import numpy as np
import pyarrow.feather as feather
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers, metrics
from sklearn.metrics import log_loss
from numpy import load

# import utils
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'E:\\Mestrado\\askonas-ids')

from config.config import Config
from utils import *

K = keras.backend

def create_model(input_dims, 
                 nr_layers, 
                 nr_units, 
                 activation, 
                 kerner_initializer,
                 optimizer,
                 dropout_layer=None):
    model = models.Sequential()
    model.add(layers.Input(shape=[input_dims]))
    
    for l in range(nr_layers):
        model.add(layers.Dense(nr_units, activation=activation, kernel_initializer=kerner_initializer))
        
    if dropout_layer:
        model.add(dropout_layer)
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=[metrics.AUC(curve='PR'),
                           metrics.Precision(), 
                           metrics.Recall()])
    return model

if __name__ == "__main__":

    '''
    X_train = load(Config.FEDERATED_FOLDER + '\X_train.npy', allow_pickle=True)
    y_train = load(Config.FEDERATED_FOLDER + '\y_train.npy', allow_pickle=True)
    X_val = load(Config.FEDERATED_FOLDER + '\X_val.npy', allow_pickle=True)
    y_val = feather.read_feather(Config.FEDERATED_FOLDER + "\\" + 'y_val.feather')
    X_test = load(Config.FEDERATED_FOLDER + '\X_test.npy', allow_pickle=True)
    y_test = feather.read_feather(Config.FEDERATED_FOLDER + "\\" + 'y_test.feather')
    # column_names = load("data\\federated" + '\column_names.npy', allow_pickle=True)'''
    X_train = load("E:\\Mestrado\\askonas-ids\\datasets\\smaller_federated"+ '\\X_train.npy', allow_pickle=True)
    y_train = load("E:\\Mestrado\\askonas-ids\\datasets\\smaller_federated" + '\\y_train.npy', allow_pickle=True)
    X_val = load("E:\\Mestrado\\askonas-ids\\datasets\\smaller_federated" + '\\X_val.npy', allow_pickle=True)
    y_val = feather.read_feather("E:\\Mestrado\\askonas-ids\\datasets\\smaller_federated" + "\\" + 'y_val.feather')
    X_test = load("E:\\Mestrado\\askonas-ids\\datasets\\smaller_federated" + '\\X_test.npy', allow_pickle=True)
    y_test = feather.read_feather("E:\\Mestrado\\askonas-ids\\datasets\\smaller_federated" + "\\" + 'y_test.feather')

    input_dims = X_train.shape[1]

    # using the best params from DL 
    nr_layers = 5
    nr_units = 300
    dropout_rate = 0.22339774943469998
    lr = (0.001 * 0.61157158868869)

    y_train_is_attack = (y_train != 0).astype('int')

    minority_class_weight = len(y_train_is_attack[y_train_is_attack == 0]) / len(y_train_is_attack[y_train_is_attack == 1])

    class_weights = { 
            0: 1, 
            1: minority_class_weight
    }

    model = create_model(input_dims=input_dims,
                        nr_layers=nr_layers,
                        nr_units=nr_units,
                        activation='elu',
                        kerner_initializer='he_normal',
                        dropout_layer=layers.Dropout(dropout_rate),
                        optimizer=optimizers.Adam(lr=lr))

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            # model.fit(X_train, y_train, epochs=1, batch_size=32)
            model.fit(x=X_train, 
                    y=y_train_is_attack,
                    validation_data=(X_val, y_val.label_is_attack.values),
                    batch_size=4096,
                    epochs=1)
            model.save(Config.MODELS_FOLDER + "\\federated\\" + 'fed_model.h5')
            return model.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            e = model.evaluate(X_test, y_test.label_is_attack.values)
            e = {out: e[i] for i, out in enumerate(model.metrics_names)}

            return float(e['loss']), len(X_test), {"auc": float(e['auc']), 'precision': float(e['precision']), 'recall': float(e['recall'])}

    # Start Flower client
    fl.client.start_numpy_client("localhost:5040", client=CifarClient())