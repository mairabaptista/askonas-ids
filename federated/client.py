import warnings
import flwr as fl
import numpy as np
import pyarrow.feather as feather
import pandas as pd

from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.metrics import log_loss

# import utils
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'E:\\Mestrado\\askonas-ids')

from config.config import Config

if __name__ == "__main__":
    # Load MNIST dataset from https://www.openml.org/d/554
    # (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    # partition_id = np.random.choice(10)
    # (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # catboost starts here
    X_train = pd.DataFrame = feather.read_feather('E:\\Mestrado\\askonas-ids\\federated\\X_train.feather')
    X_test = pd.DataFrame = feather.read_feather('E:\\Mestrado\\askonas-ids\\federated\\X_test.feather')
    X_eval = pd.DataFrame = feather.read_feather('E:\\Mestrado\\askonas-ids\\federated\\X_eval.feather')
    y_eval = pd.DataFrame = feather.read_feather('E:\\Mestrado\\askonas-ids\\federated\\y_eval.feather')
    y_train = pd.DataFrame = feather.read_feather('E:\\Mestrado\\askonas-ids\\federated\\y_train.feather')
    y_test = pd.DataFrame = feather.read_feather('E:\\Mestrado\\askonas-ids\\federated\\y_test.feather')

    train_pool = Pool(X_train, y_train.label_is_attack, cat_features=['protocol'])
    eval_pool = Pool(X_eval, y_eval.label_is_attack, cat_features=['protocol'])
    test_pool = Pool(X_test, cat_features=['protocol'])

    minority_class_weight = len(y_train[y_train.label_is_attack == 0]) / len(y_train[y_train.label_is_attack == 1])
    
    model = CatBoostClassifier(loss_function='Logloss',
                                eval_metric='Recall',                        
                                class_weights=[1, 4.906679153],
                                task_type='GPU',
                                verbose=True)

    # cls_cb.fit(train_pool, eval_set=eval_pool)

    # Create LogisticRegression Model
    '''
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    '''
    # Setting initial parameters, akin to model.compile for keras models
    # utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            # return utils.get_model_parameters(model)
            return model.feature_importance_

        def fit(self, parameters, config):  # type: ignore
            # utils.set_model_params(model, parameters)
            model.get_feature_importance()
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            # return utils.get_model_parameters(model), len(X_train), {}
            return model.feature_importance_, len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            # utils.set_model_params(model, parameters)
            model.get_feature_importance()
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=MnistClient())