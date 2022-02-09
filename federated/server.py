import flwr as fl
import pandas as pd
import pyarrow.feather as feather
import time

from sklearn.metrics import log_loss
from typing import Dict

# import utils
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'E:\\Mestrado\\askonas-ids')

from config.config import Config

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    fl.server.start_server("localhost:5040", config={"num_rounds": 3})


'''
def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: CatBoostClassifier):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    #_, (X_test, y_test) = utils.load_mnist()

    X_test = pd.DataFrame = feather.read_feather('E:\\Mestrado\\askonas-ids\\federated\\X_test.feather')
    y_test = pd.DataFrame = feather.read_feather('E:\\Mestrado\\askonas-ids\\federated\\y_test.feather')

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        # utils.set_model_params(model, parameters)
        model.get_feature_importance()
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = CatBoostClassifier()
    # utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server("0.0.0.0:8080", strategy=strategy, config={"num_rounds": 3})
    '''