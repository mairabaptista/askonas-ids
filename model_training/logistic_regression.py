from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import time

from utils import save_model
from config.config import Config

class LogisticRegressionClassifier():
    def __init__(self) -> None:
        self.start_time: float
        self.end_time: float
        self.time_stats_file = open(Config.STATS_AND_IMAGES_FOLDER + "\\time_stats.txt", "a")
        self.param_grid = [
                            {'penalty' : ['elasticnet', 'l1', 'l2'],
                            'solver' : ['saga']
                            }
                        ]
        self.param_grid_1 = [
                            {'penalty' : ['elasticnet'],
                            'l1_ratio': [0.5],
                            'solver' : ['saga']
                            }
                        ]
        self.param_grid_2 = [
                            {'penalty' : ['l1'],
                            'solver' : ['saga']
                            }
                        ]
        self.param_grid_3 = [
                            {'penalty' : ['l2'],
                            'solver' : ['saga']
                            }
                        ]


    def train(self, X_train_oh, X_train, y_train) -> None:
        print("Starting LogReg Classifier")
        self.start_time = time.time()
        
        scaler = StandardScaler()
        scaler.fit(X_train_oh)

        cls_lr = LogisticRegression(verbose=2)
        gridF_1 = GridSearchCV(estimator=cls_lr, param_grid=self.param_grid_1, cv = 3, verbose = 1, n_jobs = 10)
        gridF_2 = GridSearchCV(estimator=cls_lr, param_grid=self.param_grid_2, cv = 3, verbose = 1, n_jobs = 10)
        gridF_3 = GridSearchCV(estimator=cls_lr, param_grid=self.param_grid_3, cv = 3, verbose = 1, n_jobs = 10)

        # cls_lr.fit(scaler.transform(X_train_oh), y_train.label_is_attack)
        gridF_1.fit(scaler.transform(X_train_oh), y_train.label_is_attack)
        gridF_2.fit(scaler.transform(X_train_oh), y_train.label_is_attack)
        gridF_3.fit(scaler.transform(X_train_oh), y_train.label_is_attack)

        self.end_time = time.time()
        self.time_stats_file.write("---- Time stats for Logistic Regression Classifier ----")
        self.time_stats_file.write("\n")
        self.time_stats_file.write("--- %s seconds---" % (self.end_time - self.start_time))
        self.time_stats_file.write("\n")
        self.time_stats_file.close()

        save_model(gridF_1, "logreg_model_1.sav")
        save_model(gridF_2, "logreg_model_1.sav")
        save_model(gridF_3, "logreg_model_1.sav")
        print("Saved LogReg Classifier")