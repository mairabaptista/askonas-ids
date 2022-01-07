from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time

from utils import save_model
from config.config import Config

class RFClassifier():
    def __init__(self) -> None:
        self.start_time: float
        self.end_time: float
        self.time_stats_file = open(Config.STATS_AND_IMAGES_FOLDER + "\\time_stats.txt", "a")

        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Minimum number of samples required to split a node
        # min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        # min_samples_leaf = [1, 2]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        self.grid = {'max_features': max_features,
                    'bootstrap': bootstrap,
                    'class_weight': 'balanced'}
        self.grid_1 = {'max_features': ['auto'],
                    'bootstrap': [True],
                    'class_weight': 'balanced'}
        self.grid_2 = {'max_features': ['sqrt'],
                    'bootstrap': [True],
                    'class_weight': 'balanced'}
        self.grid_3 = {'max_features': ['auto'],
                    'bootstrap': [False],
                    'class_weight': 'balanced'}
        self.grid_4 = {'max_features': ['sqrt'],
                    'bootstrap': [False],
                    'class_weight': 'balanced'}


    def train(self, X_train_oh, X_train, y_train) -> None:
        print("Starting Random Forest Classifier")
        self.start_time = time.time()

        cls_forest = RandomForestClassifier(verbose=1, n_jobs=-1)

        gridF_1 = GridSearchCV(estimator=cls_forest, param_grid=self.grid_1, cv = 3, verbose = 1, n_jobs = 10)
        gridF_2 = GridSearchCV(estimator=cls_forest, param_grid=self.grid_2, cv = 3, verbose = 1, n_jobs = 10)
        gridF_3 = GridSearchCV(estimator=cls_forest, param_grid=self.grid_3, cv = 3, verbose = 1, n_jobs = 10)
        gridF_4 = GridSearchCV(estimator=cls_forest, param_grid=self.grid_4, cv = 3, verbose = 1, n_jobs = 10)

        # cls_forest.fit(X_train_oh, y_train.label_is_attack)
        gridF_1.fit(X_train_oh, y_train.label_is_attack)
        gridF_2.fit(X_train_oh, y_train.label_is_attack)
        gridF_3.fit(X_train_oh, y_train.label_is_attack)
        gridF_4.fit(X_train_oh, y_train.label_is_attack)

        self.end_time = time.time()
        self.time_stats_file.write("---- Time stats for Random Forest Classifier ----")
        self.time_stats_file.write("\n")
        self.time_stats_file.write("--- %s seconds---" % (self.end_time - self.start_time))
        self.time_stats_file.write("\n")
        self.time_stats_file.close()

        save_model(gridF_1, "rf_model_1.sav")
        save_model(gridF_2, "rf_model_2.sav")
        save_model(gridF_3, "rf_model_3.sav")
        save_model(gridF_4, "rf_model_4.sav")
        print("Saved Random Forest Classifier")