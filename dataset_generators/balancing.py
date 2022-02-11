import numpy as np
import pandas as pd
import glob
import os
import pyarrow.feather as feather
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from config.config import Config

class Balancer():
    def __init__(self) -> None:
        self.features: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\" + 'features_dataset.feather')
        self.target: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\" + 'target_dataset.feather')
    
    def feature_target_separation(self) -> None:
        print("splitting")
        #self.features = df.drop(columns=['label', 'label_cat', 'label_is_attack'])
        self.features = self.df[self.df.columns.drop(list(self.df.filter(regex='label')))]
        self.target = self.df[['label_is_attack', 'label_cat', 'label']]
        

    def undersamppler(self):
        cnts = self.target.label_cat.value_counts()
        sample_dict = {}

        for i in np.unique(self.target.label_cat):
            sample_dict[i] = max(cnts[i], 139890)

        rus = RandomUnderSampler(sampling_strategy=sample_dict, random_state=42)
        X_train_undersampled, y_train_undersampled = rus.fit_resample(self.features, self.target.label_cat)

        features = pd.DataFrame(X_train_undersampled, columns=self.features.columns)
        target = pd.DataFrame(y_train_undersampled, columns=self.target.label_cat)

        self.serialize_dataset(features, 'balanced/undersample_features_dataset.feather')
        self.serialize_dataset(target, 'balanced/undersample_target_dataset.feather')

    def oversampler(self):
        cnts = self.target.label_cat.value_counts()
        sample_dict = {}

        for i in np.unique(self.target.label_cat):
            sample_dict[i] = max(cnts[i], 10000000)        

        sm = SMOTENC(sampling_strategy=sample_dict, categorical_features=[0], n_jobs=24, random_state=42)
        X_train_oversampled, y_train_oversampled = sm.fit_resample(self.features, self.target.label_cat)

        features = pd.DataFrame(X_train_oversampled, columns=self.features.columns)
        target = pd.DataFrame(y_train_oversampled, columns=self.target.label_cat)

        self.serialize_dataset(features, 'balanced/oversample_features_dataset.feather')
        self.serialize_dataset(target, 'balanced/oversample_target_dataset.feather')

    def sampler(self):
        self.oversampler()
        self.undersamppler




