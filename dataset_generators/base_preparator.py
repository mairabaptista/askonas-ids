import numpy as np
import pandas as pd
import glob
import os
import pyarrow.feather as feather

from config.config import Config

class BasePreparator():
    def __init__(self) -> None:
        self.csv_files = glob.glob(os.path.join(Config.CIC_IDS_2018_PROCESSED_CSVS, '*.csv'))
        self.df: pd.Dataframe = pd.concat((pd.read_csv(f, dtype=Config.data_types) for f in self.csv_files))
        self.features: pd.DataFrame
        self.target: pd.Dataframe

    def replace_inf_in_columns(self, df) -> None:
        print("replacing infs")
        inf_columns = [c for c in df.columns if df[df[c] == np.inf][c].count() > 0]
        for col in inf_columns:
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            mean = df[col].mean()
            df[col].fillna(mean, inplace=True)

    def replace_negative_values_with_mean(self, df) -> None:
        print("replacing numerics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.values
        
        columns = [c for c in numeric_cols if df[df[c] < 0][c].count() > 0]
        for col in columns:
            mask = df[col] < 0
            df.loc[mask, col] = np.nan
            mean = df[col].mean()
            df[col].fillna(mean, inplace=True)

    def create_labels(self, df) -> None:
        print("creating labels")
        df['label'] = df.label.astype('category')
        df['label_code'] = df['label'].cat.codes
        df['label_is_attack'] = df.label.apply(lambda x: 0 if x == 'Benign' else 1)

        attack_types = [a for a in df.label.value_counts().index.tolist() if a != 'Benign']

        for a in attack_types:
            l = 'label_is_attack_' + a.replace('-', ' ').replace(' ', '_').lower()
            df[l] = df.label.apply(lambda x: 1 if x == a else 0)
        
        df['label_cat'] = df.label.astype('category').cat.codes
        df['label_is_attack'] = (df.label != 'Benign').astype('int')
    
    def serialize_dataset(self, df, file_name) -> None:
        print("Serializing at: ", Config.DATASETS_FOLDER + "\\" + file_name)
        feather.write_feather(df, Config.DATASETS_FOLDER + "\\" + file_name)

    def drop_undesired_features(self, df) -> None:
        print("dropping features")
        stats = df.describe()
        std = stats.loc['std']
        features_no_variance = std[std == 0.0].index
        df = df.drop(columns=features_no_variance)  # dropping features with no variance
        
        df = df.drop(columns=['timestamp', 'dst_port'])  # dropping features that will not be relevant to final estimation
    
    def feature_target_separation(self, df) -> None:
        print("splitting")
        self.features = df.drop(columns=['label', 'label_cat', 'label_is_attack'])
        self.target = df[['label_is_attack', 'label_cat', 'label']]

    def base_dataset_pipeline(self):
        self.replace_inf_in_columns(self.df)
        self.replace_negative_values_with_mean(self.df)
        self.create_labels(self.df)
        self.drop_undesired_features(self.df)
        self.feature_target_separation(self.df)
        self.serialize_dataset(self.features, 'features_dataset.feather')
        self.serialize_dataset(self.target, 'target_dataset.feather')
