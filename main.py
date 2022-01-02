from dataset_generators.base_preparator import BasePreparator
from dataset_generators.individual_datasets import IndividualDatasetPrep
from dataset_generators.PCA_generators import pcaGenerator
from dataset_generators.correlation_generators import CorrelationGenerator

import pandas as pd
import pyarrow.feather as feather

#features = feather.read_feather('E:\Mestrado\\askonas-ids\datasets\\features_dataset.feather')
#print(len(features.columns.tolist()))

#dataset_generator = BasePreparator()
#dataset_generator.base_dataset_pipeline()

#dataset_generator = IndividualDatasetPrep()
#dataset_generator.pipeline()

#dataset_generator = pcaGenerator()
#dataset_generator.pipeline_binary()

dataset_generator = CorrelationGenerator()
dataset_generator.correlation_pipeline()


