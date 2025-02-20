# This file is created to transform the data for example 
# categorical to numerical and so on

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pickle
from src.utils import save_object

@dataclass
class Data_transform_config:
    preprocessor_data_config_path=os.path.join('artifacts','preprocessor.pkl')

class Data_transform:
    def __init__(self):
        self.data_transform_config=Data_transform_config()

    def data_transformer(self):
        try:
            numerical_columns=["math_score","reading_score","writing_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "test_preparation_course",
                "lunch"
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(sparse_output=False))
                ]
            )

            logging.info("Scaling has Initiated")
            logging.info("Encoding has Initiated")

            preprocessor=ColumnTransformer(
                [
                ('numerical',num_pipeline,numerical_columns),
                ('categorical',cat_pipeline,categorical_columns)
                ]
            )
            logging.info("Transformation has done")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiated_data_transformer(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            logging.info("Reading of test and train data is completed")
            logging.info("fetching the preprocessor object")
            preprocessor=self.data_transformer()
            logging.info("preprocessor object is fetched")
            target_col="average"
            logging.info("Spliting train and test data into input and target section")
            input_train_df=train_data.drop(target_col,axis=1)
            target_train_df=train_data[target_col]

            input_test_df=test_data.drop(target_col,axis=1)
            target_test_df=test_data[target_col]

            logging.info("preprocessing the data")
            train_transformed_data=preprocessor.fit_transform(input_train_df)
            test_transformed_data=preprocessor.transform(input_test_df)
            
            ## concatenation
            train_arr=np.c_[train_transformed_data,np.array(target_train_df)]
            test_arr=np.c_[test_transformed_data,np.array(target_test_df)]
            logging.info("saving preprocessing object...... Done..")

            save_object(
                file_path=self.data_transform_config.preprocessor_data_config_path,
                obj=preprocessor
            )
            return train_arr,test_arr,self.data_transform_config.preprocessor_data_config_path
        
        except Exception as e:
            raise CustomException(e,sys)