# this file is created to train and evaluate the model.

import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from src.utils import validate_model
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,validate_model

@dataclass
class TrainerConfig:
    model_train_file_path=os.path.join("artifacts","model.pkl")

class ModelTrain:
    def __init__(self):
        self.model_train_config=TrainerConfig()
    
    def initiateTraining(self,train_arr,test_arr):
        try:    
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1], #label
                test_arr[:,:-1], #label
                test_arr[:,-1]
            )

            models={
                "RandomForest":RandomForestRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "GradientBoost":GradientBoostingRegressor(),
                "Linear":LinearRegression(),
                "KNN":KNeighborsRegressor(),
                "XGB":XGBRegressor(),
                "Ada":AdaBoostRegressor(),
                "Cat":CatBoostRegressor(verbose=False)
            }

            model_report:dict=validate_model(X_train,y_train,X_test,y_test,models=models)

            maxi_Score=max(sorted(model_report.values()))
            print(model_report.values())
            best_model_name=list(model_report.keys())[list(model_report.values()).index(maxi_Score)]

            best_model=models[best_model_name]

            if(maxi_Score<0.6):
                raise CustomException("No best Model found")
            
            logging.info("Best model has founded and trained")

            logging.info("Saving the model pkl")
            save_object(self.model_train_config.model_train_file_path,best_model)
            

            predicted=best_model.predict(X_test)
            return maxi_Score
        
        except Exception as e:
            raise CustomException(e,sys)