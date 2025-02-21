import os
import sys

import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def validate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        models_list=list(models.values())
        params_list=list(params.keys())
        score_test={}
        for i in range(len(models_list)):
            model=models_list[i]
            param=params[params_list[i]]

            grid_model=GridSearchCV(model,param,cv=3)
            grid_model.fit(X_train,y_train)

            model.set_params(**grid_model.best_params_)
            model.fit(X_train,y_train)

            X_pred_test=grid_model.predict(X_test)

            test_score=r2_score(y_test,X_pred_test)

            score_test[list(models.keys())[i]]=test_score

        return score_test
    except Exception as e:
        raise CustomException(e,sys)