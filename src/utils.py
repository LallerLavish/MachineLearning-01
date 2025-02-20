import os
import sys

import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def validate_model(X_train,y_train,X_test,y_test,models):
    try:
        models_list=list(models.values())
        score_test={}
        for i in range(len(models_list)):
            model=models_list[i]

            model.fit(X_train,y_train)
            X_pred_test=model.predict(X_test)

            test_score=r2_score(y_test,X_pred_test)

            score_test[list(models.keys())[i]]=test_score

        return score_test
    except Exception as e:
        raise CustomException(e,sys)