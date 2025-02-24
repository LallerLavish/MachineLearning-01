import sys
import pandas as pd
import numpy as np
from src.utils import load_object
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            pre_path='artifacts/preprocessor.pkl'

            model=load_object(model_path)
            preprocessor=load_object(pre_path)

            trans_feature=preprocessor.transform(features)
            preds=model.predict(trans_feature)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class InputData:
    def __init__(self,gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,test_preparation_course,
        reading_score,writing_score):

        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def data_to_frame(self):
        try:
            input_data_df={
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }

            return pd.DataFrame(input_data_df)
        
        except Exception as e:
            raise CustomException(e,sys)