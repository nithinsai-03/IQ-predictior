from dataclasses import dataclass
import sys
import os

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utiles import save_object

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join(PROJECT_ROOT, 'artifacts','preprocessor.pkl')


class Datatransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation

        '''
        try:
            numerical_columns =['writing_score','reading_score']
            categorial_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]

            num_pipeline=Pipeline(
                steps=[
                     #imputer i.e., SimpleImputer is used to fill the missing values using the strategies
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler()),#standard scalar is used to transform the numerical features
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    #imputer i.e., SimpleImputer is used to fill the missing values using the strategies
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder()),# onehot encoder is used to transform the categrial columns
                    ('standardscaling',StandardScaler(with_mean=False))
                    
                ]
            )
            logging.info('numerical columns standard scaling completed')

            logging.info('categroial columns encoding completed')

            #pipeline for both numerical and categorial using ColumnTransformer
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorial_columns)
                ]

            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('read train and test data completed' )

            logging.info('obtaining preprocessor object')
            preprocessor_object=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_columns =['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('applyig preprocessor object on training dataframe and test dataframe.')

            input_feature_train_array=preprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_object.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_array,np.array(target_feature_test_df)]

            logging.info('saved preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_object


            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e,sys)
        