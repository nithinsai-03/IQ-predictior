import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utiles import save_object
from src.pipeline.utiles import evaluate_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join(PROJECT_ROOT, 'artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    

    def initate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models={
                "RandomForest Regressor": RandomForestRegressor(),
                "GradientBoosting Regressor": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "ExtraTrees Regressor": ExtraTreesRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "DecisionTree Regressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor()
            }

            model_report : dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            

            best_model_score=max(sorted(model_report.values()))

            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6 :
                raise CustomException("No best model found", sys)
            
            logging.info("Best model found on both training and testing Datasets")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            score=r2_score(y_test,predicted)

            return score

        except Exception as e:
            raise CustomException(e,sys)
