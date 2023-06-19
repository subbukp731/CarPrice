import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import sys
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
import warnings

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,1:],
                train_array[:,0:1],
                test_array[:,1:],
                test_array[:,0:1]
            )
            
            models = {
                        "Linear Regression": LinearRegression(),
                        "Lasso": Lasso(),
                        "Ridge": Ridge(),
                        "K-Neighbors Regressor": KNeighborsRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "AdaBoost Regressor": AdaBoostRegressor()
                }
            
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                              models=models)
             
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and test dataset")
            
            
            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                
             )
             
             
            predicted=best_model.predict(x_test)
             
            r2_square= r2_score(y_test,predicted)
            return r2_square
         
         
        except Exception as e:
            raise CustomException(e,sys)
             
    
    