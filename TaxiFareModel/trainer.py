# imports
import pandas as pd
import numpy as numpy
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property


MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "RexElardo"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"

class Trainer():
    
    
    


    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
      
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        distance_pipe = Pipeline([
            ('distance',DistanceTransformer()),
            ('scaler', StandardScaler())
            ])
        time_pipe = Pipeline([
            ('timefeatures', TimeFeaturesEncoder("pickup_datetime")),
            ('encoding', OneHotEncoder(handle_unknown='ignore'))
            ])
        preproc = ColumnTransformer([
            ('distance', distance_pipe, ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']),
            ('time', time_pipe,['pickup_datetime'])
            ])
        self.pipeline = Pipeline([
                ('preproc',preproc),
                ('KNN',KNeighborsRegressor())
                ])
        

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)


    

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        a=self.pipeline.predict(X_test)
        self.mlflow_log_param('model', 'KnnRegressor')
        self.mlflow_log_metric('rmse', compute_rmse(a,y_test))
        return compute_rmse(a,y_test)
    
    
    # def train(self):

    #     for model in ["KnnRegressor"]:
    #         self.mlflow_create_run()
    #         self.mlflow_log_metric("rmse", evaluate(X_test,y_test) )
    #         self.mlflow_log_param("model", model)



if __name__ == "__main__":
    # get data
    df = get_data()
    df = clean_data(df)
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']
    # clean data
    # set X and y
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    # train
    train = Trainer(X_train,y_train)
    train.run()
    # train.train()

    # evaluate
    print(train.evaluate(X_test, y_test))
