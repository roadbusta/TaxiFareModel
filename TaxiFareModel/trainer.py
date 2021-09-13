# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from TaxiFareModel.utils import compute_rmse, minkowski_distance_gps
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib

#Estimators
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor



class Trainer():
    def __init__(self, X, y, estimator, experiment_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = experiment_name
        self.mflow_uri = "https://mlflow.lewagon.co/"
        self.estimator = estimator

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        self.pipe = Pipeline([
        ('preproc', preproc_pipe),
        ('estimator', self.estimator)
        ])

        self.mlflow_log_param("Model type", self.estimator)
        return self.pipe

    def run(self):
        """set and train the pipeline"""
        # pipeline = Trainer.set_pipeline(self)
        self.pipeline = self.pipe.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self,X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        # pipeline = Trainer.run(X_test,y_test)
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric('RMSE', rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.mflow_uri)  #ensures that this only happens once
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    #Saving the model with joblib
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')

# 🚨 replace with your country code, city, github_nickname and model name and version

if __name__ == "__main__":
    #Experiment name
    experiment_name = "[AUS] [MEL] [roadbusta] TaxiFare v1.3"

    # get data
    df = get_data()

    # clean data
    df = clean_data(df)

    # Drop number of passengers
    df = df.drop(columns = 'passenger_count')

    # set X and y
    y = df["fare_amount"]

    #Drop the fare amount
    X = df.drop("fare_amount", axis=1)

    # # Note: DO NOT IMPLEMENT MANHATTEN HERE_ IT SHOULD BE USED IN DISTANCE
    # df['manhattan_dist'] = minkowski_distance_gps(df['pickup_latitude'], df['dropoff_latitude'],
    #                                           df['pickup_longitude'], df['dropoff_longitude'], 1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    #Create an estimator list
    estimators = [LinearRegression(), KNeighborsRegressor(), RandomForestRegressor(),
                  SVR(), AdaBoostRegressor()]


    for estimator in estimators:
        #instanciate  pipeline class
        trainer_inst = Trainer(X_train, y_train, estimator, experiment_name)

        #build a pipeline
        pipe = trainer_inst.set_pipeline()

        # train
        pipeline = trainer_inst.run()

        # evaluate
        rmse = trainer_inst.evaluate(X_val, y_val)
        print(f'rmse for {estimator}: ', rmse)

    # print trainer model
    experiment_id = trainer_inst.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

    #dump the model
    # trainer_inst.save_model()
