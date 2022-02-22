# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data , clean_data


class Trainer():
    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None


    def set_pipeline(self):
        #distance pipeline
         dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
            ])
         #time feature pipeline
         time_pipe = Pipeline([
             ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
             ('ohe', OneHotEncoder(handle_unknown='ignore'))
             ])
         #preprocessing pipeline
         preproc_pipe = ColumnTransformer([
             ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
             ('time', time_pipe, ['pickup_datetime'])
             ], remainder="drop")
         #Model pipeline
         pipe = Pipeline([('preproc', preproc_pipe),
                          ('linear_model', LinearRegression())
                          ])
         return pipe

    def run(self, X_train, y_train, pipeline):
        """set and train the pipeline"""
        # train the pipelined model
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, X_test, y_test, pipeline):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse



if __name__ == "__main__":
    # get data
    df = get_data(nrows=1000)
    print(df.head)
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    train_toto = Trainer()
    trained_pipe = train_toto.run(X_train, y_train, train_toto.set_pipeline())
    # evaluate
    rmse_print = train_toto.evaluate(X_test, y_test,trained_pipe )
    print(rmse_print)
