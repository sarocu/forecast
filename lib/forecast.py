import json
import pickle
import numpy
from datetime import datetime, timedelta

import statsmodels.api as sm
from sklearn import linear_model
import pandas
from scipy import stats


class Forecast:
    """Stochastic forecasting of weather for use in building energy simulations

    Given time series weather data for some location, we seek to build a forecasting
    model so that we may generate a number of possible weather scenarios.
    """
    def __init__(self, horizon=24):
        """Constructor method for class forecast

        Accepts a parameter to define the forecast horizon in hours and sets it
        as the instance variable self.horizon

        Keyword arguments:
        horizon -- defines forecast horizon in hours. Defualt = 24 hours.
        """
        self.horizon = horizon
        self.dataBox = None

    def persist(self, fileName='..\\data\\forecast.p'):
        """Simple pickle method, accepts a file name

        Keyword arguments:
        fileName -- a string for naming the pickle. Default = 'forecast.p'
        """
        pickle.dump(self, open(fileName, 'wb'))

    def load_tmy(self, data):
        """Parses a csv via pandas, expects a TMY3 file as 'data'

        A csv is loaded and parsed by pandas. The first row of the file is immediately removed (in a TMY it's garbage)
        and the remaining data is sorted by date and stored in the instance variable 'self.dataBox'

        Keyword arguments:
        data -- the filename of a csv containing hourly output of an energy model
        """
        just_data = pandas.read_csv(data, parse_dates=True,  low_memory=False)
        just_data['Date (MM/DD/YYYY)'] = just_data['Date (MM/DD/YYYY)'].map(lambda x: x[:-5])
        just_data['index'] = just_data['Date (MM/DD/YYYY)'] + '/1900' + ' ' + just_data['Time (HH:MM)']
        just_data.sort_values(by='index')
        self.dataBox = just_data.set_index('index')
        return self

    def add_response(self, response):
        pass

    def add_predictor(self, predictor):
        pass

    def auto_regressive(self, lag, start_date):
        """
        Fits a vector autoregressive model based on the previously set response and predictor variables
        :return:
        A vector of predictions over the horizon
        """
        start_horizon = datetime.strptime(start_date, '%m/%d/%Y %H:%M')
        end_horizon = start_horizon + timedelta(hours=self.horizon)
        data = self.dataBox.ix[start_horizon.strftime('%m/%d/%Y %H:%M').replace(' 0', ' ').lstrip('0').replace('/0', '/'): end_horizon.strftime('%m/%d/%Y %H:%M').replace(' 0', ' ').lstrip('0').replace('/0', '/')]


    def fit_glm(self):
        """Fits a Bayes Ridge Regression model to the data

        Before this method will work, at least one response and one predictor must be set 
        """
        if not self.dataBox:
            return None
        return self
