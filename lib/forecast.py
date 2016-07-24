import json
import pickle
import numpy as np
from datetime import datetime, timedelta
import re
import csv

import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import normalize
import pandas
from scipy.stats import norm


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
        self.normal = None
        self.simulation_time = '1/2/1900 01:00'
        self.response_variables = []
        self.predictor_variables = []

    def clean_tmy(self, data):
        """The first line of the tmy3 file is removed

        :param data: data is the filename of the csv (assuming TMY3).
        :return:
        Saves a copy of the file in the data directory
        """

        thing = open(data, 'r')
        original = csv.reader(thing)
        next(original)
        name = open(data.replace('.csv', '_clean.csv'), 'w', newline='')
        updated = csv.writer(name, delimiter=',')
        updated.writerows(original)


    def persist(self, fileName='..\\data\\forecast.p'):
        """Simple pickle method, accepts a file name

        Keyword arguments:
        fileName -- a string for naming the pickle. Default = 'forecast.p'
        """
        pickle.dump(self, open(fileName, 'wb'))

    def load_tmy(self, data):
        """Parses a csv via pandas, expects a TMY3 file as 'data'

        A csv is loaded and parsed by pandas.The data is sorted by date and stored in the instance variable
        'self.dataBox'. An additional instance variable, self.normal, stores the mean normalized data set.

        Keyword arguments:
        data -- the filename of a csv containing hourly output of an energy model
        """
        just_data = pandas.read_csv(data, parse_dates=True,  low_memory=False)
        just_data['Date (MM/DD/YYYY)'] = just_data['Date (MM/DD/YYYY)'].map(lambda x: x[:-5])
        just_data['Time (HH:MM)'] = just_data['Time (HH:MM)'].map(lambda x: '00:00' if x == '24:00:00' else x)
        just_data['index'] = just_data['Date (MM/DD/YYYY)'] + '/1900' + ' ' + just_data['Time (HH:MM)']
        just_data.sort_values(by='index')
        self.dataBox = just_data.set_index('index')

        source_column = '.* source'
        pattern = re.compile(source_column)
        for column in just_data:
            if pattern.match(column):
                just_data = just_data.drop(column, axis=1)
        just_data = just_data.drop(['Date (MM/DD/YYYY)', 'Time (HH:MM)'], axis=1)
        self.normal = normalize(just_data.set_index('index'), axis=0)
        return self

    def add_response(self, response):
        """
        Checks if the variable exists in the dataBox and if it does, adds it to the list of response variables
        :param response: a string referring to a variable in the weather data
        :return: self
        """
        if response in self.dataBox.index:
            self.response_variables.append(response)
        return self

    def add_predictor(self, predictor):
        """
        Checks if the variable exists in the dataBox and if it does, adds it to the list of predictors
        :param predictor: a string referring to a variable in the weather data
        :return: self
        """
        if predictor in self.dataBox:
            self.predictor_variables.append(predictor)
        return self


    def auto_regressive(self, lag, scenarios, trim_data=True):
        """
        Fit a vector AR model to the data according to the given lag and compute a number of samples
        :param lag: Determines AR order of the model
        :param scenarios: Number of possible weather scenarios to return
        :return:
        """
        end_horizon = datetime.strptime(self.simulation_time, '%m/%d/%Y %H:%M') + timedelta(hours=self.horizon)
        start_horizon = end_horizon - timedelta(hours=2*self.horizon)

        if trim_data:
            data = self.dataBox.ix[start_horizon.strftime('%m/%d/%Y %H:%M').replace(' 0', ' ').lstrip('0').replace('/0', '/'): end_horizon.strftime('%m/%d/%Y %H:%M').replace(' 0', ' ').lstrip('0').replace('/0', '/')]
        else:
            data = self.dataBox

        data.index = pandas.to_datetime(data.index, format='%m/%d/%Y %H:%M')
        for field in data:
            if field not in self.predictor_variables:
                data = data.drop(field, axis=1)

        model = sm.tsa.VAR(data)
        print(len(data))
        fit = model.fit(maxlags=lag, ic='aic')
        predictions = {}

        std = data.std()

        for i in range(0, scenarios):
            # First, mutate the data with a random number within +-1 standard deviation:
            copy = data
            for field in copy:
                copy[field] = copy[field].apply(lambda x: np.random.normal(loc=x, scale=std[field]))
            predictions[i] = fit.forecast(data.values[-lag:], self.horizon)
        return predictions


    def fit_glm(self):
        """Fits a Bayes Ridge Regression model to the data

        Before this method will work, at least one response and one predictor must be set
        """
        if not self.dataBox:
            return None
        return self
