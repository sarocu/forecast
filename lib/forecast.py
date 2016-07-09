import json
import pickle
import numpy

import statsmodels.api as sm
from sklearn import linear_model
import pandas
from scipy import stats

class forecast:
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

    def persist(self, fileName='..\\data\\forecast.p'):
        """Simple pickle method, accepts a file name

        Keyword arguments:
        fileName -- a string for naming the pickle. Default = 'forecast.p'
        """
        pickle.dump(self, open(fileName, 'wb'))


    def loadTMY(self, data):
        """Parses a csv via pandas, expects a TMY3 file as 'data'

        A csv is parsed and stored in the variable self.dataBox, self is
        returned to support method chaining.

        Keyword arguments:
        data -- the filename of a csv containing hourly output of an energy model
        """
        self.dataBox = pandas.read_csv(data, parse_dates=True,  low_memory=False)
        return self

    def addResponse(self, response):
        pass

    def addPredictor(self, predictor):
        pass

    def fitGLM(self):
        """Fits a Bayes Ridge Regression model to the data

        Before this method will work, at least one response and one predictor must be set 
        """
        if self.dataBox == None:
            return None
        return self
