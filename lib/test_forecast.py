from forecast import Forecast

weather = Forecast()

weather.load_tmy('../data/golden.csv')
weather.auto_regressive(4, '1/1/1900 04:00')

weather.persist()
