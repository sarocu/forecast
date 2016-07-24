from forecast import Forecast

weather = Forecast()

# weather.clean_tmy('../data/golden.csv')
weather.load_tmy('../data/golden_clean.csv')

weather.add_predictor('Dry-bulb (C)').add_predictor('Dew-point (C)')

predictions = weather.auto_regressive(12, 2)
print(predictions)

weather.persist()
