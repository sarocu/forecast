# forecast.py
Stochastic weather generator written in Python. Designed to use TMY3 data to
generate possible weather vectors over a specified horizon.

## Requirements:
* sklearn
* pandas
* statsmodels

## Usage:
```python
# initialize:
weather = Forecast() # optionally set the horizon, Forecast(48)

# if necessary, clean a TMY:
weather.clean_tmy('../data/golden.csv')

# load a clean TMY3 file:
weather.load_tmy('../data/golden_clean.csv')

# add variables to the model:
weather.add_predictor('Dry-bulb (C)').add_predictor('Dew-point (C)')

# generate a weather scenario which starts at the current simulation 
# time, weather.simulation_time, and runs through the horizon. 
predictions = weather.auto_regressive(lag=1, scenarios=100)

# persist the generator as a pickle:
weather.persist()
```