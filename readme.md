# forecast.py
Stochastic weather generator written in Python. Designed to use TMY3 data to
genterate possible weather vectors over a specified horizon.

## Requirements:
* sklearn
* pandas

## Usage:
```python
# initialize:
weather = Forecast()

# load a TMY3 file:
weather.load_tmy('../data/golden.csv')

# ...

# persist the generator as a pickle:
weather.persist()
```