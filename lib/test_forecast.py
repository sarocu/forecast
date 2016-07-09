from forecast import forecast

weather = forecast()

weather.loadTMY('../data/golden.csv')
weather.persist()
