
# Restaurant sales forecasting using LSTM

Sales forecasting is an essential task for the management of a business. Being able to estimate the volume and demand of sales that a restaurant is going to have in the future will allow the owners to be prepared when they will need.

Deep learning can help us to discover the factors that determine the number of sales that a retail store will have in the future.

During this article we are going to use the information about the sales of an restaurant, weather data and macroeconomic data from the last two years in order to predict the amount of sales that it is going to have one days in advance.

Neural networks like Long Short-Term Memory (LSTM) recurrent neural networks are able to almost seamlessly model problems with multiple input variables.

This is a great benefit in time series forecasting, where classical linear methods can be difficult to adapt to multivariate or multiple input forecasting problems.

After completing this article, we will:

* Transform a raw dataset into something we can use for time series forecasting.
* Prepare data and fit an LSTM for a multivariate time series forecasting problem.
* Make a forecast and rescale the result back into the original units.


Let’s get started.

## Data analysis

In this article, we are going to use a Brazilian restaurant sales data, the data is anonymized, and the business is an indoor restaurant inside a company, the price is cheaper than the restaurants and food services outside in the neighborhood. Because of that, the owners of the restaurant believe that weather data (like rain) and macroeconomic data (like inflation) can influence in the sales volume. Another thing to be aware is that the restaurant are closed on Saturdays and Sundays, then there will be no sales on these days.

With that in mind, I extracted some weather data from the following sites: http://www.inmet.gov.br and some macroeconomic data from the https://ibge.gov.br/. You can check here the code I created to merge these data with the restaurant sales data. 

The complete feature list in the final raw data is as follows:

* **date**: the date in this row
* **year_month**: year and month of data in this row
* **year**: year of data in this row
* **month**: month of data in this row
* **inflation**: the inflation value on the month in this row
* **inf_accum**: the accumulated inflation value on the month in this row
* **quarter**: the quarter of data in this row
* **gdp**: the quarter country gdp in this row
* **precipitation_vol**: the precipitation volume in this row
* **max_temp**: the max temperature, in celsius, on the day in this row
* **min_temp**: the min temperature, in celsius, on the day in this row
* **humidity**: the humidity level in this row
* **wind_speed**: the windy speed in this row
* **min_sale:** the sale with max value on that day
* **max_sale**: the sale with max value on that day
* **total_sales**: the total value of sales on that day
* **total_invoices**: the total number of sales/invoices on that day
* **total_cpfs**: the total number of people how informed their id on that day
* **avg_tickect**: the avarage tickect of sales values per invoice on that day
* **week_day**: the week day of data in this row (0-6 -> monday-sunday)
* **holiday**: If that date is a holiday
* **day**: day of data in this row
* **week_day_str**: the week day in string format
* **after_holiday**: If is after a holiday
* **before_holiday**: If is before a holiday

We can use this data and frame a forecasting problem where, given the weather conditions, macroeconomic data   and total invoices/sales for prior days, we forecast the 'total_invoices' (sales volume) at the next day.

In this article we will using python and keras to develop our LSTM model

OK to start, we will load the libs.


```python
# Basic libs
import pandas as pd
from math import sqrt
import numpy
import datetime as dt

# Sklearn libs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Keras libs
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM

# Chart libs
import matplotlib as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Offline plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
import plotly.graph_objs as go
import plotly.io as pio

# Run this the command before at the start of notebook to plotly work
init_notebook_mode(connected=True)

def print_plot(fig, file_name):
    iplot(fig)
    pio.write_image(fig, file_name, scale=3)
```


<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script><script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window._Plotly) {require(['plotly'],function(plotly) {window._Plotly=plotly;});}</script>


Now we will read the dataset


```python
# Read the dataset
df = pd.read_csv('processed/sales.csv')
df['date']  = pd.to_datetime(df['date'])
```

First, we are going to study how the sales timeseries looks like. Lets plot a chart to see the data


```python
# Plot sales volume timeseries
data = [go.Scatter(x=df.date, y=df.total_invoices, marker=dict(color='#00b5bd'))]
layout=go.Layout(title="Sales volume per Day", xaxis={'title':'Sales volume'}, yaxis={'title':'Date'})
fig=go.Figure(data=data,layout=layout)
print_plot(fig, 'imgs/sales_vol_per_day.png')
```


<div id="9776fe00-ed69-4d2c-9825-831a5d351144" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("9776fe00-ed69-4d2c-9825-831a5d351144", [{"marker": {"color": "#00b5bd"}, "x": ["2018-11-29", "2018-11-28", "2018-11-27", "2018-11-26", "2018-11-25", "2018-11-24", "2018-11-23", "2018-11-22", "2018-11-21", "2018-11-20", "2018-11-19", "2018-11-18", "2018-11-17", "2018-11-16", "2018-11-15", "2018-11-14", "2018-11-13", "2018-11-12", "2018-11-11", "2018-11-10", "2018-11-09", "2018-11-08", "2018-11-07", "2018-11-06", "2018-11-05", "2018-11-04", "2018-11-03", "2018-11-02", "2018-11-01", "2018-10-31", "2018-10-30", "2018-10-29", "2018-10-28", "2018-10-27", "2018-10-26", "2018-10-25", "2018-10-24", "2018-10-23", "2018-10-22", "2018-10-21", "2018-10-20", "2018-10-19", "2018-10-18", "2018-10-17", "2018-10-16", "2018-10-15", "2018-10-14", "2018-10-13", "2018-10-12", "2018-10-11", "2018-10-10", "2018-10-09", "2018-10-08", "2018-10-07", "2018-10-06", "2018-10-05", "2018-10-04", "2018-10-03", "2018-10-02", "2018-10-01", "2018-09-30", "2018-09-29", "2018-09-28", "2018-09-27", "2018-09-26", "2018-09-25", "2018-09-24", "2018-09-23", "2018-09-22", "2018-09-21", "2018-09-20", "2018-09-19", "2018-09-18", "2018-09-17", "2018-09-16", "2018-09-15", "2018-09-14", "2018-09-13", "2018-09-12", "2018-09-11", "2018-09-10", "2018-09-09", "2018-09-08", "2018-09-07", "2018-09-06", "2018-09-05", "2018-09-04", "2018-09-03", "2018-09-02", "2018-09-01", "2018-08-31", "2018-08-30", "2018-08-29", "2018-08-28", "2018-08-27", "2018-08-26", "2018-08-25", "2018-08-24", "2018-08-23", "2018-08-22", "2018-08-21", "2018-08-20", "2018-08-19", "2018-08-18", "2018-08-17", "2018-08-16", "2018-08-15", "2018-08-14", "2018-08-13", "2018-08-12", "2018-08-11", "2018-08-10", "2018-08-09", "2018-08-08", "2018-08-07", "2018-08-06", "2018-08-05", "2018-08-04", "2018-08-03", "2018-08-02", "2018-08-01", "2018-07-31", "2018-07-30", "2018-07-29", "2018-07-28", "2018-07-27", "2018-07-26", "2018-07-25", "2018-07-24", "2018-07-23", "2018-07-22", "2018-07-21", "2018-07-20", "2018-07-19", "2018-07-18", "2018-07-17", "2018-07-16", "2018-07-15", "2018-07-14", "2018-07-13", "2018-07-12", "2018-07-11", "2018-07-10", "2018-07-09", "2018-07-08", "2018-07-07", "2018-07-06", "2018-07-05", "2018-07-04", "2018-07-03", "2018-07-02", "2018-07-01", "2018-06-30", "2018-06-29", "2018-06-28", "2018-06-27", "2018-06-26", "2018-06-25", "2018-06-24", "2018-06-23", "2018-06-22", "2018-06-21", "2018-06-20", "2018-06-19", "2018-06-18", "2018-06-17", "2018-06-16", "2018-06-15", "2018-06-14", "2018-06-13", "2018-06-12", "2018-06-11", "2018-06-10", "2018-06-09", "2018-06-08", "2018-06-07", "2018-06-06", "2018-06-05", "2018-06-04", "2018-06-03", "2018-06-02", "2018-06-01", "2018-05-31", "2018-05-30", "2018-05-29", "2018-05-28", "2018-05-27", "2018-05-26", "2018-05-25", "2018-05-24", "2018-05-23", "2018-05-22", "2018-05-21", "2018-05-20", "2018-05-19", "2018-05-18", "2018-05-17", "2018-05-16", "2018-05-15", "2018-05-14", "2018-05-13", "2018-05-12", "2018-05-11", "2018-05-10", "2018-05-09", "2018-05-08", "2018-05-07", "2018-05-06", "2018-05-05", "2018-05-04", "2018-05-03", "2018-05-02", "2018-05-01", "2018-04-30", "2018-04-29", "2018-04-28", "2018-04-27", "2018-04-26", "2018-04-25", "2018-04-24", "2018-04-23", "2018-04-22", "2018-04-21", "2018-04-20", "2018-04-19", "2018-04-18", "2018-04-17", "2018-04-16", "2018-04-15", "2018-04-14", "2018-04-13", "2018-04-12", "2018-04-11", "2018-04-10", "2018-04-09", "2018-04-08", "2018-04-07", "2018-04-06", "2018-04-05", "2018-04-04", "2018-04-03", "2018-04-02", "2018-04-01", "2018-03-31", "2018-03-30", "2018-03-29", "2018-03-28", "2018-03-27", "2018-03-26", "2018-03-25", "2018-03-24", "2018-03-23", "2018-03-22", "2018-03-21", "2018-03-20", "2018-03-19", "2018-03-18", "2018-03-17", "2018-03-16", "2018-03-15", "2018-03-14", "2018-03-13", "2018-03-12", "2018-03-11", "2018-03-10", "2018-03-09", "2018-03-08", "2018-03-07", "2018-03-06", "2018-03-05", "2018-03-04", "2018-03-03", "2018-03-02", "2018-03-01", "2018-02-28", "2018-02-27", "2018-02-26", "2018-02-25", "2018-02-24", "2018-02-23", "2018-02-22", "2018-02-21", "2018-02-20", "2018-02-19", "2018-02-18", "2018-02-17", "2018-02-16", "2018-02-15", "2018-02-14", "2018-02-13", "2018-02-12", "2018-02-11", "2018-02-10", "2018-02-09", "2018-02-08", "2018-02-07", "2018-02-06", "2018-02-05", "2018-02-04", "2018-02-03", "2018-02-02", "2018-02-01", "2018-01-31", "2018-01-30", "2018-01-29", "2018-01-28", "2018-01-27", "2018-01-26", "2018-01-25", "2018-01-24", "2018-01-23", "2018-01-22", "2018-01-21", "2018-01-20", "2018-01-19", "2018-01-18", "2018-01-17", "2018-01-16", "2018-01-15", "2018-01-14", "2018-01-13", "2018-01-12", "2018-01-11", "2018-01-10", "2018-01-09", "2018-01-08", "2018-01-07", "2018-01-06", "2018-01-05", "2018-01-04", "2018-01-03", "2018-01-02", "2018-01-01", "2017-12-31", "2017-12-30", "2017-12-29", "2017-12-28", "2017-12-27", "2017-12-26", "2017-12-25", "2017-12-24", "2017-12-23", "2017-12-22", "2017-12-21", "2017-12-20", "2017-12-19", "2017-12-18", "2017-12-17", "2017-12-16", "2017-12-15", "2017-12-14", "2017-12-13", "2017-12-12", "2017-12-11", "2017-12-10", "2017-12-09", "2017-12-08", "2017-12-07", "2017-12-06", "2017-12-05", "2017-12-04", "2017-12-03", "2017-12-02", "2017-12-01", "2017-11-30", "2017-11-29", "2017-11-28", "2017-11-27", "2017-11-26", "2017-11-25", "2017-11-24", "2017-11-23", "2017-11-22", "2017-11-21", "2017-11-20", "2017-11-19", "2017-11-18", "2017-11-17", "2017-11-16", "2017-11-15", "2017-11-14", "2017-11-13", "2017-11-12", "2017-11-11", "2017-11-10", "2017-11-09", "2017-11-08", "2017-11-07", "2017-11-06", "2017-11-05", "2017-11-04", "2017-11-03", "2017-11-02", "2017-11-01", "2017-10-31", "2017-10-30", "2017-10-29", "2017-10-28", "2017-10-27", "2017-10-26", "2017-10-25", "2017-10-24", "2017-10-23", "2017-10-22", "2017-10-21", "2017-10-20", "2017-10-19", "2017-10-18", "2017-10-17", "2017-10-16", "2017-10-15", "2017-10-14", "2017-10-13", "2017-10-12", "2017-10-11", "2017-10-10", "2017-10-09", "2017-10-08", "2017-10-07", "2017-10-06", "2017-10-05", "2017-10-04", "2017-10-03", "2017-10-02", "2017-10-01", "2017-09-30", "2017-09-29", "2017-09-28", "2017-09-27", "2017-09-26", "2017-09-25", "2017-09-24", "2017-09-23", "2017-09-22", "2017-09-21", "2017-09-20", "2017-09-19", "2017-09-18", "2017-09-17", "2017-09-16", "2017-09-15", "2017-09-14", "2017-09-13", "2017-09-12", "2017-09-11", "2017-09-10", "2017-09-09", "2017-09-08", "2017-09-07", "2017-09-06", "2017-09-05", "2017-09-04", "2017-09-03", "2017-09-02", "2017-09-01", "2017-08-31", "2017-08-30", "2017-08-29", "2017-08-28", "2017-08-27", "2017-08-26", "2017-08-25", "2017-08-24", "2017-08-23", "2017-08-22", "2017-08-21", "2017-08-20", "2017-08-19", "2017-08-18", "2017-08-17", "2017-08-16", "2017-08-15", "2017-08-14", "2017-08-13", "2017-08-12", "2017-08-11", "2017-08-10", "2017-08-09", "2017-08-08", "2017-08-07", "2017-08-06", "2017-08-05", "2017-08-04", "2017-08-03", "2017-08-02", "2017-08-01", "2017-07-31", "2017-07-30", "2017-07-29", "2017-07-28", "2017-07-27", "2017-07-26", "2017-07-25", "2017-07-24", "2017-07-23", "2017-07-22", "2017-07-21", "2017-07-20", "2017-07-19", "2017-07-18", "2017-07-17", "2017-07-16", "2017-07-15", "2017-07-14", "2017-07-13", "2017-07-12", "2017-07-11", "2017-07-10", "2017-07-09", "2017-07-08", "2017-07-07", "2017-07-06", "2017-07-05", "2017-07-04", "2017-07-03", "2017-07-02", "2017-07-01", "2017-06-30", "2017-06-29", "2017-06-28", "2017-06-27", "2017-06-26", "2017-06-25", "2017-06-24", "2017-06-23", "2017-06-22", "2017-06-21", "2017-06-20", "2017-06-19", "2017-06-18", "2017-06-17", "2017-06-16", "2017-06-15", "2017-06-14", "2017-06-13", "2017-06-12", "2017-06-11", "2017-06-10", "2017-06-09", "2017-06-08", "2017-06-07", "2017-06-06", "2017-06-05", "2017-06-04", "2017-06-03", "2017-06-02", "2017-06-01", "2017-05-31", "2017-05-30", "2017-05-29", "2017-05-28", "2017-05-27", "2017-05-26", "2017-05-25", "2017-05-24", "2017-05-23", "2017-05-22", "2017-05-21", "2017-05-20", "2017-05-19", "2017-05-18", "2017-05-17", "2017-05-16", "2017-05-15", "2017-05-14", "2017-05-13", "2017-05-12", "2017-05-11", "2017-05-10", "2017-05-09", "2017-05-08", "2017-05-07", "2017-05-06", "2017-05-05", "2017-05-04", "2017-05-03", "2017-05-02", "2017-05-01", "2017-04-30", "2017-04-29", "2017-04-28", "2017-04-27", "2017-04-26", "2017-04-25", "2017-04-24", "2017-04-23", "2017-04-22", "2017-04-21", "2017-04-20", "2017-04-19", "2017-04-18", "2017-04-17", "2017-04-16", "2017-04-15", "2017-04-14", "2017-04-13", "2017-04-12", "2017-04-11", "2017-04-10", "2017-04-09", "2017-04-08", "2017-04-07", "2017-04-06", "2017-04-05", "2017-04-04", "2017-04-03", "2017-04-02", "2017-04-01", "2017-03-31", "2017-03-30", "2017-03-29", "2017-03-28", "2017-03-27", "2017-03-26", "2017-03-25", "2017-03-24", "2017-03-23", "2017-03-22", "2017-03-21", "2017-03-20", "2017-03-19", "2017-03-18", "2017-03-17", "2017-03-16", "2017-03-15", "2017-03-14", "2017-03-13", "2017-03-12", "2017-03-11", "2017-03-10", "2017-03-09", "2017-03-08", "2017-03-07", "2017-03-06", "2017-03-05", "2017-03-04", "2017-03-03", "2017-03-02", "2017-03-01", "2017-02-28", "2017-02-27", "2017-02-26", "2017-02-25", "2017-02-24", "2017-02-23", "2017-02-22", "2017-02-21", "2017-02-20", "2017-02-19", "2017-02-18", "2017-02-17", "2017-02-16", "2017-02-15", "2017-02-14", "2017-02-13", "2017-02-12", "2017-02-11", "2017-02-10", "2017-02-09", "2017-02-08", "2017-02-07", "2017-02-06", "2017-02-05", "2017-02-04", "2017-02-03", "2017-02-02", "2017-02-01", "2017-01-31", "2017-01-30", "2017-01-29", "2017-01-28", "2017-01-27", "2017-01-26", "2017-01-25", "2017-01-24", "2017-01-23", "2017-01-22", "2017-01-21", "2017-01-20", "2017-01-19", "2017-01-18", "2017-01-17", "2017-01-16", "2017-01-15", "2017-01-14", "2017-01-13", "2017-01-12", "2017-01-11", "2017-01-10", "2017-01-09", "2017-01-08", "2017-01-07", "2017-01-06", "2017-01-05", "2017-01-04", "2017-01-03", "2017-01-02", "2017-01-01", "2016-12-31", "2016-12-30", "2016-12-29", "2016-12-28", "2016-12-27", "2016-12-26", "2016-12-25", "2016-12-24", "2016-12-23", "2016-12-22", "2016-12-21", "2016-12-20", "2016-12-19", "2016-12-18", "2016-12-17", "2016-12-16", "2016-12-15", "2016-12-14", "2016-12-13", "2016-12-12", "2016-12-11", "2016-12-10", "2016-12-09", "2016-12-08", "2016-12-07", "2016-12-06", "2016-12-05", "2016-12-04", "2016-12-03", "2016-12-02", "2016-12-01", "2016-11-30", "2016-11-29", "2016-11-28", "2016-11-27", "2016-11-26", "2016-11-25", "2016-11-24", "2016-11-23", "2016-11-22", "2016-11-21", "2016-11-20", "2016-11-19", "2016-11-18", "2016-11-17", "2016-11-16", "2016-11-15", "2016-11-14", "2016-11-13", "2016-11-12", "2016-11-11", "2016-11-10", "2016-11-09", "2016-11-08", "2016-11-07", "2016-11-06", "2016-11-05", "2016-11-04", "2016-11-03", "2016-11-02", "2016-11-01", "2016-10-31", "2016-10-30", "2016-10-29", "2016-10-28", "2016-10-27", "2016-10-26", "2016-10-25", "2016-10-24", "2016-10-23", "2016-10-22", "2016-10-21", "2016-10-20", "2016-10-19", "2016-10-18", "2016-10-17", "2016-10-16", "2016-10-15", "2016-10-14", "2016-10-13", "2016-10-12", "2016-10-11", "2016-10-10", "2016-10-09", "2016-10-08", "2016-10-07", "2016-10-06", "2016-10-05", "2016-10-04", "2016-10-03", "2016-10-02", "2016-10-01", "2016-09-30", "2016-09-29", "2016-09-28", "2016-09-27", "2016-09-26", "2016-09-25", "2016-09-24", "2016-09-23", "2016-09-22", "2016-09-21", "2016-09-20", "2016-09-19", "2016-09-18", "2016-09-17", "2016-09-16", "2016-09-15", "2016-09-14", "2016-09-13", "2016-09-12", "2016-09-11", "2016-09-10", "2016-09-09", "2016-09-08", "2016-09-07", "2016-09-06", "2016-09-05", "2016-09-04", "2016-09-03", "2016-09-02", "2016-09-01", "2016-08-31", "2016-08-30", "2016-08-29", "2016-08-28", "2016-08-27", "2016-08-26", "2016-08-25", "2016-08-24", "2016-08-23", "2016-08-22", "2016-08-21", "2016-08-20", "2016-08-19", "2016-08-18", "2016-08-17", "2016-08-16", "2016-08-15", "2016-08-14", "2016-08-13", "2016-08-12", "2016-08-11", "2016-08-10", "2016-08-09", "2016-08-08", "2016-08-07", "2016-08-06", "2016-08-05", "2016-08-04", "2016-08-03", "2016-08-02", "2016-08-01", "2016-07-31", "2016-07-30", "2016-07-29", "2016-07-28", "2016-07-27", "2016-07-26", "2016-07-25", "2016-07-24", "2016-07-23", "2016-07-22", "2016-07-21", "2016-07-20", "2016-07-19", "2016-07-18", "2016-07-17", "2016-07-16", "2016-07-15", "2016-07-14", "2016-07-13", "2016-07-12", "2016-07-11", "2016-07-10", "2016-07-09", "2016-07-08", "2016-07-07", "2016-07-06", "2016-07-05", "2016-07-04", "2016-07-03", "2016-07-02", "2016-07-01", "2016-06-30", "2016-06-29", "2016-06-28", "2016-06-27", "2016-06-26", "2016-06-25", "2016-06-24", "2016-06-23", "2016-06-22", "2016-06-21", "2016-06-20"], "y": [61.0, 124.0, 115.0, 165.0, null, null, 138.0, 173.0, 183.0, 219.0, 232.0, null, null, null, null, 182.0, 244.0, 194.0, null, null, 245.0, 323.0, 299.0, 211.0, 236.0, null, null, null, 230.0, 57.0, 111.0, 136.0, null, null, 102.0, 141.0, 154.0, 205.0, 142.0, null, null, 128.0, 186.0, 276.0, 256.0, 243.0, null, null, null, 213.0, 244.0, 296.0, 282.0, null, null, 214.0, 322.0, 308.0, 289.0, 216.0, null, null, 118.0, 125.0, 130.0, 139.0, 209.0, null, null, 139.0, 215.0, 235.0, 254.0, 218.0, null, null, 96.0, 220.0, 258.0, 275.0, 215.0, null, null, null, 202.0, 281.0, 256.0, 306.0, null, null, 114.0, 96.0, 94.0, 125.0, 135.0, null, null, 121.0, 150.0, 180.0, 190.0, 192.0, null, null, 122.0, 179.0, 206.0, 260.0, 231.0, null, null, 152.0, 225.0, 252.0, 276.0, 280.0, null, null, 250.0, 374.0, 350.0, 280.0, 294.0, null, null, 264.0, 311.0, 307.0, 283.0, 276.0, null, null, 271.0, 295.0, 335.0, 322.0, 306.0, null, null, 305.0, 297.0, 302.0, 151.0, 318.0, null, null, 75.0, 284.0, 326.0, 289.0, 228.0, null, null, 231.0, 250.0, 146.0, 301.0, 311.0, null, null, 84.0, 262.0, 245.0, 346.0, 305.0, null, null, 321.0, 345.0, 545.0, 425.0, 356.0, null, null, 306.0, 324.0, 414.0, 355.0, 391.0, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 50.0, 388.0, null, null, 284.0, 318.0, 307.0, null, null, null, null, 269.0, 311.0, 292.0, 282.0, 312.0, null, null, 294.0, 341.0, 352.0, 329.0, 245.0, null, null, null, 237.0, 347.0, 363.0, 342.0, null, null, 301.0, 289.0, 405.0, 352.0, 338.0, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 266.0, 291.0, 315.0, null, null, 258.0, 336.0, 343.0, 313.0, 320.0, null, null, 261.0, 320.0, null, null, null, null, null, 265.0, 285.0, 291.0, 290.0, 299.0, null, null, 295.0, 312.0, 255.0, 263.0, 308.0, null, null, 255.0, 323.0, 334.0, 334.0, 359.0, null, null, 284.0, 320.0, 298.0, 311.0, 307.0, null, null, 250.0, 283.0, 288.0, 311.0, 308.0, null, null, 270.0, 285.0, 263.0, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 221.0, 310.0, null, null, 265.0, 368.0, 384.0, 365.0, 380.0, null, null, 314.0, 354.0, 351.0, 374.0, 379.0, null, null, 274.0, null, 273.0, 303.0, 351.0, null, null, 247.0, 367.0, 348.0, 361.0, 348.0, null, null, 289.0, 355.0, null, 367.0, 401.0, null, null, 424.0, 441.0, 445.0, 391.0, 398.0, null, null, null, null, 310.0, 382.0, 352.0, null, null, 320.0, 332.0, 300.0, 267.0, 298.0, null, null, 280.0, 285.0, 264.0, 334.0, 343.0, null, null, null, null, 373.0, 364.0, 312.0, null, null, 276.0, 320.0, 321.0, 341.0, 320.0, null, null, 231.0, 319.0, 313.0, 368.0, 358.0, null, null, 269.0, 304.0, 306.0, 345.0, 300.0, null, null, 288.0, 318.0, 301.0, 300.0, 313.0, null, null, null, null, 320.0, 318.0, 312.0, null, null, 287.0, 332.0, 279.0, 303.0, 309.0, null, null, 278.0, 320.0, 307.0, 281.0, 320.0, null, null, 265.0, 303.0, 336.0, 295.0, 338.0, null, null, 226.0, 274.0, 310.0, 299.0, 358.0, null, null, 266.0, 297.0, 291.0, 288.0, 317.0, null, null, 265.0, 283.0, 265.0, 234.0, 334.0, null, null, 246.0, 239.0, 236.0, 258.0, 283.0, null, null, 251.0, 290.0, 291.0, 339.0, 337.0, null, null, 297.0, 295.0, 315.0, 289.0, 278.0, null, null, 170.0, 268.0, 302.0, 310.0, 338.0, null, null, 245.0, 324.0, 330.0, 319.0, 318.0, null, null, null, null, 294.0, 311.0, 312.0, null, null, 303.0, 376.0, 357.0, 304.0, 387.0, null, null, 297.0, 366.0, 310.0, 355.0, 378.0, null, null, 281.0, 319.0, 312.0, 320.0, 353.0, null, null, 339.0, 352.0, 434.0, 451.0, 421.0, null, null, 257.0, 314.0, 362.0, 351.0, 349.0, null, null, 303.0, 341.0, 350.0, 368.0, null, null, null, null, 307.0, 333.0, 344.0, 326.0, null, null, null, 266.0, 322.0, 342.0, 314.0, null, null, null, 32.0, null, 375.0, 375.0, null, null, 269.0, 290.0, 344.0, 354.0, 360.0, null, null, 287.0, 328.0, 330.0, 362.0, 374.0, null, null, 314.0, 337.0, 313.0, 378.0, 360.0, null, null, 325.0, 323.0, 163.0, 267.0, 68.0, null, null, 262.0, 360.0, 343.0, 360.0, 352.0, null, null, 286.0, 310.0, null, null, null, null, null, 292.0, 303.0, 283.0, 322.0, 359.0, null, null, 269.0, 361.0, 305.0, 302.0, 357.0, null, null, 319.0, 396.0, 366.0, 379.0, 347.0, null, null, 247.0, 340.0, 280.0, 276.0, 321.0, null, null, 315.0, 299.0, 328.0, 311.0, 344.0, null, null, 280.0, 290.0, 288.0, 278.0, 293.0, null, null, 290.0, 268.0, 264.0, 288.0, 271.0, null, null, 252.0, 270.0, 254.0, 258.0, null, null, null, null, null, null, null, null, null, null, null, null, 192.0, 281.0, 345.0, null, null, 190.0, 319.0, 312.0, 316.0, 347.0, null, null, 248.0, 332.0, 353.0, 362.0, 369.0, null, null, 274.0, 291.0, null, 322.0, 295.0, null, null, 273.0, 294.0, 290.0, 319.0, 339.0, null, null, 237.0, 296.0, 303.0, null, null, null, null, 282.0, 395.0, 360.0, 341.0, 332.0, null, null, 316.0, 354.0, null, 390.0, 409.0, null, null, 264.0, 346.0, 249.0, 298.0, 315.0, null, null, 293.0, 343.0, 313.0, 302.0, 314.0, null, null, 262.0, 315.0, null, 304.0, 335.0, null, null, 292.0, 315.0, 322.0, 311.0, 340.0, null, null, 245.0, 287.0, 278.0, 313.0, 324.0, null, null, 257.0, 274.0, 287.0, 286.0, 341.0, null, null, 267.0, 329.0, 343.0, 357.0, 336.0, null, null, 274.0, 356.0, null, 311.0, 372.0, null, null, 233.0, 272.0, 361.0, 361.0, 300.0, null, null, 250.0, 315.0, 319.0, 302.0, 338.0, null, null, 240.0, 264.0, 316.0, 305.0, 258.0, null, null, 243.0, 298.0, 275.0, 300.0, 277.0, null, null, 194.0, 27.0, null, 18.0, 22.0, null, null, 19.0, 17.0, 32.0, 26.0, 42.0, null, null, 38.0, 34.0, 19.0, 15.0, 118.0, null, null, 1.0, null, null, null, null, null, null, null, null, null, null, 43.0, null, null, 28.0, 28.0, 36.0, null, 104.0, null, null, 214.0, 252.0, 255.0, 155.0, 161.0], "type": "scatter", "uid": "d390c203-da2f-4cca-b455-be807aa6bd98"}], {"title": "Sales volume per Day", "xaxis": {"title": "Sales volume"}, "yaxis": {"title": "Date"}}, {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("9776fe00-ed69-4d2c-9825-831a5d351144"));});</script>


As we can see, there is some missing values on Saturdays, Sundays, and holidays, the restaurant is closed these days, and the others missing values can be just really missing, like the values of march 2018, they are all missing. There are some outliers too, but we will try to fit the model even with outliers. Now let's remove the missing values


```python
# Remove missing values
df.dropna(inplace=True)
```

Secondly, we are going to study how the average sales and rain volume are distributed by month. The next chart shows the average number of sales/rains by month.


```python
# Group sales volume by year-month
sum_by_m = df.groupby(['year', 'month'], as_index=False)['total_invoices'].sum()
sum_by_m['idx'] = sum_by_m.apply(lambda x: str(int(x.year)) + '-' + str(int(x.month)),  axis=1)
# Get the avg by month
avg_by_m = sum_by_m.groupby(['month'], as_index=False)['total_invoices'].mean()
# Create bars
trace1 = go.Scatter(x=avg_by_m.month,y=avg_by_m.total_invoices, name='Sales Vol.', marker=dict(color='#00b5bd'))

# Group sales volume by year-month
rain_avg_by_m = df.groupby(['month'], as_index=False)['precipitation_vol'].mean()
# Create bars
trace2 = go.Scatter(x=rain_avg_by_m.month,y=rain_avg_by_m.precipitation_vol, name='Rain Vol.', yaxis='y2', marker=dict(color='#4BAF49'))

# Create Chart data and layout
data = [trace1, trace2]
layout = go.Layout(
    title='Avg sales volume vs Avg rain volume',
    yaxis=dict(
        title='yaxis title',
        titlefont=dict(
            color='#00b5bd'
        ),
        tickfont=dict(
            color='#00b5bd'
        ),
    ),
    yaxis2=dict(
        title='yaxis2 title',
        titlefont=dict(
            color='#4BAF49'
        ),
        tickfont=dict(
            color='#4BAF49'
        ),
        overlaying='y',
        side='right'
    )
)

# Plot chart
fig = go.Figure(data=data, layout=layout)
print_plot(fig, 'imgs/avg_sales_vs_rain.png')
```


<div id="2fa78601-adb7-432f-9b93-cbe5cb1523e2" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("2fa78601-adb7-432f-9b93-cbe5cb1523e2", [{"marker": {"color": "#00b5bd"}, "name": "Sales Vol.", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "y": [6123.5, 5443.5, 6442.0, 5477.0, 4483.5, 4566.333333333333, 3972.6666666666665, 5670.666666666667, 5367.666666666667, 5715.666666666667, 5243.666666666667, 4435.0], "type": "scatter", "uid": "fc36be0c-85ea-484c-9b6b-d3a3f4cc1a83"}, {"marker": {"color": "#4BAF49"}, "name": "Rain Vol.", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "y": [4.130952380952381, 9.16, 7.752380952380953, 5.122857142857143, 1.4222222222222223, 0.0, 0.0, 0.49705882352941183, 1.1966101694915257, 3.9596774193548385, 7.392592592592593, 11.292857142857143], "yaxis": "y2", "type": "scatter", "uid": "26d80478-fbbb-4a1b-ba9a-f1db1863eef3"}], {"title": "Avg sales volume vs Avg rain volume", "yaxis": {"tickfont": {"color": "#00b5bd"}, "title": "yaxis title", "titlefont": {"color": "#00b5bd"}}, "yaxis2": {"overlaying": "y", "side": "right", "tickfont": {"color": "#4BAF49"}, "title": "yaxis2 title", "titlefont": {"color": "#4BAF49"}}}, {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("2fa78601-adb7-432f-9b93-cbe5cb1523e2"));});</script>


As we can see, most of the sales are made in January and October. And the month seasonality can influence the volume of sales, Its look like there some relation between the sales volume and the rain volume, this can help the model to predict sales volumes using weather data
The following scatter plot show indications that there is a positive relation between the inflation data with the volume of sales

```python
# Create Scatter plot
data = [go.Scatter(x=df.inflation, y=df.total_invoices, marker=dict(color='#4BAF49'), mode = 'markers')]
layout=go.Layout(title="Sales volume vs Inflation", xaxis={'title':'Inflation'}, yaxis={'title':'Sales volume'})
fig=go.Figure(data=data,layout=layout)
print_plot(fig, 'imgs/sales_vol_vs_inf.png')
```


<div id="cb8fd97c-bdfd-4feb-8aa6-e557aee8ec0c" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("cb8fd97c-bdfd-4feb-8aa6-e557aee8ec0c", [{"marker": {"color": "#4BAF49"}, "mode": "markers", "x": [-0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, -0.21, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, -0.09, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 0.4, 0.4, 0.4, 0.4, 0.4, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, -0.23, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35], "y": [61.0, 124.0, 115.0, 165.0, 138.0, 173.0, 183.0, 219.0, 232.0, 182.0, 244.0, 194.0, 245.0, 323.0, 299.0, 211.0, 236.0, 230.0, 57.0, 111.0, 136.0, 102.0, 141.0, 154.0, 205.0, 142.0, 128.0, 186.0, 276.0, 256.0, 243.0, 213.0, 244.0, 296.0, 282.0, 214.0, 322.0, 308.0, 289.0, 216.0, 118.0, 125.0, 130.0, 139.0, 209.0, 139.0, 215.0, 235.0, 254.0, 218.0, 96.0, 220.0, 258.0, 275.0, 215.0, 202.0, 281.0, 256.0, 306.0, 114.0, 96.0, 94.0, 125.0, 135.0, 121.0, 150.0, 180.0, 190.0, 192.0, 122.0, 179.0, 206.0, 260.0, 231.0, 152.0, 225.0, 252.0, 276.0, 280.0, 250.0, 374.0, 350.0, 294.0, 264.0, 311.0, 307.0, 283.0, 276.0, 271.0, 335.0, 322.0, 306.0, 305.0, 297.0, 302.0, 151.0, 318.0, 75.0, 284.0, 326.0, 289.0, 228.0, 231.0, 250.0, 146.0, 301.0, 311.0, 84.0, 262.0, 245.0, 346.0, 305.0, 321.0, 345.0, 545.0, 425.0, 356.0, 306.0, 324.0, 414.0, 355.0, 391.0, 50.0, 388.0, 284.0, 318.0, 307.0, 269.0, 311.0, 292.0, 282.0, 312.0, 294.0, 341.0, 352.0, 329.0, 245.0, 237.0, 347.0, 363.0, 342.0, 301.0, 289.0, 405.0, 352.0, 338.0, 266.0, 291.0, 315.0, 258.0, 336.0, 343.0, 313.0, 320.0, 261.0, 320.0, 265.0, 285.0, 291.0, 290.0, 299.0, 295.0, 312.0, 255.0, 263.0, 308.0, 255.0, 323.0, 334.0, 334.0, 359.0, 284.0, 320.0, 298.0, 311.0, 307.0, 250.0, 283.0, 288.0, 311.0, 308.0, 270.0, 285.0, 263.0, 221.0, 310.0, 265.0, 368.0, 384.0, 365.0, 380.0, 314.0, 354.0, 351.0, 374.0, 379.0, 274.0, 273.0, 303.0, 351.0, 247.0, 367.0, 348.0, 361.0, 348.0, 289.0, 355.0, 367.0, 401.0, 424.0, 441.0, 445.0, 391.0, 398.0, 310.0, 382.0, 352.0, 320.0, 332.0, 300.0, 267.0, 298.0, 280.0, 285.0, 264.0, 334.0, 343.0, 373.0, 364.0, 312.0, 276.0, 320.0, 321.0, 341.0, 320.0, 231.0, 319.0, 313.0, 368.0, 358.0, 269.0, 304.0, 306.0, 345.0, 300.0, 288.0, 318.0, 301.0, 300.0, 313.0, 320.0, 318.0, 312.0, 287.0, 332.0, 279.0, 303.0, 309.0, 278.0, 320.0, 307.0, 281.0, 320.0, 265.0, 303.0, 336.0, 295.0, 338.0, 226.0, 274.0, 310.0, 299.0, 358.0, 266.0, 297.0, 291.0, 288.0, 317.0, 265.0, 283.0, 265.0, 234.0, 334.0, 246.0, 239.0, 236.0, 258.0, 283.0, 251.0, 290.0, 291.0, 339.0, 337.0, 297.0, 295.0, 315.0, 289.0, 278.0, 170.0, 268.0, 302.0, 310.0, 338.0, 245.0, 324.0, 330.0, 319.0, 318.0, 294.0, 311.0, 312.0, 303.0, 376.0, 357.0, 304.0, 387.0, 297.0, 366.0, 310.0, 355.0, 378.0, 281.0, 319.0, 312.0, 320.0, 353.0, 339.0, 352.0, 434.0, 451.0, 421.0, 257.0, 314.0, 362.0, 351.0, 349.0, 303.0, 341.0, 350.0, 368.0, 307.0, 333.0, 344.0, 326.0, 266.0, 322.0, 342.0, 314.0, 32.0, 375.0, 375.0, 269.0, 290.0, 344.0, 354.0, 360.0, 287.0, 328.0, 330.0, 362.0, 374.0, 314.0, 337.0, 313.0, 378.0, 360.0, 325.0, 323.0, 163.0, 267.0, 68.0, 262.0, 343.0, 360.0, 352.0, 286.0, 310.0, 292.0, 303.0, 283.0, 322.0, 359.0, 269.0, 361.0, 305.0, 302.0, 357.0, 319.0, 396.0, 366.0, 379.0, 347.0, 247.0, 340.0, 280.0, 276.0, 321.0, 315.0, 299.0, 328.0, 311.0, 344.0, 280.0, 290.0, 288.0, 278.0, 293.0, 290.0, 268.0, 264.0, 288.0, 271.0, 252.0, 270.0, 254.0, 258.0, 192.0, 281.0, 345.0, 190.0, 319.0, 312.0, 316.0, 347.0, 248.0, 332.0, 353.0, 362.0, 369.0, 274.0, 291.0, 322.0, 295.0, 273.0, 294.0, 290.0, 319.0, 339.0, 237.0, 296.0, 303.0, 282.0, 395.0, 360.0, 341.0, 332.0, 316.0, 354.0, 390.0, 409.0, 264.0, 346.0, 249.0, 298.0, 315.0, 293.0, 343.0, 313.0, 302.0, 314.0, 262.0, 315.0, 304.0, 335.0, 292.0, 315.0, 322.0, 311.0, 340.0, 245.0, 287.0, 278.0, 313.0, 324.0, 257.0, 274.0, 287.0, 286.0, 341.0, 267.0, 329.0, 343.0, 357.0, 336.0, 274.0, 356.0, 311.0, 372.0, 233.0, 272.0, 361.0, 361.0, 300.0, 250.0, 315.0, 319.0, 302.0, 338.0, 240.0, 264.0, 316.0, 305.0, 258.0, 243.0, 298.0, 275.0, 300.0, 277.0, 194.0, 27.0, 18.0, 22.0, 19.0, 17.0, 32.0, 26.0, 42.0, 38.0, 34.0, 19.0, 15.0, 118.0, 1.0, 43.0, 28.0, 28.0, 36.0, 104.0, 214.0, 252.0, 255.0, 155.0, 161.0], "type": "scatter", "uid": "c7789a0d-a743-4b3c-8795-1e7a73f03d16"}], {"title": "Sales volume vs Inflation", "xaxis": {"title": "Inflation"}, "yaxis": {"title": "Sales volume"}}, {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("cb8fd97c-bdfd-4feb-8aa6-e557aee8ec0c"));});</script>


Ok, let's take a look on the avarage sales volume per week day, Its visible that the there is a downtrend between monday to friday, this pattern can help with the predictive power of the model as well.


```python
sum_by_w = df.groupby(['week_day'], as_index=False)['total_invoices'].mean()
data = [go.Bar(x=sum_by_w.week_day, y=sum_by_w.total_invoices, marker=dict(color='#00b5bd'))]

layout=go.Layout(title="Avg. Sales volume per weekday (Mon-Fri)", xaxis={'title':'Week day'}, yaxis={'title':'Sales volume'})
fig=go.Figure(data=data,layout=layout)
print_plot(fig, 'imgs/avg_sales_vl_weekday.png')
```


<div id="a082c18c-c4ac-480c-9565-6ee91c66a311" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("a082c18c-c4ac-480c-9565-6ee91c66a311", [{"marker": {"color": "#00b5bd"}, "x": [0, 1, 2, 3, 4], "y": [297.8378378378378, 293.1727272727273, 287.688679245283, 281.87619047619046, 243.32673267326732], "type": "bar", "uid": "c17cd99f-1c2f-4f3a-a2a4-05a3823e1367"}], {"title": "Avg. Sales volume per weekday (Mon-Fri)", "xaxis": {"title": "Week day"}, "yaxis": {"title": "Sales volume"}}, {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("a082c18c-c4ac-480c-9565-6ee91c66a311"));});</script>


Now, cheking the Autocorrelatin plot: We can see that for each 5 lags there is a seasonal pattern as the peacks is higher than for the other lags. The dashed blue lines indicate whether the correlations are significantly different from zero. Reinforcing the week influence in the sales volume


```python
df.total_invoices.fillna(0, inplace=True)
acf = plot_acf(df.total_invoices, lags = 50)
acf.show()
#plt.title("ACF")
#pacf = plot_pacf(df.total_invoices, lags = 20)
#plt.title("PACF")
#pacf.show()
```

    /Users/marcus/anaconda3/envs/nlp/lib/python3.6/site-packages/matplotlib/figure.py:445: UserWarning:
    
    Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.
    



![png](output_19_1.png)


## Data preparation

The next step is to select and prepare the variables that we are going to use, the *total_invoices* is the varible we will try to predict. The following code will transform the dataset into a supervised learning problem, that will:
* Select the right variables
* Add previeus 14 days sales volume(total_invoices) for each row, using a window function
* One-hot encode some variables


```python
# Fill missing values with 0
df_processed = df.fillna(0)

# Removing saturday and sunday, the days that the restaurant are closed
df_processed = df_processed[df_processed['week_day']  != 5]
df_processed = df_processed[df_processed['week_day']  != 6]

# Select variables
df_processed = df_processed[[
    'total_invoices', 
    #'year', 
    'month',
    'day',
    'week_day', 
    'holiday', 
    'after_holiday', 
    'before_holiday', 
    'max_temp', 
    'min_temp',
    'precipitation_vol',
    'humidity',
    'gdp',
    'quarter',
    'inflation',
    'inf_accum'
]]

# Add previous sales volume 
for i in range(1,15):
    df_processed['total_invoices' + '-' + str(i)] = df_processed.total_invoices.shift(-1*i)
    
# One-hot encode week day and month variables
df_processed = pd.get_dummies(df_processed, prefix_sep="_", columns=['week_day'])

# Removing last 25 rows
df_processed = df_processed[:-25]

# Print head
df_processed.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_invoices</th>
      <th>month</th>
      <th>day</th>
      <th>holiday</th>
      <th>after_holiday</th>
      <th>before_holiday</th>
      <th>max_temp</th>
      <th>min_temp</th>
      <th>precipitation_vol</th>
      <th>humidity</th>
      <th>...</th>
      <th>total_invoices-10</th>
      <th>total_invoices-11</th>
      <th>total_invoices-12</th>
      <th>total_invoices-13</th>
      <th>total_invoices-14</th>
      <th>week_day_0</th>
      <th>week_day_1</th>
      <th>week_day_2</th>
      <th>week_day_3</th>
      <th>week_day_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>61.0</td>
      <td>11</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>28.6</td>
      <td>17.4</td>
      <td>0.0</td>
      <td>60.25</td>
      <td>...</td>
      <td>244.0</td>
      <td>194.0</td>
      <td>245.0</td>
      <td>323.0</td>
      <td>299.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>124.0</td>
      <td>11</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>26.6</td>
      <td>17.4</td>
      <td>0.0</td>
      <td>69.00</td>
      <td>...</td>
      <td>194.0</td>
      <td>245.0</td>
      <td>323.0</td>
      <td>299.0</td>
      <td>211.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>115.0</td>
      <td>11</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25.8</td>
      <td>17.6</td>
      <td>0.1</td>
      <td>74.00</td>
      <td>...</td>
      <td>245.0</td>
      <td>323.0</td>
      <td>299.0</td>
      <td>211.0</td>
      <td>236.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>165.0</td>
      <td>11</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25.1</td>
      <td>17.7</td>
      <td>28.8</td>
      <td>81.50</td>
      <td>...</td>
      <td>323.0</td>
      <td>299.0</td>
      <td>211.0</td>
      <td>236.0</td>
      <td>230.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>138.0</td>
      <td>11</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27.4</td>
      <td>18.3</td>
      <td>0.0</td>
      <td>75.25</td>
      <td>...</td>
      <td>299.0</td>
      <td>211.0</td>
      <td>236.0</td>
      <td>230.0</td>
      <td>57.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>



We must split the prepared dataset into the train and test sets. I will use 75:25 ratio with shuffl false, because its a time series, and we want the test set to be the last 25% of the data. All features are normalized to help the network to converge.


```python
# define parameters
values = df_processed.values    
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)
# Detach X and Y 
x,y = values[:, 1:], values[:, 0]    
# Split into train, test sets
train_x, test_x = train_test_split(x, test_size=0.25, shuffle=False)
train_y, test_y = train_test_split(y, test_size=0.25, shuffle=False)
# Print the arrays shapes
print('X shape: ', train_x.shape)
print('Y shape: ', train_y.shape)
```

    X shape:  (381, 32)
    Y shape:  (381,)


## Now we can create and train our LSTM model.

We will define the LSTM with 7 neurons in the first hidden layer and 40 neurons in the output layer for predicting sales volume. The input shape will be 1 time step with 33 features.

We will use the Mean Absolute Error (MAE) loss function and the Adam for optmization.

There will be a dropout rate of 15%, which will add a probability of setting each input of the layer to zero, this is a cheap way to regularize a neural network.

The model will be fit for 381 training epochs with a batch size of 15. 


```python
# function to build and train the model
def build_model(train_x, train_y, test_x, test_y):
    # Get the num of features
    n_features = train_x.shape[1]
    
    # Reshape the input vector to a 3D format for the LSTM layer
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

    # Set model fit params
    verbose, epochs, batch_size = 1, 70, 15
    n_outputs = 1
    
    # define model 36,80
    model = Sequential()
    model.add(LSTM(7, activation='relu', input_shape=(1,n_features), dropout=0.15))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam')
    
    # fit network
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), 
                        epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model,history

# Train the model
model,history  = build_model(train_x, train_y, test_x, test_y)
```

    Train on 381 samples, validate on 127 samples
    Epoch 1/70
    381/381 [==============================] - 2s 5ms/step - loss: 0.3808 - val_loss: 0.2623
    Epoch 2/70
    381/381 [==============================] - 0s 307us/step - loss: 0.1539 - val_loss: 0.1119
    Epoch 3/70
    381/381 [==============================] - 0s 313us/step - loss: 0.1255 - val_loss: 0.0956
    Epoch 4/70
    381/381 [==============================] - 0s 318us/step - loss: 0.1072 - val_loss: 0.0807
    Epoch 5/70
    381/381 [==============================] - 0s 347us/step - loss: 0.1016 - val_loss: 0.0823
    Epoch 6/70
    381/381 [==============================] - 0s 355us/step - loss: 0.0974 - val_loss: 0.0673
    Epoch 7/70
    381/381 [==============================] - 0s 318us/step - loss: 0.0920 - val_loss: 0.0634
    Epoch 8/70
    381/381 [==============================] - 0s 370us/step - loss: 0.0912 - val_loss: 0.0642
    Epoch 9/70
    381/381 [==============================] - 0s 352us/step - loss: 0.0887 - val_loss: 0.0613
    Epoch 10/70
    381/381 [==============================] - 0s 359us/step - loss: 0.0847 - val_loss: 0.0611
    Epoch 11/70
    381/381 [==============================] - 0s 343us/step - loss: 0.0862 - val_loss: 0.0647
    Epoch 12/70
    381/381 [==============================] - 0s 323us/step - loss: 0.0893 - val_loss: 0.0697
    Epoch 13/70
    381/381 [==============================] - 0s 537us/step - loss: 0.0908 - val_loss: 0.0605
    Epoch 14/70
    381/381 [==============================] - 0s 369us/step - loss: 0.0880 - val_loss: 0.0664
    Epoch 15/70
    381/381 [==============================] - 0s 383us/step - loss: 0.0857 - val_loss: 0.0608
    Epoch 16/70
    381/381 [==============================] - 0s 367us/step - loss: 0.0847 - val_loss: 0.0621
    Epoch 17/70
    381/381 [==============================] - 0s 374us/step - loss: 0.0839 - val_loss: 0.0600
    Epoch 18/70
    381/381 [==============================] - 0s 388us/step - loss: 0.0844 - val_loss: 0.0598
    Epoch 19/70
    381/381 [==============================] - 0s 487us/step - loss: 0.0838 - val_loss: 0.0612
    Epoch 20/70
    381/381 [==============================] - 0s 378us/step - loss: 0.0789 - val_loss: 0.0681
    Epoch 21/70
    381/381 [==============================] - 0s 340us/step - loss: 0.0847 - val_loss: 0.0576
    Epoch 22/70
    381/381 [==============================] - 0s 343us/step - loss: 0.0844 - val_loss: 0.0604
    Epoch 23/70
    381/381 [==============================] - 0s 375us/step - loss: 0.0830 - val_loss: 0.0584
    Epoch 24/70
    381/381 [==============================] - 0s 374us/step - loss: 0.0779 - val_loss: 0.0582
    Epoch 25/70
    381/381 [==============================] - 0s 384us/step - loss: 0.0777 - val_loss: 0.0594
    Epoch 26/70
    381/381 [==============================] - 0s 426us/step - loss: 0.0789 - val_loss: 0.0603
    Epoch 27/70
    381/381 [==============================] - 0s 372us/step - loss: 0.0785 - val_loss: 0.0603
    Epoch 28/70
    381/381 [==============================] - 0s 365us/step - loss: 0.0780 - val_loss: 0.0602
    Epoch 29/70
    381/381 [==============================] - 0s 532us/step - loss: 0.0776 - val_loss: 0.0604
    Epoch 30/70
    381/381 [==============================] - 0s 480us/step - loss: 0.0773 - val_loss: 0.0637
    Epoch 31/70
    381/381 [==============================] - 0s 365us/step - loss: 0.0768 - val_loss: 0.0570
    Epoch 32/70
    381/381 [==============================] - 0s 331us/step - loss: 0.0760 - val_loss: 0.0574
    Epoch 33/70
    381/381 [==============================] - 0s 410us/step - loss: 0.0749 - val_loss: 0.0632
    Epoch 34/70
    381/381 [==============================] - 0s 432us/step - loss: 0.0765 - val_loss: 0.0565
    Epoch 35/70
    381/381 [==============================] - 0s 386us/step - loss: 0.0761 - val_loss: 0.0606
    Epoch 36/70
    381/381 [==============================] - 0s 370us/step - loss: 0.0787 - val_loss: 0.0581
    Epoch 37/70
    381/381 [==============================] - 0s 412us/step - loss: 0.0781 - val_loss: 0.0577
    Epoch 38/70
    381/381 [==============================] - 0s 372us/step - loss: 0.0744 - val_loss: 0.0562
    Epoch 39/70
    381/381 [==============================] - 0s 409us/step - loss: 0.0743 - val_loss: 0.0585
    Epoch 40/70
    381/381 [==============================] - 0s 380us/step - loss: 0.0736 - val_loss: 0.0624
    Epoch 41/70
    381/381 [==============================] - 0s 383us/step - loss: 0.0747 - val_loss: 0.0598
    Epoch 42/70
    381/381 [==============================] - 0s 395us/step - loss: 0.0737 - val_loss: 0.0581
    Epoch 43/70
    381/381 [==============================] - 0s 590us/step - loss: 0.0720 - val_loss: 0.0608
    Epoch 44/70
    381/381 [==============================] - 0s 341us/step - loss: 0.0749 - val_loss: 0.0582
    Epoch 45/70
    381/381 [==============================] - 0s 357us/step - loss: 0.0749 - val_loss: 0.0572
    Epoch 46/70
    381/381 [==============================] - 0s 380us/step - loss: 0.0762 - val_loss: 0.0580
    Epoch 47/70
    381/381 [==============================] - 0s 361us/step - loss: 0.0747 - val_loss: 0.0566
    Epoch 48/70
    381/381 [==============================] - 0s 402us/step - loss: 0.0734 - val_loss: 0.0610
    Epoch 49/70
    381/381 [==============================] - 0s 380us/step - loss: 0.0767 - val_loss: 0.0643
    Epoch 50/70
    381/381 [==============================] - 0s 360us/step - loss: 0.0725 - val_loss: 0.0680
    Epoch 51/70
    381/381 [==============================] - 0s 364us/step - loss: 0.0756 - val_loss: 0.0589
    Epoch 52/70
    381/381 [==============================] - 0s 358us/step - loss: 0.0750 - val_loss: 0.0609
    Epoch 53/70
    381/381 [==============================] - 0s 402us/step - loss: 0.0739 - val_loss: 0.0574
    Epoch 54/70
    381/381 [==============================] - 0s 374us/step - loss: 0.0719 - val_loss: 0.0610
    Epoch 55/70
    381/381 [==============================] - 0s 454us/step - loss: 0.0738 - val_loss: 0.0651
    Epoch 56/70
    381/381 [==============================] - 0s 434us/step - loss: 0.0717 - val_loss: 0.0617
    Epoch 57/70
    381/381 [==============================] - 0s 328us/step - loss: 0.0709 - val_loss: 0.0602
    Epoch 58/70
    381/381 [==============================] - 0s 374us/step - loss: 0.0722 - val_loss: 0.0600
    Epoch 59/70
    381/381 [==============================] - 0s 382us/step - loss: 0.0697 - val_loss: 0.0656
    Epoch 60/70
    381/381 [==============================] - 0s 347us/step - loss: 0.0728 - val_loss: 0.0615
    Epoch 61/70
    381/381 [==============================] - 0s 368us/step - loss: 0.0698 - val_loss: 0.0595
    Epoch 62/70
    381/381 [==============================] - 0s 336us/step - loss: 0.0746 - val_loss: 0.0610
    Epoch 63/70
    381/381 [==============================] - 0s 352us/step - loss: 0.0696 - val_loss: 0.0579
    Epoch 64/70
    381/381 [==============================] - 0s 378us/step - loss: 0.0706 - val_loss: 0.0584
    Epoch 65/70
    381/381 [==============================] - 0s 357us/step - loss: 0.0742 - val_loss: 0.0582
    Epoch 66/70
    381/381 [==============================] - 0s 369us/step - loss: 0.0754 - val_loss: 0.0578
    Epoch 67/70
    381/381 [==============================] - 0s 355us/step - loss: 0.0703 - val_loss: 0.0598
    Epoch 68/70
    381/381 [==============================] - 0s 546us/step - loss: 0.0660 - val_loss: 0.0591
    Epoch 69/70
    381/381 [==============================] - 0s 340us/step - loss: 0.0756 - val_loss: 0.0594
    Epoch 70/70
    381/381 [==============================] - 0s 323us/step - loss: 0.0722 - val_loss: 0.0605


Finally, we keep track of both the training and test loss during training by setting the validation_data argument in the fit() function. At the end of the run both the training and test loss are plotted.


```python
# Create train, test loss plot
trace1 = go.Scatter(y=history.history['loss'], name='Train', marker=dict(color='#00b5bd'))
trace2 = go.Scatter(y=history.history['val_loss'], name='Test', marker=dict(color='#4BAF49'))
data = [trace1, trace2]

layout=go.Layout(title="Train/Test loss over epochs", xaxis={'title':'Loss'}, yaxis={'title':'Epochs'})
fig=go.Figure(data=data,layout=layout)
print_plot(fig, 'imgs/train_test_loss.png')
```


<div id="c7dacfc4-c618-40c8-89f0-28a5dae23896" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("c7dacfc4-c618-40c8-89f0-28a5dae23896", [{"marker": {"color": "#00b5bd"}, "name": "Train", "y": [0.3807693234109503, 0.15390977908776501, 0.12547459993071444, 0.10723777138811397, 0.10157168357391057, 0.09740974355166353, 0.09197980183081364, 0.09119771288957183, 0.0887233905904875, 0.08474665727671676, 0.08615884026439172, 0.08932051570044727, 0.09084287526335304, 0.08803481145167914, 0.08574243590939702, 0.08465443921136105, 0.08386732285886299, 0.08439641132012127, 0.0838102267131092, 0.07885096506692293, 0.08474174470413388, 0.08444032527682349, 0.08297749233292782, 0.07790146162366773, 0.07768639341348738, 0.07890737432194507, 0.07845725544442342, 0.07798313666281738, 0.07755530291185604, 0.07734699424092226, 0.07683021980007802, 0.07600431695697814, 0.07486548395020755, 0.07652513841240424, 0.0760792374141573, 0.078719442226286, 0.07813024503274227, 0.07442189565324407, 0.07434953655314258, 0.07363318188453284, 0.07468507039969362, 0.07373781179583917, 0.07199206659350339, 0.07494877536935131, 0.07491616473540547, 0.07621559342296105, 0.07465012928866964, 0.07343865713969928, 0.07674133657353131, 0.07251103678498212, 0.07560291276203365, 0.0749656649262417, 0.07393381325161363, 0.07190896913878561, 0.07378571383713738, 0.07168498670491646, 0.0709410224725881, 0.07215152085414082, 0.06969422638768286, 0.07284932865167228, 0.06981856202868027, 0.07461969924020016, 0.06955052005845731, 0.0706142709245832, 0.07423064740389351, 0.07538922142794752, 0.07025112025439739, 0.06600803832017531, 0.0756470480361792, 0.0721923546997581], "type": "scatter", "uid": "ee0e1990-a0bd-4dea-973d-23b7371e7a68"}, {"marker": {"color": "#4BAF49"}, "name": "Test", "y": [0.26231102070470497, 0.1119426041841507, 0.09563279392447059, 0.08071082586965223, 0.08233361799768575, 0.06732845095198924, 0.0633869623164023, 0.0641963875845192, 0.06133725218416199, 0.06108589206389555, 0.06468214933562466, 0.0697040606260769, 0.06051991439945116, 0.06640462914672424, 0.06079336730983314, 0.06210870665358746, 0.05998339520136672, 0.05981396382131914, 0.061168151885623065, 0.06808636836179598, 0.05756407496556053, 0.060448967800365655, 0.05839285725273016, 0.05817404848442772, 0.05936045576561624, 0.060256317052550204, 0.060251280342734706, 0.06024738185577036, 0.060360476550624126, 0.06366100520130218, 0.057023709053246996, 0.05742791737979791, 0.0631979030710975, 0.05647060247211475, 0.060620409913185076, 0.058082127166310636, 0.057694942184437915, 0.056164273024192, 0.05848595719989829, 0.062428409717683715, 0.05984316645996777, 0.05811445643817346, 0.060754246261762825, 0.058191013397780926, 0.05718288541309477, 0.057970596008061426, 0.05661211954796408, 0.060975324875843805, 0.064273056904162, 0.06798503788437431, 0.05894601963988439, 0.060948421899962614, 0.05735182864811477, 0.06101588288864752, 0.06510288353393397, 0.06165294707056106, 0.06017318244759492, 0.05996038189788503, 0.06564385860454379, 0.061478194846646995, 0.05946139756619461, 0.06100399727661779, 0.057912656289386, 0.05839638538130625, 0.058233173813406876, 0.05781955711954222, 0.05975451643072714, 0.05908864853888985, 0.0594150689760531, 0.06050857514377654], "type": "scatter", "uid": "222bdf91-e5cd-4cc4-b482-f899c2239b34"}], {"title": "Train/Test loss over epochs", "xaxis": {"title": "Loss"}, "yaxis": {"title": "Epochs"}}, {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("c7dacfc4-c618-40c8-89f0-28a5dae23896"));});</script>


## Evalutate the model

After the model is fit, we can forecast for the entire test dataset.

We combine the forecast with the test dataset and invert the scaling. We also invert scaling on the test dataset with the expected 'total_invoices' values.

With forecasts and actual values in their original scale, we can then calculate an error score for the model. In this case, we calculate the Root Mean Squared Error (RMSE) that gives error in the same units as the variable itself.


```python
# make a prediction
test_input = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
yhat = model.predict(test_input)
#yhat = yhat.reshape(yhat.shape[0],)
print('yhat shape: ', yhat.shape)
print('y shape: ', test_y.shape)
print('x shape: ', test_x.shape)
# invert scaling for forecast
inv_yhat = numpy.concatenate((yhat, test_x), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
inv_y = numpy.concatenate((test_y.reshape(test_y.shape[0],1), test_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.2f' % rmse)
```

    yhat shape:  (127, 1)
    y shape:  (127,)
    x shape:  (127, 32)
    Test RMSE: 39.04


We can see that the model achieves a reasonable RMSE of **39.04**, Now we can see, in follwing code, a plot showing the real vs predicted values:


```python
# Let's check the predictions
trace1 = go.Scatter(y=inv_y, name='Real', marker=dict(color='#00b5bd'))
trace2 = go.Scatter(y=inv_yhat, name='Predicted', marker=dict(color='#4BAF49'))
data = [trace1, trace2]
layout=go.Layout(title="Sales volume Real vs Predicted", xaxis={'title':'Days'}, yaxis={'title':'Sales volume'})
fig=go.Figure(data=data,layout=layout)
print_plot(fig, 'imgs/real_vs_predicted.png')
```


<div id="1d4cb95f-9fdd-4735-b158-8cdddab4f3ba" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("1d4cb95f-9fdd-4735-b158-8cdddab4f3ba", [{"marker": {"color": "#00b5bd"}, "name": "Real", "y": [359.0, 269.0, 361.0, 305.0, 302.0, 357.0, 319.0, 396.0000305175781, 366.0, 379.0, 347.0, 247.0, 339.9999694824219, 280.0, 275.9999694824219, 321.0, 315.0, 299.0, 328.0, 311.0, 344.0, 280.0, 290.0, 288.0, 278.0, 293.0, 290.0, 268.0000305175781, 264.0, 288.0, 271.0, 252.0, 270.0, 254.00001525878906, 258.0, 192.0, 281.0, 345.0, 190.00001525878906, 319.0, 312.0, 316.0000305175781, 347.0, 248.0, 332.0000305175781, 353.0, 362.0, 369.0, 274.0, 291.0, 322.0, 295.0, 273.0, 294.0, 290.0, 319.0, 339.0, 237.0, 296.0, 303.0, 282.0, 395.0, 360.0, 341.0, 332.0000305175781, 316.0000305175781, 354.0, 390.0, 409.0, 264.0, 346.0, 249.0, 298.0, 315.0, 293.0, 343.0, 313.0, 302.0, 314.0, 262.0, 315.0, 304.0, 335.0, 291.9999694824219, 315.0, 322.0, 311.0, 339.9999694824219, 245.0, 287.0, 278.0, 313.0, 323.9999694824219, 257.0, 274.0, 287.0, 286.0, 341.0, 267.0, 329.0, 343.0, 357.0, 336.0, 274.0, 355.9999694824219, 311.0, 371.9999694824219, 233.0, 272.0, 361.0, 361.0, 300.0000305175781, 249.99998474121094, 315.0, 319.0, 302.0, 338.0, 240.0, 264.0, 316.0000305175781, 305.0, 258.0, 243.0, 298.0, 275.0, 300.0000305175781, 277.0], "type": "scatter", "uid": "7d325f27-3f33-454e-9883-3f2c6703d217"}, {"marker": {"color": "#4BAF49"}, "name": "Predicted", "y": [341.221435546875, 293.5426025390625, 325.35418701171875, 333.8768310546875, 346.2271423339844, 358.92047119140625, 306.109619140625, 339.8941955566406, 357.64111328125, 352.7998962402344, 344.9698181152344, 295.58203125, 327.36151123046875, 337.44610595703125, 324.6900634765625, 329.71356201171875, 282.68487548828125, 319.3027648925781, 327.4781494140625, 335.6392517089844, 331.6879577636719, 278.8442077636719, 314.86199951171875, 330.4131164550781, 327.31927490234375, 336.03790283203125, 285.0097961425781, 314.1287536621094, 325.79229736328125, 319.4796142578125, 326.89215087890625, 277.88775634765625, 313.8270568847656, 331.7955017089844, 323.9250793457031, 325.2098693847656, 333.5535888671875, 346.0062561035156, 300.0242919921875, 332.3910217285156, 357.0578308105469, 355.9677429199219, 363.7078857421875, 307.0633544921875, 342.447021484375, 367.1009216308594, 355.7101135253906, 363.1329040527344, 309.7084655761719, 339.9791259765625, 313.957275390625, 334.78424072265625, 278.9073181152344, 314.92913818359375, 329.1236267089844, 331.4697570800781, 348.7334899902344, 295.98779296875, 329.4521484375, 367.46844482421875, 309.2557067871094, 344.82623291015625, 371.9847717285156, 355.15740966796875, 372.2510986328125, 305.08953857421875, 360.0462646484375, 356.0507507324219, 326.22479248046875, 267.5635681152344, 292.1799011230469, 306.0977783203125, 317.3939208984375, 336.09442138671875, 285.8543395996094, 309.6275939941406, 317.7956237792969, 322.90447998046875, 336.80816650390625, 286.0816345214844, 337.1382141113281, 327.98419189453125, 347.6319885253906, 300.332275390625, 330.80718994140625, 367.27587890625, 361.0289001464844, 348.0666809082031, 210.55560302734375, 255.51316833496094, 278.1169128417969, 286.2138366699219, 295.4079284667969, 228.08853149414062, 272.2198486328125, 295.2568359375, 305.2046203613281, 310.4231872558594, 259.1163024902344, 296.1867980957031, 312.01715087890625, 306.6505432128906, 324.95220947265625, 273.3956298828125, 318.90167236328125, 305.9277038574219, 324.34954833984375, 271.90802001953125, 312.70953369140625, 278.3963928222656, 277.4961242675781, 289.43450927734375, 243.3527069091797, 268.0745544433594, 281.5680847167969, 279.19024658203125, 282.82086181640625, 214.20477294921875, 251.3252410888672, 285.1829833984375, 268.71295166015625, 265.4419860839844, 225.0809783935547, 233.94386291503906, 254.45094299316406, 225.7471923828125, 199.90614318847656], "type": "scatter", "uid": "84c8fa9c-a628-4c95-ae6e-31adee7ecbe8"}], {"title": "Sales volume Real vs Predicted", "xaxis": {"title": "Days"}, "yaxis": {"title": "Sales volume"}}, {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("1d4cb95f-9fdd-4735-b158-8cdddab4f3ba"));});</script>


## Conclusion
During this article we have developed a predictive model, using LSTM, that can help restaurants to determine the number of sales that they are going to make in the future, using supervised learning aproach, with weather and macroeconomics data.

## References
* https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/]
* https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
* https://blog.cambridgespark.com/robust-one-hot-encoding-in-python-3e29bfcec77e
* http://barnesanalytics.com/basics-of-arima-models-with-statsmodels-in-python

