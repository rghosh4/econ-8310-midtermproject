# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:10:53 2023

@author: rghosh
"""

import pandas as pd
import plotly.express as px
import os
from pygam import LinearGAM, s, f, l
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_percentage_error
import plotly.io as pio
pio.renderers.default='browser'

os.chdir("C:/Users/rghosh/Documents/Graduate Curicullum/Spring'23/ECON 8310/Midterm Project/archive")


# Load the dataset
orders = pd.read_csv('olist_orders_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
category_names = pd.read_csv('product_category_name_translation.csv')

# Merge the datasets
merged_data = pd.merge(pd.merge(orders, order_items, on='order_id'), products, on='product_id')

# Filter out rows with 'canceled' and 'unavailable' order statuses
merged_data = merged_data[(merged_data['order_status'] != 'canceled') & (merged_data['order_status'] != 'unavailable')]

# Convert order_purchase_timestamp to datetime format
merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])

# Check the time span and orders
daily_orders_prelim = merged_data.set_index('order_purchase_timestamp').resample('D')['order_id'].count()

# Create the preliminary time series plot using Plotly Express
fig0 = px.line(daily_orders_prelim, x=daily_orders_prelim.index, y=daily_orders_prelim.values, labels={'x': 'Date', 'y': 'Number of Orders'})
fig0.show()



# Filter the data between 1st January 2017 and 31st August 2018
start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2018-08-31')
filtered_data = merged_data[(merged_data['order_purchase_timestamp'] >= start_date) & (merged_data['order_purchase_timestamp'] <= end_date)]



# Get the daily number of orders
daily_orders = filtered_data.set_index('order_purchase_timestamp').resample('D')['order_id'].count()


# Create the time series plot using Plotly Express
fig = px.line(daily_orders, x=daily_orders.index, y=daily_orders.values, labels={'x': 'Date', 'y': 'Number of Orders'})
fig.show()



# Create a new DataFrame with daily orders, month, and day of the week columns
daily_orders_df = pd.DataFrame({'orders': daily_orders,
                                'year': daily_orders.index.year,
                                'month': daily_orders.index.month,
                                'day_of_week': daily_orders.index.dayofweek})


x = daily_orders_df[['year','month', 'day_of_week']]
y = daily_orders_df['orders']

gam = LinearGAM(s(0) + s(1) + f(2))
gam = gam.gridsearch(x.values, y)
gam.summary()

# Name each figure
titles = ['year','month', 'dayofweek']

# Create the subplots in a single-row grid
fig = tools.make_subplots(rows=1, cols=3, subplot_titles=titles)
# Dictate the size of the figure, title, etc.
fig['layout'].update(height=500, width=1000, title='pyGAM', showlegend=False)

# Loop over the titles, and create the corresponding figures
for i, title in enumerate(titles):
    # Create the grid over which to estimate the effect of parameters
    XX = gam.generate_X_grid(term=i)
    # Calculate the value and 95% confidence intervals for each parameter
    # This will become the expected effect on the dependent variable for a given value of x
    pdep, confi = gam.partial_dependence(term=i, width=.95)
    
    # Create the effect and confidence interval traces (there are 3 total)
    trace = go.Scatter(x=XX[:,i], y=pdep, mode='lines', name='Effect')
    ci1 = go.Scatter(x = XX[:,i], y=confi[:,0], line=dict(dash='dash', color='grey'), name='95% CI')
    ci2 = go.Scatter(x = XX[:,i], y=confi[:,1], line=dict(dash='dash', color='grey'), name='95% CI')

    # Add each of the three traces to the figure in the relevant grid position
    fig.append_trace(trace, 1, i+1)
    fig.append_trace(ci1, 1, i+1)
    fig.append_trace(ci2, 1, i+1)

#Plot the figure
py.iplot(fig)


# Get the fitted values for the data used to fit the model
fitted_values = gam.predict(x)
train_mape=mean_absolute_percentage_error(daily_orders, fitted_values)
print(f'Training set MAPE: {train_mape:.2}')
fitted_values_series = pd.Series(fitted_values, index=daily_orders.index)


# Define the start date for the next 90 days
start_date = pd.to_datetime('2018-09-01')

# Define the end date for the next 90 days
end_date = start_date + pd.Timedelta(days=89)

# Create a DataFrame with a date range for the next 90 days
next_90_days = pd.date_range(start=start_date, end=end_date, freq='D')

# Convert the date range to a DataFrame
next_90_days_df = pd.DataFrame({'date': next_90_days})

# Add columns for month and day of the week
next_90_days_df['year'] = next_90_days_df['date'].dt.year
next_90_days_df['month'] = next_90_days_df['date'].dt.month
next_90_days_df['day_of_week'] = next_90_days_df['date'].dt.dayofweek

# Use the pyGAM model to make predictions for the next 90 days
predictions = gam.predict(next_90_days_df[['year','month', 'day_of_week']])

# Convert the predictions to a pandas Series with dates as the index
predictions_series = pd.Series(predictions, index=next_90_days_df['date'])


# Combine the historical data and the predictions into a single DataFrame
combined_data = pd.concat([daily_orders, fitted_values_series, predictions_series], axis=1)
combined_data.columns = ['orders', 'fitted_values', 'predictions']

# Create the time series plot using Plotly Express
fig = px.line(combined_data, x=combined_data.index, y=['orders', 'fitted_values', 'predictions'], labels={'x': 'Date', 'value': 'Number of Orders'})

# Update the line colors
fig.update_traces(line_color='#636EFA', selector=dict(name='orders'))
fig.update_traces(line_color='#EF553B', selector=dict(name='fitted_values'))
fig.update_traces(line_color='#00CC96', selector=dict(name='predictions'))

# Add a title to the plot
fig.update_layout(title='Daily Number of Orders (Historical, Fitted, and Predicted)')

# Show the plot
fig.show()







