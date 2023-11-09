# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:30:11 2023

@author: rghosh
"""

import pandas as pd
import plotly.express as px
import os
from statsmodels.tsa.api import ExponentialSmoothing
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


# Check for NAs in product_category_name column and Impute them with 'Unknown'
merged_data['product_category_name'].isna().sum()
merged_data['product_category_name'] = merged_data['product_category_name'].fillna('Unknown')

# Convert product category names to English using category_names table
# Keep portugese name if English name is not found 
merged_data = pd.merge(merged_data, category_names, on='product_category_name', how='left')
merged_data['product_category_name_english'] = merged_data['product_category_name_english'].fillna(merged_data['product_category_name'])



# Group the data by product category and count the number of rows
category_sales = merged_data.groupby('product_category_name_english')['order_id'].count().reset_index()

# Rename the count column
category_sales = category_sales.rename(columns={'order_id': 'sales_count'})

# Sort the data by sales count, in descending order
category_sales_sorted = category_sales.sort_values(by='sales_count', ascending=False)

# Get the top 3 categories, based on the highest number of sales
top_categories = category_sales_sorted.head(3)['product_category_name_english'].tolist()

# Print the top 3 categories
print('The top 3 best-selling product categories based on sales count are:')
print(top_categories)

# Get the top 10 categories, based on the highest number of sales
top_10_categories = category_sales_sorted.head(10)['product_category_name_english'].tolist()

# Convert order_purchase_timestamp to datetime format
merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])


# Group the data by product category and count the number of rows
category_monthly_sales = merged_data.groupby(['product_category_name_english', pd.to_datetime(merged_data['order_purchase_timestamp']).dt.to_period('M')])['order_item_id'].count().reset_index()

# Rename the count column
category_monthly_sales = category_monthly_sales.rename(columns={'order_item_id': 'monthly_sales'})


# Filter the top 10 categories
top_10_categories = category_monthly_sales[category_monthly_sales['product_category_name_english'].isin(top_10_categories[:10])]



# Convert order_purchase_timestamp to timestamp format
top_10_categories['order_purchase_timestamp'] = top_10_categories['order_purchase_timestamp'].dt.to_timestamp()

# Plot the top 10 categories as time series
fig = px.line(top_10_categories, x='order_purchase_timestamp', y='monthly_sales', color='product_category_name_english')

# Set the title
fig.update_layout(title='Monthly Sales by Product Category')

# Update the legend to show only the top 3 categories in bold
for i, trace in enumerate(fig.data):
    if trace.name not in top_categories:
        fig.data[i].line.color = 'lightgrey'
    else:
        fig.data[i].line.width = 3

fig.show()


# Filter merged_data to include only the top 3 categories
merged_data_top3 = merged_data[merged_data['product_category_name_english'].isin(top_categories)]

# Create a time series dataframe for each category, with monthly sales frequency as the value
category_ts_list = []
    
for category in top_categories:
    category_data = merged_data_top3[(merged_data_top3['product_category_name_english'] == category) & 
                                     (merged_data_top3['order_purchase_timestamp'] >= '2017-01-01') & 
                                     (merged_data_top3['order_purchase_timestamp'] <= '2018-08-31')]
    category_sales_ts = pd.DataFrame({'month': pd.date_range(start='2017-01-01', end='2018-08-01', freq='MS')})
    category_sales_ts['sales_count'] = category_data.groupby(pd.Grouper(key='order_purchase_timestamp', freq='MS'))['order_item_id'].count().values
    category_sales_ts.set_index('month', inplace=True)
    category_ts_list.append(category_sales_ts)    

# Fit the exponential smoothing model to each time series from Jan 2017 to July 2018
for i, category in enumerate(top_categories):
    model = ExponentialSmoothing(category_ts_list[i].iloc[:19], trend='add',damped_trend=True, use_boxcox= True).fit()

    # Forecast the sales count for August 2018 and thereby succeeding 3 months using the fitted model
    forecast = model.forecast(steps=4)

    # Calculate the Mean Absolute Percentage Error (MAPE) for the forecasted value
    actual = category_ts_list[i].iloc[19]['sales_count']
    #first value of the forecast is for August 2018, which is our test month 
    #loosely based on leave one out validation approach
    predicted = forecast[0]
    train_mape = mean_absolute_percentage_error(category_ts_list[i]['sales_count'][:19], model.fittedvalues)
    test_mape = mean_absolute_percentage_error([actual], [predicted])

    print(f'Category: {category}')
    print(f'Training set MAPE: {train_mape:.2}')
    print(f'Test set MAPE for August 2018: {test_mape:.2}\n')
    
    # Plot the actual and predicted time series plots for each category
    fig = px.line(title=category)
    fig.add_scatter(x=category_ts_list[i].index, y=category_ts_list[i]['sales_count'], mode='lines', name='Actual', line=dict(color='blue'))
    fig.add_scatter(x=model.fittedvalues.index, y=model.fittedvalues, mode='lines', name='Fitted', line=dict(color='orange'))
    fig.add_scatter(x=forecast.index, y=forecast, name='Forecasted', line=dict(color='green'))

    fig.update_layout(xaxis_title='Month', yaxis_title='Sales Count')
    fig.show()

