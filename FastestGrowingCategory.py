# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:55:50 2023

@author: rghosh
"""

import pandas as pd
import plotly.express as px
import os
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing
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

# Filter data to include only orders made between January 2017 and August 2018
merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])
filtered_data = merged_data[(merged_data['order_purchase_timestamp'] >= '2017-01-01') & (merged_data['order_purchase_timestamp'] <= '2018-08-31')]

# Aggregate data by product category and month and calculate total sales for each category in each month
sales_data = filtered_data.groupby(['product_category_name_english', pd.Grouper(key='order_purchase_timestamp', freq='M')])['order_item_id'].count().reset_index()
sales_data = sales_data.rename(columns={'order_item_id': 'total_sales'})

# Calculate monthly sales growth rate for each category
sales_data['monthly_growth'] = sales_data.groupby('product_category_name_english')['total_sales'].pct_change()


# Calculate average monthly sales growth rate for each category
avg_growth_data = sales_data.groupby('product_category_name_english')['monthly_growth'].mean().reset_index()

# Identify the category with the most number of positive monthly growth values
# positive_growth_data = sales_data[sales_data['monthly_growth'] > 0]
# category_with_most_positive_growth = positive_growth_data.groupby('product_category_name_english')['monthly_growth'].count(). reset_index()
# Sort the data by monthly_growth, in descending order
# category_with_most_positive_growth.sort_values(by='monthly_growth', ascending=False)
# fastest_growing_category = category_with_most_positive_growth.loc[category_with_most_positive_growth['monthly_growth'].idxmax(), 'product_category_name_english']


# Identify the category with the highest average monthly sales growth rate
fastest_growing_category = avg_growth_data.loc[avg_growth_data['monthly_growth'].idxmax(), 'product_category_name_english']



print(f"The fastest growing category based on average monthly sales growth between January 2017 and August 2018 is {fastest_growing_category}.")

# Filter out top 5 categories from avg_growth_data
top_5_categories = avg_growth_data.nlargest(5, 'monthly_growth')['product_category_name_english'].tolist()
sales_data_top_5 = sales_data[sales_data['product_category_name_english'].isin(top_5_categories)]


# Set color for 'computers' category
colors = ['lightgrey' if cat != 'computers' else 'black' for cat in sales_data_top_5['product_category_name_english'].unique()]

# Plot time series with highlighted 'computers' category
fig = px.line(sales_data_top_5, x='order_purchase_timestamp', y='monthly_growth', color='product_category_name_english',
              title='Top 5 Fastest Growing Product Categories', color_discrete_sequence=colors)
fig.update_traces(line=dict(width=2))
fig.show()


# Filter out the 'computers' category and select the relevant columns
computers_data = sales_data[sales_data['product_category_name_english'] == 'computers'][['order_purchase_timestamp', 'total_sales']]

# Set the month start date as the index
computers_data.set_index(pd.to_datetime(computers_data['order_purchase_timestamp'].dt.to_period('M').dt.start_time), inplace=True)
computers_data.index = pd.DatetimeIndex(computers_data.index).to_period('M')
computers_data=computers_data[['total_sales']]

#Split the data into training and test sets
train_data = computers_data.loc[:'2018-07-31']
test_data = computers_data.loc['2018-08-01':]

#Fit an additive exponential smoothing model to the training data
computers_exp_model = ExponentialSmoothing(train_data,trend='add',damped_trend=True).fit()

# Make forecasts for the test set
forecast = computers_exp_model.forecast(steps=4)

actual = computers_data.iloc[-1]['total_sales']
# first value of the forecast is for August 2018, which is our test month 
# loosely based on leave one out validation approach
predicted = forecast[0]
train_mape = mean_absolute_percentage_error(computers_data['total_sales'][:-1], computers_exp_model.fittedvalues)
test_mape = mean_absolute_percentage_error([actual], [predicted])

print(f'Training set MAPE: {train_mape:.2}')
print(f'Test set MAPE for August 2018: {test_mape:.2}\n')


# Prepare for plotting

fig_data = pd.concat([computers_data, computers_exp_model.fittedvalues, forecast], axis=1)
fig_data.columns = ['Actual', 'Fitted', 'Forecasted']

# Convert PeriodIndex to DatetimeIndex
fig_data.index = fig_data.index.to_timestamp()

fig = px.line(fig_data, x=fig_data.index, y=fig_data.columns,
              labels={'value': 'Sales Count', 'variable': 'Type'})
fig.update_layout(xaxis_title='Month', title_text='Computers-Fastest Growing Product Category' )
fig.show()

