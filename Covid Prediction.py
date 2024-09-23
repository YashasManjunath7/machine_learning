# Import necessary libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Step 1: Data Preparation
# Load COVID-19 case data (JHU dataset)
# Download 'time_series_covid19_confirmed_global.csv' from JHU repository: 
# https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
df = pd.read_csv('time_series_covid19_confirmed_global.csv')

# Aggregate data for a single country (for example, 'India')
df = df[df['Country/Region'] == 'India'].drop(['Province/State', 'Lat', 'Long'], axis=1).T
df.columns = ['Cases']  # Rename the column
df = df[1:]  # Remove the first row (since it's the country name)

# Reset the index and format the data for Prophet (date as 'ds', cases as 'y')
df.reset_index(inplace=True)
df.columns = ['ds', 'y']  # Prophet requires 'ds' (date) and 'y' (target variable)

# Convert 'ds' to datetime
df['ds'] = pd.to_datetime(df['ds'])

# Step 2: FB Prophet Model Setup
# Initialize Prophet model with yearly and weekly seasonality
model = Prophet(
    yearly_seasonality=True,  # Detect yearly seasonality
    weekly_seasonality=True,  # Detect weekly seasonality
    daily_seasonality=False   # Daily seasonality is unnecessary
)

# Optional: Add holidays or known changepoints (e.g., lockdown dates)
lockdowns = pd.DataFrame({
    'holiday': 'lockdown',
    'ds': pd.to_datetime(['2020-03-15', '2020-12-15', '2021-04-01']),  # Example lockdown dates
    'lower_window': 0,
    'upper_window': 1,
})
model = model.add_country_holidays(country_name='IN')  # Add public holidays for India


# Step 3: Fit the Prophet model to historical case data
model.fit(df)

# Step 4: Make Future Predictions
# Create a dataframe for future dates
future = model.make_future_dataframe(periods=30)  # Predict the next 30 days

# Make predictions
forecast = model.predict(future)

# Step 5: Plot the forecast with a custom legend
fig = model.plot(forecast)
plt.title('COVID-19 Case Forecast (India)')
plt.xlabel('Date')
plt.ylabel('Predicted Cases')

# Add a custom legend
plt.legend(['Predicted', 'Actual Data', 'Uncertainty Interval'], loc='upper left')

plt.show()

# Step 6: Plot forecast components (Trend, Weekly Seasonality, Yearly Seasonality, Holidays)
model.plot_components(forecast)
plt.show()

# Step 7: Model Evaluation (Backtesting)
# Split the data into train and test sets (train before June 2021, test after)
train_df = df[df['ds'] < '2021-06-01']  # Train set
test_df = df[df['ds'] >= '2021-06-01']  # Test set

# Create a new Prophet model instance
model_test = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True,
    daily_seasonality=False
)

# Fit the new model on the training data
model_test.fit(train_df)

# Predict for the test period
future_test = model_test.make_future_dataframe(periods=len(test_df))
forecast_test = model_test.predict(future_test)

# Extract actual and predicted values
actual = test_df['y'].values
predicted = forecast_test['yhat'][-len(test_df):].values

# Calculate evaluation metrics (MAE and RMSE)
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

