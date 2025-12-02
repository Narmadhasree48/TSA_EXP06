# TSA_EXP06
# Ex.No: 6 HOLT WINTERS METHOD

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative
trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and
Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it

### PROGRAM:
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset and select the 'close' column
data = pd.read_csv('tsla_2014_2023.csv', parse_dates=['date'], index_col='date')['close']

# Resample to Weekly End ('W')
data_weekly = data.resample('W').last()

# Plot resampled data
data_weekly.plot()
plt.title('TSLA Weekly Close Price')
plt.show()

# Scale the data
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_weekly.values.reshape(-1, 1)).flatten(),
    index=data_weekly.index
)

# Plot scaled data
scaled_data.plot()
plt.title('Scaled TSLA Weekly Close Price')
plt.show()

# Check for seasonality
# Using period=52 for yearly seasonality in weekly data
decomposition = seasonal_decompose(scaled_data.dropna(), model="additive", period=52)
decomposition.plot()
plt.show()

# Split test, train data, create a model using Holt-Winters method, train with train data and Evaluate
scaled_data = scaled_data + 1 # Add 1 to ensure positive values for multiplicative seasonality
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Use seasonal_periods=52 for weekly data
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=52
).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

# Visual evaluation
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')
plt.show()

# Evaluation Metrics
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"Test RMSE: {rmse}")
print(f"Scaled Data Std Dev: {np.sqrt(scaled_data.var())}, Scaled Data Mean: {scaled_data.mean()}")

# Create the final model and predict future data and plot it
# Use seasonal_periods=52 for weekly data
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=52
).fit()

# Predict for the next 52 weeks (next year)
final_predictions = final_model.forecast(steps=52)

# Final prediction plot
ax=scaled_data.plot()
final_predictions.plot(ax=ax)
ax.legend(["Scaled Weekly Close", "Forecast (Next 52 Weeks)"])
ax.set_xlabel('Time (Weeks)')
ax.set_ylabel('Scaled Price')
ax.set_title('Holt-Winters Final Prediction')
plt.show()
~~~

### OUTPUT:
Scaled_data plot:
<img width="1089" height="920" alt="Screenshot 2025-09-30 093731" src="https://github.com/user-attachments/assets/4383dcaf-d986-4a07-adf0-b9cff6dba7f1" />

Decomposed plot:
<img width="1252" height="956" alt="Screenshot 2025-09-30 093752" src="https://github.com/user-attachments/assets/c65eb298-355f-44e4-9e11-84eaad64fcb5" />

Test prediction:
<img width="1088" height="902" alt="Screenshot 2025-09-30 093802" src="https://github.com/user-attachments/assets/1702fdb2-8e61-4f13-a3ce-3a294e12b00a" />

Model performance metrics:
RMSE:
<img width="446" height="35" alt="Screenshot 2025-09-30 093813" src="https://github.com/user-attachments/assets/0f719807-b7dd-410d-ac9c-2cbfc1dcc923" />

Standard deviation and mean:
<img width="1182" height="32" alt="Screenshot 2025-09-30 093819" src="https://github.com/user-attachments/assets/a92bd1c4-a2ef-4a8e-b35e-1526ee920eb3" />

Final prediction:
<img width="1137" height="906" alt="Screenshot 2025-09-30 093829" src="https://github.com/user-attachments/assets/decd0cb1-3cc2-4325-8454-f1ae07231a0f" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
