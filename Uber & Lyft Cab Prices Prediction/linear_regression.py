# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Silence warning from sklearn
import warnings
warnings.filterwarnings('ignore')

# csv files import
cab_rides = pd.read_csv('cab_rides.csv')

# Dataframes
cab_rides_df = cab_rides[['distance','cab_type','time_stamp','destination','source','price','surge_multiplier','name']]

cab_rides_df = cab_rides_df.dropna(subset=['price'])
X = pd.get_dummies(cab_rides_df.drop(columns=['price']), drop_first=True)
y = cab_rides_df['price']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set aside 10% of instances for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Split training again into 70% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2222, random_state=42)

# Initializing Linear Regression
linRegModel = LinearRegression()

# Linear Regression fitting and training data
linRegModel.fit(X_train, np.ravel(y_train))
lr_pred = linRegModel.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

# Printing output for both models
## Linear Regression model
print("Linear Regression Test Evaluation:")
print(f"RMSE: {lr_rmse:.2f}")
print(f"R² Score: {lr_r2:.3f}")