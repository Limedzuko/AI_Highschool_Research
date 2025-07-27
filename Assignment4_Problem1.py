from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Pune_House_Data.csv')
print(df)

print(df['area_type'].unique())
print(df['availability'].unique())
# Function to encode availability based on month name
def encode_availability(value):
    if "Ready To Move" in value or "Immediate Possession" in value:
        return 0
    elif "Jan" in value:
        return 1
    elif "Feb" in value:
        return 2
    elif "Mar" in value:
        return 3
    elif "Apr" in value:
        return 4
    elif "May" in value:
        return 5
    elif "Jun" in value:
        return 6
    elif "Jul" in value:
        return 7
    elif "Aug" in value:
        return 8
    elif "Sep" in value:
        return 9
    elif "Oct" in value:
        return 10
    elif "Nov" in value:
        return 11
    elif "Dec" in value:
        return 12
    else:
        return None  # In case the value does not match any of the above

# Apply the encoding function
df['availability'] = df['availability'].apply(encode_availability)

# View the transformed data
print(df['availability'].head())

print(df['size'].unique())
import numpy as np
import pandas as pd

# Function to extract the number of rooms from the 'size' column
def extract_rooms(size):
    if pd.isna(size):  # Check for NaN values
        return np.nan  # Keep NaN as is or handle it separately later
    else:
        # Split the string by space and take the first part, which is the number
        return int(size.split()[0])

# Apply the function to the 'size' column
df['size'] = df['size'].apply(extract_rooms)

# View the transformed data
print(df[['size']].head())
print(df['society'].unique())

df = df.drop(columns = ['society'])
print(df['site_location'].unique())
nan_columns = df.isna().any()
print(nan_columns)
nan_counts = df.isna().sum()
print(nan_counts)
df['bath'] = df['bath'].fillna(df['bath'].mean())
df['balcony'] = df['balcony'].fillna(df['balcony'].mean())
df['size'] = df['size'].fillna(df['size'].mean())
df['site_location'] = df['site_location'].fillna('Aundh')
df.isna().sum()
import numpy as np

# Code from cell pBWPbpQ7nwBI to define features (X) and target (y) and add the new feature
# Define features (X) and target variable (y)

# Drop the target column 'price' from features
X = df_encoded.drop(columns=['price']).copy() # Use .copy() to avoid SettingWithCopyWarning
y = df_encoded['price']

# Add the new feature: 1.02 raised to the power of the square root of total_sqft
X['less_exp_sqrt_total_sqft'] = 1.02 ** np.sqrt(X['total_sqft']) THIS FEATURE WAS NOT PRESENT FOR THE BEST MODEL


# Print the columns of X immediately after running the code
print("Columns in X after executing cell pBWPbpQ7nwBI's logic:")
print(X.columns)
# Function to extract and convert the value before the dash
def replace_dash(value):
    if isinstance(value, str) and '-' in value:
        return float(value.split(' - ')[0])
    try:
        return float(value)
    except ValueError:
        return None

# Apply the function to the 'total_sqft' column
df['total_sqft'] = df['total_sqft'].apply(replace_dash)
print(df.dtypes)
df['total_sqft'] = df['total_sqft'].fillna(1000)
df.isna().sum()
# Apply one-hot encoding to categorical columns
df_encoded = pd.get_dummies(df, columns=['area_type',  'site_location'])

# View the transformed data
print(df_encoded.head())

#REAL MODELLING
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Assuming `df` is your DataFrame

# Define features (X) and target variable (y)
X = df_encoded.drop(columns=['price'])  # Drop the target column 'price'
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)
accuracy = model.score(X_train, y_train)
print(f"Model Training Accuracy (R^2 Score): {accuracy:.2f}")

#BELOW IS RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # Import mean_squared_error
import pandas as pd

# Assuming X_train, X_test, y_train, and y_test are defined from the initial train/test split
# You can experiment with hyperparameters like n_estimators, max_depth, etc.
model_rf = RandomForestRegressor(n_estimators=50, random_state=42) # You can adjust n_estimators and other params here

# Train the model using the initial train/test split data
model_rf.fit(X_train, y_train) # Use X_train and y_train from the first split

# Evaluate the model on the training data
train_r2_rf = model_rf.score(X_train, y_train)
print(f'Random Forest Training R^2 Score: {train_r2_rf:.2f}')

# Evaluate the model on the testing data
test_r2_rf = model_rf.score(X_test, y_test) # Use X_test and y_test from the first split
print(f'Random Forest Testing R^2 Score: {test_r2_rf:.2f}')

# You can also make predictions and calculate other metrics like Mean Squared Error
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'Random Forest Mean Squared Error: {mse_rf:.2f}')

#BELOW IS RIDGE REGRESSION
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Adding polynomial features
poly = PolynomialFeatures(degree=2) 
X_poly = poly.fit_transform(X_scaled)

# Train/test split
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y_train, test_size=0.1, random_state=42)

# Linear Regression with Ridge regularization
model = Ridge(alpha=0.02) #this is not the optimal hyperparameter onfiguration since none of the ridge models were the best
model.fit(X_train_poly, y_train)

accuracy_train = model.score(X_train_poly, y_train)
print(f'R^2 Score training: {accuracy_train}')


# Evaluate model
r2_score = model.score(X_test_poly, y_test)
print(f'R^2 Score testing: {r2_score}')

#BELOW IS LASSO REGRESSION
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Assuming X_train, X_test, y_train, and y_test are defined from the initial train/test split 

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Apply the same scaling to test data

# Adding polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled) # Apply the same polynomial transformation to test data

# Removed the second train/test split (which was causing the error)
# X_train_poly, X_test_poly, y_train_split, y_test_split = train_test_split(X_poly, y_train, test_size=0.2, random_state=42)


# Linear Regression with Lasso regularization
# Using Lasso or Ridge - currently set to Lasso
model = Lasso(alpha=4.7)  #THIS IS WHAT GAVE THE BEST QUADRATIC REGRESSION
model.fit(X_train_poly, y_train) # Train on the polynomial features of the initial training data

accuracy_train = model.score(X_train_poly, y_train)
print(f'R^2 Score training: {accuracy_train}')


# Evaluate model on the initial test set transformed with polynomial features
r2_score_test = model.score(X_test_poly, y_test) # Evaluate on the polynomial features of the initial test data
print(f'R^2 Score testing: {r2_score_test}')

#BELOW IS GRADIENT BOOSTER REGRESSION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Assuming X_train, X_test, y_train, and y_test are already defined from your latest train/test split
# This split should ideally include the 'less_exp_sqrt_total_sqft' feature if you added it to X and re-ran the split cell.


# You can experiment with hyperparameters like n_estimators, learning_rate, max_depth, etc.
model_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
model_gbr.fit(X_train, y_train)

# Evaluate the model on the training data
train_r2_gbr = model_gbr.score(X_train, y_train)
print(f'Gradient Boosting Training R^2 Score: {train_r2_gbr:.2f}')

# Evaluate the model on the testing data
test_r2_gbr = model_gbr.score(X_test, y_test)
print(f'Gradient Boosting Testing R^2 Score: {test_r2_gbr:.2f}')

# Calculate and print Mean Squared Error
y_pred_gbr = model_gbr.predict(X_test)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
print(f'Gradient Boosting Mean Squared Error: {mse_gbr:.2f}')
