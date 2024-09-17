import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load train and test datasets
train_data = pd.read_csv(r"C:\Users\reesh\Downloads\house-prices-advanced-regression-techniques\train.csv")
test_data = pd.read_csv(r"C:\Users\reesh\Downloads\house-prices-advanced-regression-techniques\test (2).csv")

# Display basic information about the train dataset
print(train_data.info())

# Display the first few rows of the train dataset
print(train_data.head())

# Display statistical summary of the train dataset
print(train_data.describe())

# Check for missing values
missing_values = train_data.isnull().sum()
print("Missing values in each column:\n", missing_values[missing_values > 0])

# Visualize the distribution of the target variable (assumed to be 'SalePrice')
plt.figure(figsize=(10, 6))
sns.histplot(train_data['SalePrice'], bins=30, kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# Fill missing values for numerical columns with the median
for column in train_data.select_dtypes(include=[np.number]).columns:
    train_data[column] = train_data[column].fillna(train_data[column].median())

# Fill missing values for categorical columns with the mode
for column in train_data.select_dtypes(include=[object]).columns:
    train_data[column] = train_data[column].fillna(train_data[column].mode()[0])

# Convert categorical variables into dummy/indicator variables
train_data = pd.get_dummies(train_data, drop_first=True)

# Check the shape of the processed train dataset
print("Shape of train data after preprocessing:", train_data.shape)

# Define the target variable and features (Only using relevant features)
X = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]  # Replace with actual column names for square footage, bedrooms, and bathrooms
y = train_data['SalePrice']

# Optionally, split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the split datasets
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

# Initialize the model (Linear Regression)
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')

# Process the test data (same preprocessing steps as train)
# Extract the 'Id' column before dropping it
test_data_id = test_data['Id']  # Assuming there's an 'Id' column in the test data

# Fill missing values for numerical columns with the median
for column in test_data.select_dtypes(include=[np.number]).columns:
    test_data[column] = test_data[column].fillna(test_data[column].median())

# Fill missing values for categorical columns with the mode
for column in test_data.select_dtypes(include=[object]).columns:
    test_data[column] = test_data[column].fillna(test_data[column].mode()[0])

# Convert categorical variables into dummy/indicator variables
test_data = pd.get_dummies(test_data, drop_first=True)

# Align the columns of test data with the training data
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Select relevant features for the test data
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]  # Replace with actual column names

# Make predictions on the test data
test_predictions = model.predict(X_test)

# Create a submission DataFrame with the retained 'Id' column
submission = pd.DataFrame({
    'Id': test_data_id,  # Use the extracted 'Id' column
    'SalePrice': test_predictions
})

# Save the submission DataFrame to a CSV file
submission.to_csv(r"C:\Users\reesh\Downloads\house-prices-advanced-regression-techniques\submissions.csv", index=False)

# Display the predictions
print("Predictions saved to submissions.csv")
